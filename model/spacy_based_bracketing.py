"""
Pre processing based on the general spacy library and the spacy-transformers
"""

from tqdm import tqdm
import cupy
from model.utils import add_space_to_special_characters, expand_indices, txt2list


class ClassicPipeline:
    def __init__(
        self, pipeline=[], language="en_core_web_sm", special_characters=["/", "|", "_"]
    ):
        """
        Initialize the pipeline.
        """
        spacy.prefer_gpu()
        self.special_characters = special_characters
        self.nlp = spacy.load(language)
        for pipe in pipeline:
            P = self.nlp.create_pipe(pipe)
            self.nlp.add_pipe(P)

    def compact_tokens(self, sequences):
        docs = self.make_docs(sequences)
        batch_base_tokenization = self.make_base_tokenization(docs)
        batch_noun_chunks = self.make_noun_phrase_chunks(docs)
        bracketed_tokenization = self.compact_bracketing(
            batch_base_tokenization, batch_noun_chunks
        )
        chunk2spacy_idx = self.chunk2spacy_indices(bracketed_tokenization)

        return bracketed_tokenization, chunk2spacy_idx

    def make_docs(self, sequences_batch):
        """
        Convert list of sentences (strings) into list of spacy docs
        """
        docs = list(self.nlp.pipe(sequences_batch))
        return docs

    def make_base_tokenization(self, docs):
        """
        Get the tokenization from docs.
        INPUT
            - docs: list of doc objects from spacy
        OUTPUT
            - base_tokenization: list of lists of strings (batch[sequence][token])
        """
        base_tokenization = []
        for doc in docs:
            for token in doc:
                token_list = [token.text.lower() for token in doc]
            base_tokenization.append(token_list)
        return base_tokenization  # list of lists of strings

    def make_noun_phrase_chunks(self, docs):
        """
        Get noun chunks from docs.
        INPUT
            - docs: list of doc objecs form spaCy
        OUTPUT
            - noun_chunks: list of lists of tuples of strings (batch[sequence][chunk][token in chunk])
        """
        noun_chunks = []
        for doc in docs:
            chunks = [str(chunk).lower() for chunk in doc.noun_chunks]
            chunk_list_list = []
            for chunk in chunks:
                # Add space before and after special characters:
                # special_characters=['/']: "rnn/lstm" -> "rnn / lstm"
                chunk = add_space_to_special_characters(
                    chunk, characters=self.special_characters
                )
                # Add space before a ' character for correct splitting matching spaCy:
                # layman's -> layman 's
                chunk = chunk.replace("'", " '")
                chunk_list_list.append(tuple(chunk.split()))
            noun_chunks.append(chunk_list_list)
        return noun_chunks  # lists of lists of tuples of strings

    def compact_bracketing(self, batch_base_tokenization, batch_noun_chunks):
        """
        Given the base tokenization and a list of chunks to compact, generate the "bracketing" around
        the base tokenization that groups the chunks.
        INPUT
            - batch_noun_chunks: * list of sequences in the batch
                                   * list of noun chunks in the sequence
                                     * tuple of the tokens (strings) forming each noun chunk

            - batch_base_tokenization: * list of sequences in the batch
                                         * list of tokens (strings) in the sequence
        OUTPUT
            - base_tokenization_brackets: list of tuples of strings
                                          (tuples of 1 element if they belong to no chunk)
        """

        base_tokenization_brackets = []

        for base_tokenization, sequence_chunks in zip(
            batch_base_tokenization, batch_noun_chunks
        ):
            compact_representation = sequence_chunks.copy()
            original_pos = 0
            max_pos = len(base_tokenization) - 1
            new_pos = 0

            for chunk in sequence_chunks:
                finished = False
                while not finished:
                    try:
                        token = base_tokenization[original_pos]
                    except:
                        print("Sentence: ", base_tokenization)
                        print("original_pos: ", original_pos)
                        raise ValueError("L'index se t'ha anat... :( ", original_pos)

                    if token not in chunk:
                        compact_representation.insert(new_pos, (token,))
                        original_pos += 1
                        new_pos += 1
                        if original_pos > max_pos:
                            finished = True

                    elif token == chunk[-1]:
                        original_pos += 1
                        new_pos += 1
                        finished = True

                    else:  # (token in chunk)
                        original_pos += 1
                        if original_pos > max_pos:
                            finished = True

                if original_pos > max_pos:
                    break

            # Add everything after last chunk
            remaining_tokens = base_tokenization[original_pos:]
            for token in remaining_tokens:
                compact_representation.append((token,))

            base_tokenization_brackets.append(compact_representation)

        return base_tokenization_brackets

    def chunk2spacy_indices(self, batch_sequence_bracketed):
        """
        Given the "bracketed sequence" computes the index correspondence between the original
        tokenization and the new compacted one.
        INPUT
            - batch_sequence_bracketed: list of lists of str/tuples (output of compact bracketing)
        """
        indices = []
        for sequence in batch_sequence_bracketed:
            seq_indices = []
            i = 0
            for item in sequence:
                if len(item) > 1:
                    L = len(item)
                    seq_indices.append(tuple(range(i, i + L)))
                    i += L
                else:
                    seq_indices.append((i,))
                    i += 1
            indices.append(seq_indices)
        return indices


class TransformerPipeline:
    def __init__(
        self, language="en_trf_bertbaseuncased_lg", device=torch.device("cpu")
    ):
        """
        Initialize the pipeline with the transformer.
        """
        spacy.prefer_gpu()
        self.trf_nlp = spacy.load(language)
        self.device = device

    def make_docs(self, sequences_batch):
        """
        Convert list of sentences (strings) into list of spacy docs
        """
        docs = list(self.trf_nlp.pipe(sequences_batch))
        return docs

    def spacy2trf_indices(self, docs):
        spacy2trf_idx = []
        for doc in docs:
            spacy2trf_idx.append(doc._.trf_alignment)
        return spacy2trf_idx

    def chunk2trf_indices(self, chunk2spacy_idx_batch, spacy2trf_idx_batch):
        chunk2trf_idx_batch = []
        for chunk2spacy_idx, spacy2trf_idx in zip(
            chunk2spacy_idx_batch, spacy2trf_idx_batch
        ):
            chunk2trf_idx = []
            for tup in chunk2spacy_idx:
                c2t = []
                for idx in tup:
                    # THIS 'TRY' IS UGLY CHEATING, I SHOULDNT BE DOING THIS. Indeed...
                    # it gave problems in the end hahha
                    try:
                        c2t.extend(spacy2trf_idx[idx])
                    except:
                        pass
                if c2t:  # make sure we don't append an empty c2t
                    chunk2trf_idx.append(tuple(c2t))

            chunk2trf_idx_batch.append(chunk2trf_idx)

        return chunk2trf_idx_batch

    def get_batch_tensor(self, docs):
        list_of_tensors = []
        for doc in docs:
            trf_embeddings = doc._.trf_last_hidden_state
            # print("Type of the last_hidden: ", type(doc._.trf_last_hidden_state))
            # print("Type of the all_hidden: ", type(doc._.trf_all_hidden_states))
            # print("Are they the same?: ", doc._.trf_all_hidden_states[-1] == trf_embeddings)
            # raise ValueError("The assertion was right!")
            if isinstance(trf_embeddings, cupy.core.core.ndarray):
                # The .get() method moves the doc._.... from gpu to cpu
                list_of_tensors.append(torch.Tensor(trf_embeddings.get()))
            else:
                list_of_tensors.append(torch.Tensor(trf_embeddings))
        return list_of_tensors


class PreProcess:
    def __init__(
        self,
        classic_pipeline=ClassicPipeline(),
        transformer_pipeline=TransformerPipeline(),
    ):
        spacy.prefer_gpu()
        self.classic_pipeline = classic_pipeline
        self.transformer_pipeline = transformer_pipeline

    def load_text(self, txt_path=None):
        """
        Load txt file and split the sentences into a list of strings
        """
        if txt_path is None:
            raise ValueError(
                "txt_path must be specified as a named argument! \
            E.g. txt_path=../dataset/yourfile.txt"
            )

        # Read input sequences from .txt file and put them in a list
        with open(txt_path) as f:
            text = f.read()
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
        try:
            sentences.remove("")  # remove possible empty strings
        except:
            None

        return sentences

    def make_docs(self, sequence_batch):
        classic_docs = self.classic_pipeline.make_docs(sequence_batch)
        transformer_docs = self.transformer_pipeline.make_docs(sequence_batch)
        return classic_docs, transformer_docs

    def forward(self, sequence_batch):

        classic_docs, transformer_docs = self.make_docs(sequence_batch)

        # Compute chunks2spacy tokenization index mapping
        batch_base_tokenization = self.classic_pipeline.make_base_tokenization(
            classic_docs
        )
        batch_noun_chunks = self.classic_pipeline.make_noun_phrase_chunks(classic_docs)
        bracketed_tokenization = self.classic_pipeline.compact_bracketing(
            batch_base_tokenization, batch_noun_chunks
        )
        chunk2spacy_idx = self.classic_pipeline.chunk2spacy_indices(
            bracketed_tokenization
        )

        # Compute spacy2transformer tokenization index mapping
        spacy2trf_idx = self.transformer_pipeline.spacy2trf_indices(transformer_docs)
        target_lengths = [len(doc._.trf_word_pieces) for doc in transformer_docs]

        chunk2trf_idx = self.transformer_pipeline.chunk2trf_indices(
            chunk2spacy_idx, spacy2trf_idx
        )
        chunk2trf_idx = expand_indices(chunk2trf_idx, target_lengths)

        list_of_tensors = self.transformer_pipeline.get_batch_tensor(transformer_docs)
        tensors_batch = torch.nn.utils.rnn.pad_sequence(
            list_of_tensors, batch_first=True
        )

        return tensors_batch, chunk2trf_idx

    def trf_forward(self, sequence_batch):
        """Transformer only forward pass"""
        transformer_docs = self.transformer_pipeline.make_docs(sequence_batch)
        list_of_tensors = self.transformer_pipeline.get_batch_tensor(transformer_docs)
        tensors_batch = torch.nn.utils.rnn.pad_sequence(
            list_of_tensors, batch_first=True
        )

        return tensors_batch
