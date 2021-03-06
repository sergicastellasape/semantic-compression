import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import abs_max_pooling, hotfix_pack_padded_sequence
from model.customlayers import Attention

# this class is heavily based on the one implemented in the tutorial from pytorch
# and forum https://discuss.pytorch.org/t/lstm-to-bi-lstm/12967


class BiLSTMClassifier(nn.Module):
    """Sequence classifier based on a BiLSTM architecture.
    Args:
        embedding_dim: size of the feature vectors (768 for BERT-base).
        hidden_dim: size of the hidden dimension of the network.
        num_classes: number of classes for the classifier.
        num_layers: number of stacked layers of LSTMS.
        pooling: strategy for pooling over hidden states to obtain a single
            feature vector to classify.
        task: name of the task (string) to save as an attribute to the object.
        bidirectional: boolean which if True, implements bidirectional LSTM,
            and if False, falls back to a traditional left-to-right.
        mask_special_tokens: boolean which if True masks not only padding but
            also special tokens such as <cls> and <sep> from the transformer
            encoder.
        dropout: ratio of dropout for inner layers during training; helps in
            preventing overfitting. As a rule of thumb, 0.3 is a good ratio to
            begin with.
        device: `torch.device` to use, for cpu or cuda.
    """
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        num_classes,
        num_layers,
        pooling=None,
        task=None,
        bidirectional=True,
        mask_special_tokens=True,
        dropout=0.0,
        device=torch.device("cpu"),
    ):
        super(BiLSTMClassifier, self).__init__()
        assert task is not None
        self.task = task
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pooling = pooling
        self.mask_special_tokens = mask_special_tokens
        self.directions = 2 if bidirectional else 1
        if pooling not in ['mean_pooling', 'abs_max_pooling']:
            raise Warning("Concatenation Pooling for BiLSTM Classifier is not \
                tested, and only implemented for bidirectional LSTM. \
                Please Debug before using!")

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            bidirectional=bidirectional,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bias=True,
        )

        # attend to LSTM outputs w/ sentiment queries
        # self.attend = nlpnn.Attention(embedding_dim, attention_type='dot')

        # The linear layer that maps from hidden state space to sentiment classification space
        self.hidden2sent = nn.Linear(hidden_dim * self.directions, num_classes)
        self.loss_fn = nn.NLLLoss(reduction="mean")

    def init_hidden(self, batch_size):
        # As of the documentation from nn.LSTM in pytorch, the input to the lstm cell is
        # the input and a tuple of (h, c) hidden state and memory state. We initialize that
        # tuple with the proper shape: num_layers*directions, batch_size, hidden_dim. Don't worry
        # that the batch here is second, this is dealt with internally if the lstm is created with
        # batch_first=True
        shape = (self.num_layers * self.directions, batch_size, self.hidden_dim)
        return (
            torch.zeros(shape, requires_grad=True, device=self.device),
            torch.zeros(shape, requires_grad=True, device=self.device),
        )

    def loss(self, predicted, target):
        """Calls the loss function defined in __init__(), which is a
        Negative Log Likelihood Loss, equivalend to a cross entorpy loss, but
        a previous log softmax activation. See PyTorch documentation for further
        details.
        """
        return self.loss_fn(predicted, target)

    def forward(self, inp, masks_dict=None, **kwargs):
        """Performs forward pass of the BiLSTM for a batch.
        Args:
            inp: `torch.tensor` of size (batch, max_length, embedding_dim),
                which is padded to the right to the sequence of maximum length.
            masks_dict: dictionary of masks (`torch.tensor` of uint8) for the
                output with keys `padding_mask`, `seq_pair_mask` and
                `regular_tokens_mask`.
            **kwargs: just as a safeguard if a generic network call to trigger
                this method, which uses other keyword arguments.
        Returns:
            log_scores: `torch.tensor` of size (batch, num_classes) with each
                class' log score, to be later fed into a NLL loss.
        """
        assert masks_dict is not None

        if self.mask_special_tokens:
            inp *= masks_dict['regular_tokens_mask'].unsqueeze(-1)

        # Calculate original lengths
        lengths = masks_dict['padding_mask'].sum(dim=1)

        packed_tensors = hotfix_pack_padded_sequence(
            inp, lengths, enforce_sorted=False, batch_first=True
        )
        # detach to make the computation graph for the backward pass only for 1 sequence
        hidden, cell = self.init_hidden(inp.size(0))  # feed with batch_size
        lstm_out, (hidden_out, cell_out) = self.lstm(packed_tensors,
                                                     (hidden, cell))

        # we unpack and use the last lstm output for classification
        unpacked_output = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True
        )[0]
        #print(lengths - 1)
        #print(unpacked_output[:, lengths - 2, :].size())
        # max pooling input needs to be (batch, seq_lengh, embedding)
        if self.pooling == "abs_max_pooling":
            sent_space = self.hidden2sent(abs_max_pooling(unpacked_output, dim=1))
        elif self.pooling == "mean_pooling":
            sent_space = self.hidden2sent(unpacked_output.mean(dim=1)) # need to correct for padding!
        else:
            right2left = unpacked_output[:, 1, :]
            bool_mask = F.one_hot(lengths - 2, num_classes=lengths.max()).unsqueeze(-1)
            left2right = (unpacked_output * bool_mask).sum(dim=1)
            sent_space = self.hidden2sent(torch.cat([right2left, left2right], dim=1))

        log_scores = F.log_softmax(sent_space, dim=1)

        return log_scores


class AttentionClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_classes,
        dropout=0.0,
        n_sentiments=2,
        task=None,
        pool_mode="concat",
        device=torch.device("cpu"),
    ):
        super(AttentionClassifier, self).__init__()
        assert task is not None
        self.task = task
        self.device = device
        self.pool_mode = pool_mode

        # Define attention layers:
        self.self_attend = Attention(embedding_dim, attention_type="general", device=device)  # .to(device)

        self.attend = Attention(embedding_dim, attention_type="dot", device=device)  # .to(device)

        # The linear layer that maps from embedding state space to sentiment classification space
        if self.pool_mode == "concat":
            self.classifier = nn.Linear(embedding_dim * n_sentiments, num_classes).to(device)
        else:
            self.classifier = nn.Linear(embedding_dim, num_classes).to(device)

        # Loss function as negative log likelihood, which needs a logsoftmax input
        self.loss_fn = nn.NLLLoss(reduction="mean")

        # Initialize the vectors that represent sentiments characteristics,
        # and add them as model parameters so that they're trained by backprop
        init_normal = torch.empty(1, n_sentiments, embedding_dim).normal_(
            mean=0, std=0.3
        )
        self.sentiment_queries = nn.Parameter(
            init_normal.clone().detach().requires_grad_(True).to(device)
        )

    def loss(self, prediction, target):
        return self.loss_fn(prediction, target)

    def forward(self, inp, pooling=None, masks_dict=None, **kwargs):
        assert masks_dict is not None
        # concatenate along sequence length dimension so attention is calculated
        # along both positive and negative sentiments
        query = self.sentiment_queries.repeat(
            inp.size(0), 1, 1
        )  # expand for batch size

        # sequence is the input
        self_attended_context, _ = self.self_attend(inp, inp)

        # self_attended_context = self.feedforward(self_attended_context)

        attention_seq, _ = self.attend(query, self_attended_context)

        if self.pool_mode == "concat":
            sentiment = self.classifier(attention_seq.flatten(1))
        elif self.pool_mode == "mean_pooling":
            sentiment = self.classifier(attention_seq.mean(1))
        elif self.pool_mode == "max_pooling":
            sentiment = self.classifier(abs_max_pooling(attention_seq, dim=1))

        sentiment_logscore = F.log_softmax(sentiment, dim=1)
        return sentiment_logscore


class ConvAttClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_classes,
        dropout=0.3,
        n_attention_vecs=3,
        task=None,
        mask_special_tokens=True,
        device=torch.device("cpu"),
    ):
        super().__init__()
        assert task is not None
        self.task = task
        self.mask_special_tokens = mask_special_tokens
        self.device = device

        # Define attention layers:
        self.conv1D = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=5, padding=2)  # padding=(kernel-1)/2
        self.skip_weight = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.attend = Attention(embedding_dim, attention_type="general", device=device)  # .to(device)
        self.classifier = nn.Linear(n_attention_vecs * embedding_dim, num_classes)

        # Loss function as negative log likelihood, which needs a logsoftmax input
        self.loss_fn = nn.NLLLoss(reduction="mean")

        # Initialize the vectors that represent sentiments characteristics,
        # and add them as model parameters so that they're trained by backprop
        init_normal = torch.empty(1, n_attention_vecs, embedding_dim).normal_(
            mean=0, std=0.3
        )
        self.attention_vecs = nn.Parameter(
            init_normal.clone().detach().requires_grad_(True).to(device)
        )

    def loss(self, prediction, target):
        return self.loss_fn(prediction, target)

    def forward(self, inp, pooling=None, masks_dict=None, **kwargs):
        assert masks_dict is not None
        if not self.mask_special_tokens:
            mask = (masks_dict['padding_mask']).unsqueeze(-1).expand(inp.size()).to(self.device)
        else:
            mask = (masks_dict['regular_tokens_mask']).unsqueeze(-1).expand(inp.size()).to(self.device)
        inp *= mask

        # concatenate along sequence length dimension so attention is calculated
        # along both positive and negative sentiments
        query = self.attention_vecs.repeat(
            inp.size(0), 1, 1
        )  # expand for batch size

        # sequence is the input, transpose is to adapt for conv layer input convention
        conv_out = self.conv1D(inp.transpose(1, 2)).transpose(1, 2) * mask
        mix = (1 - self.skip_weight) * inp + self.skip_weight * conv_out
        attention_seq, _ = self.attend(query, mix, context_mask=mask)

        class_score = self.classifier(attention_seq.flatten(1))
        class_logscore = F.log_softmax(class_score, dim=1)

        return class_logscore

class DecAttClassifiter(nn.Module):
    """
    Seq pair classifier based on: "A Decomposable Attention Model for Natural Language Inference"
    It's basically attention between the two sentences, aggregation and classification.
    https://arxiv.org/pdf/1606.01933.pdf
    Args:
        embedding_dim: size of the feature vectors (768 for BERT-base).
        num_classes: number of classes for the classifier.
        pool_func: *function object* for pooling over hidden states to obtain a
            single feature vector to classify.
        task: name of the task (string) to save as an attribute to the object.
        bidirectional: boolean which if True, implements bidirectional LSTM,
            and if False, falls back to a traditional left-to-right.
        mask_special_tokens: boolean which if True masks not only padding but
            also special tokens such as <cls> and <sep> from the transformer
            encoder.
        device: `torch.device` to use, for cpu or cuda.
    """
    def __init__(self,
                 embedding_dim,
                 num_classes,
                 dropout=0.3,
                 task=None,
                 pool_func=abs_max_pooling,
                 mask_special_tokens=True,
                 device=torch.device('cpu')):
        super().__init__()
        assert task is not None
        self.task = task
        self.mask_special_tokens = mask_special_tokens
        self.device = device
        self.pool_func = pool_func

        # Define 1 -> 2 attention layer
        self.att12 = Attention(embedding_dim, attention_type="general", device=device)  # .to(device)
        self.att21 = Attention(embedding_dim, attention_type="general", device=device)  # .to(device)
        self.classifier = nn.Linear(embedding_dim * 2, num_classes).to(device)

        # Cross entropy loss function that's fed with the log_scores (numerical stability)
        self.loss_fn = nn.NLLLoss(reduction="mean")

    def loss(self, prediction, target):
        """Calls the loss function defined in __init__(), which is a
        Negative Log Likelihood Loss, equivalend to a cross entorpy loss, but
        a previous log softmax activation. See PyTorch documentation for further
        details.
        """
        return self.loss_fn(prediction, target)

    def forward(self, inp, masks_dict=None, **kwargs):
        """Performs forward pass of the DecAttClassifier for a batch.
        Args:
            inp: `torch.tensor` of size (batch, max_length, embedding_dim),
                which is padded to the right to the sequence of maximum length.
            masks_dict: dictionary of masks (`torch.tensor` of uint8) for the
                output with keys `padding_mask`, `seq_pair_mask` and
                `regular_tokens_mask`.
            **kwargs: just as a safeguard if a generic network call to trigger
                this method, which uses other keyword arguments.
        Returns:
            log_scores: `torch.tensor` of size (batch, num_classes) with each
                class' log score, to be later fed into a NLL loss.
        """
        assert masks_dict is not None

        mask_2 = (masks_dict['seq_pair_mask'] == 1)
        mask_1 = (masks_dict['seq_pair_mask'] == 0)
        if self.mask_special_tokens:  # remove cls and sep tokens basically
            mask_2 *= (masks_dict['regular_tokens_mask'] == 1)
            mask_1 *= (masks_dict['regular_tokens_mask'] == 1)
        mask_2 = mask_2.unsqueeze(-1).expand(inp.size()).to(self.device)
        mask_1 = mask_1.unsqueeze(-1).expand(inp.size()).to(self.device)

        inp_tensor1 = inp * mask_1
        inp_tensor2 = inp * mask_2

        att_seq1 = self.att12(inp_tensor1,
                              inp_tensor2,
                              query_mask=mask_1,
                              context_mask=mask_2)[0] * mask_1
        att_seq2 = self.att21(inp_tensor2,
                              inp_tensor1,
                              query_mask=mask_2,
                              context_mask=mask_1)[0] * mask_2

        # Watch out! if you do mean pooling, the padding might give problems!
        aggregation_seq1 = self.pool_func(att_seq1, dim=1)
        aggregation_seq2 = self.pool_func(att_seq2, dim=1)

        classifier_features = torch.cat([aggregation_seq1, aggregation_seq2], dim=1)
        class_score = self.classifier(classifier_features)
        class_log_score = F.log_softmax(class_score, dim=1)

        return class_log_score


class DecAttClassifiter_v2(nn.Module):
    """
    Seq pair classifier based on: "A Decomposable Attention Model for Natural Language Inference"
    It's basically attention between the two sentences, aggregation and classification.
    https://arxiv.org/pdf/1606.01933.pdf
    The difference between v2 and original is that this implements pytorch's
    generig multi-headed attention instead of single one. Results did not seem
    to improve by this change, so we did not end up using it.
    """
    def __init__(self,
                 embedding_dim,
                 num_classes,
                 num_heads=4,
                 dropout=0.3,
                 task=None,
                 pool_func=abs_max_pooling,
                 mask_special_tokens=True,
                 device=torch.device('cpu')):
        super().__init__()
        assert task is not None
        self.task = task
        self.mask_special_tokens = mask_special_tokens
        self.device = device
        self.pool_func = pool_func

        # Define 1 -> 2 attention layer
        self.att12 = nn.MultiheadAttention(embedding_dim,
                                           num_heads,
                                           dropout=dropout,
                                           bias=True,
                                           add_bias_kv=False,
                                           add_zero_attn=False,
                                           kdim=None,
                                           vdim=None).to(device)
        self.att21 = nn.MultiheadAttention(embedding_dim,
                                           num_heads,
                                           dropout=dropout,
                                           bias=True,
                                           add_bias_kv=False,
                                           add_zero_attn=False,
                                           kdim=None,
                                           vdim=None).to(device)
        self.classifier = nn.Linear(embedding_dim * 2, num_classes).to(device)

        # Cross entropy loss function that's fed with the log_scores (numerical stability)
        self.loss_fn = nn.NLLLoss(reduction="mean")

    def loss(self, prediction, target):
        return self.loss_fn(prediction, target)

    def forward(self, inp, masks_dict=None, **kwargs):
        assert masks_dict is not None

        mask_2 = (masks_dict['seq_pair_mask'] == 1)
        mask_1 = (masks_dict['seq_pair_mask'] == 0)
        if self.mask_special_tokens:  # remove cls and sep tokens basically
            mask_2 *= (masks_dict['regular_tokens_mask'] == 1)
            mask_1 *= (masks_dict['regular_tokens_mask'] == 1)

        inp_tensor1 = inp * mask_1.unsqueeze(-1)
        inp_tensor2 = inp * mask_2.unsqueeze(-1)

        # transpose so batch is second instead of first
        att_seq1 = self.att12(inp_tensor1.transpose(0, 1),
                              inp_tensor2.transpose(0, 1),
                              inp_tensor2.transpose(0, 1))[0].transpose(0, 1)
        att_seq2 = self.att21(inp_tensor2.transpose(0, 1),
                              inp_tensor1.transpose(0, 1),
                              inp_tensor1.transpose(0, 1))[0].transpose(0, 1)

        # Watch out! if you do mean pooling, the padding might give problems!
        aggregation_seq1 = self.pool_func(att_seq1, dim=1)
        aggregation_seq2 = self.pool_func(att_seq2, dim=1)

        classifier_features = torch.cat([aggregation_seq1, aggregation_seq2], dim=1)
        class_score = self.classifier(classifier_features)
        class_log_score = F.log_softmax(class_score, dim=1)

        return class_log_score


class SeqPairAttentionClassifier(nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_classes,
                 dropout=0.0,
                 n_attention_vecs=4,
                 task=None,
                 pool_mode="concat",
                 device=torch.device("cpu")):
        super(SeqPairAttentionClassifier, self).__init__()
        assert task is not None
        raise Warning("This module could be not properly designed, contain bugs,\
            so it should be used with caution!")
        self.task = task
        self.device = device
        self.pool_mode = pool_mode

        # Define attention layers:
        self.self_attend = Attention(embedding_dim, attention_type="general", device=device)  # .to(device)

        self.attend = Attention(embedding_dim, attention_type="dot", device=device)  # .to(device)

        # The linear layer that maps from embedding state space to sentiment classification space
        layer_multiplier = n_attention_vecs if self.pool_mode == "concat" else 1
        self.classifier = nn.Linear(
            embedding_dim * layer_multiplier, num_classes
        ).to(device)

        # Loss function as negative log likelihood, which needs a logsoftmax input
        self.loss_fn = nn.NLLLoss(reduction="mean")

        # initialize the vectors that represent attention characteristics,
        # and add them as model parameters so that they're trained by backprop
        init_normal1 = torch.empty(1, n_attention_vecs, embedding_dim).normal_(
            mean=0, std=0.3
        )
        init_normal2 = torch.empty(1, n_attention_vecs, embedding_dim).normal_(
            mean=0, std=0.3
        )
        self.attention_vecs1 = nn.Parameter(
            init_normal1.clone().detach().requires_grad_(True).to(device)
        )
        self.attention_vecs2 = nn.Parameter(
            init_normal2.clone().detach().requires_grad_(True).to(device)
        )

    def loss(self, prediction, target):
        return self.loss_fn(prediction, target)

    def forward(self, inp, masks_dict=None, **kwargs):
        """
        Input_tup should be a tuple of two tensors, containing the batches
        """
        seq_pair_mask = masks_dict['seq_pair_mask']

        # seq_pair_mask looks like [00000000001111111111] so for the second sentence
        # one needs to multiply by the mask and for the first one it's the negation
        mask_2 = (seq_pair_mask == 1).unsqueeze(-1).expand(inp.size()).to(self.device)
        mask_1 = (seq_pair_mask == 0).unsqueeze(-1).expand(inp.size()).to(self.device)

        inp_tensor1 = inp * mask_1
        inp_tensor2 = inp * mask_2

        # concatenate along sequence length dimension so attention is calculated
        # along both positive and negative sentiments
        query1 = self.attention_vecs1.repeat(inp_tensor1.size(0), 1, 1)  # expand for batch size
        query2 = self.attention_vecs2.repeat(inp_tensor2.size(0), 1, 1)

        # sequence is the input
        self_attended1, _ = self.self_attend(inp_tensor1, inp_tensor1)
        self_attended2, _ = self.self_attend(inp_tensor2, inp_tensor2)

        # attention in between two sentences
        combined_seq, _ = self.self_attend(inp_tensor1, inp_tensor2)

        # attention to the attention_vecs
        attention_seq, _ = self.attend(query1, combined_seq)

        if self.pool_mode == "concat":
            class_score = self.classifier(attention_seq.flatten(1))
        elif self.pool_mode == "mean_pooling":
            class_score = self.classifier(attention_seq.mean(1))
        elif self.pool_mode == "max_pooling":
            class_score = self.classifier(abs_max_pooling(attention_seq, dim=1))

        class_logscore = F.log_softmax(class_score, dim=1)
        return class_logscore


class NaivePoolingClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_classes,
        task=None,
        dropout=0.0,
        pool_mode="max_pooling",
        device=torch.device("cpu"),
    ):
        super().__init__()
        assert task is not None

        self.task = task
        self.device = device
        self.pre_pooling_linear = nn.Linear(embedding_dim, embedding_dim, bias=True)
        # network
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
        # loss func
        self.loss_fn = nn.NLLLoss(reduction="mean")

    def loss(self, prediction, target):
        return self.loss_fn(prediction, target)

    def forward(self, input, **kwargs):
        logging.debug('Input Classifier size', input.size())
        # input is size (batch, max_seq_length, embedding_dim)
        inp = self.pre_pooling_linear(input)
        pooled_features = abs_max_pooling(inp, dim=1)
        class_score = self.classifier(pooled_features)
        class_log_score = self.log_softmax(class_score)
        return class_log_score


class SeqPairFancyClassifier(nn.Module):
    """Network only used for toy experiments and testing. Do not use unless
    """
    def __init__(
        self,
        embedding_dim,
        num_classes,
        task=None,
        dropout=0.0,
        n_attention_vecs=4,
        device=torch.device("cpu"),
    ):
        super(SeqPairFancyClassifier, self).__init__()
        assert task is not None
        raise Warning("This module could be not properly designed, contain bugs,\
            so it should be used with caution!")
        self.task = task
        self.device = device

        # Define attention layers:
        self.self_attend = Attention(embedding_dim, attention_type="general", device=device)  # .to(device)

        self.attend = Attention(embedding_dim, attention_type="dot", device=device)  # .to(device)

        # Convolutional layer to filter attention weights
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=4, stride=1, padding=(0, 0)
        )
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.conv2 = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=5, stride=3, padding=(0, 0)
        )

        # The linear layer that maps from embedding state space to sentiment classification space
        self.seq_classifier = nn.Linear(
            embedding_dim * n_attention_vecs, num_classes
        ).to(device)
        self.weights_classifier = nn.Linear(
            486, num_classes
        )  # size of the conv. features
        self.join_classifiers = nn.Linear(num_classes * 2, num_classes)

        # Loss function as negative log likelihood, which needs a logsoftmax input
        self.loss_fn = nn.NLLLoss(reduction="mean")

        # initialize the vectors that represent attention characteristics,
        # and add them as model parameters so that they're trained by backprop
        init_normal1 = torch.empty(1, n_attention_vecs, embedding_dim).normal_(
            mean=0, std=0.3
        )
        init_normal2 = torch.empty(1, n_attention_vecs, embedding_dim).normal_(
            mean=0, std=0.3
        )
        self.attention_vecs1 = nn.Parameter(
            init_normal1.clone().detach().requires_grad_(True).to(device)
        )
        self.attention_vecs2 = nn.Parameter(
            init_normal2.clone().detach().requires_grad_(True).to(device)
        )

    def loss(self, prediction, target):
        return self.loss_fn(prediction, target)

    def forward(self, input, masks_dict=None, **kwargs):
        """
        Input_tup should be a tuple of two tensors, containing the batches
        """
        assert masks_dict is not None
        seq_pair_mask = masks_dict['seq_pair_mask']
        batch_size = input.size(0)
        # seq_pair_mask looks like [00000000001111111111] so for the second
        # sentence one needs to multiply by the mask and for the first one it's
        # the negation
        mask_2 = (seq_pair_mask == 1).unsqueeze(-1).expand(input.size()).to(self.device)
        mask_1 = (seq_pair_mask == 0).unsqueeze(-1).expand(input.size()).to(self.device)

        inp_tensor1 = input * mask_1
        inp_tensor2 = input * mask_2

        # concatenate along sequence length dimension so attention is calculated
        # along both positive and negative sentiments
        query1 = self.attention_vecs1.repeat(inp_tensor1.size(0), 1, 1)  # expand for batch size
        query2 = self.attention_vecs2.repeat(inp_tensor2.size(0), 1, 1)

        # attention in between two sentences. att_weights size (batch, seq1, seq2)
        combined_seq, att_weights = self.self_attend(inp_tensor1, inp_tensor2)

        convoluted_weights = self.conv1(
            att_weights.unsqueeze(1)
        )  # batch, channels, height, width
        # interpolate to known shape of 30 by 30
        weights3x30x30 = F.interpolate(
            convoluted_weights, size=(30, 30), mode="bilinear", align_corners=True
        )

        # further extract convolutional features and make them deeper
        weights6x10x10 = self.conv2(self.leakyrelu(weights3x30x30))  # batch, 6, 10, 10

        # attention to the attention_vecs
        attention_seq, _ = self.attend(query1, combined_seq)

        class_score_seq = self.seq_classifier(attention_seq.flatten(1))
        class_score_weights = self.weights_classifier(
            weights6x10x10.view(batch_size, -1)
        )
        score_cat = torch.cat([class_score_seq, class_score_weights], dim=1)
        joint_score = self.join_classifiers(score_cat)
        class_logscore = F.log_softmax(joint_score, dim=1)

        return class_logscore
