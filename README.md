# Introduction
Welcome to the code for compressing semantic representations and evaluating them in downstream tasks. If something is not explained here, the code is heavily commented, so the answer should be there!

# Table of Contents
1. [Model Architecture](#🏰-Model-Architecture)
2. [Code Architecture](#📁-Code-Architecture)
3. [Examples](#Examples) 

# 🏰 Model Architecture
The high level architecture of the model is shown in the following figure. Its input can be either a batch of sequences or sequence pairs (List of strings or tuples of strings). The input can be mixed, containing both sequences and sequence pairs. The output of the full model will be the predictions for the tasks that are being performed in the form of log-logits. Also, a custom model can be built using only sub-parts like a Lego.

![picture](https://user-images.githubusercontent.com/33417180/86136935-f2656180-baec-11ea-977b-0164fa000af9.png)

# 📁 Code Architecture
## `config/`
Contains configuration files for the datasets (and possible future additions).
- `datasets.yml`: yml configuration file for datasets paths, batch sizes, choice of classifier for each task, etc. Here are some important details.
  - `get_batch_fn` is the instruction to load a function that should be in `data_utils.py`.
  - `classifier` is a string that identifies the classifier to use for that dataset/task. It **must** use the names specified in the `model/builder/classifiers.py` dictionary, given that it will use that dictionary to load the *make* function.
## `assets/`
- `checkpoints/`: directory where the checkpoints from training will be saved under the name of `--run-identifier` passed as a command line argument.
- `datasets/`: directory to save the datasets used under subdirectories and as a `.tsv` file for train, test and ev sets. Each dataset path must be specified in `config/datasets/yml` file, so any other path will also work.
## `pre-trainings/`
Store any Transformer pre-training checkpoints from Huggingface here.

## `model/`
The core of the project lives here. The file system is split accordingly to the architecture of the model, where each part is a submodel (instance of `nn.Module` which means that it's treated as a Neural Network by default). Under this structure, to run any of this modules one must call the `forward()` pass like any other NN. `utils.py` and `data_utils.py` contain generic auxiliary functions that are used in other parts of the model.
- `transformers.py`: wrapper around the Huggingface Transformer model. It implements `batch_encode_plus` with the addition of returning a mask for the special tokens, which, at the time of development was a lacking feature from the original Transformers library. A part from the contextualized embeddings, it also returns a dictionary of masks (`torch.tensors` of `dtype=int8`) containing:
  - `padding_mask`: 1s for non-padding embeddings, 0s of padding embeddings.
  - `regular_tokens_mask`: 1s for regular tokens 0s for the rest (padding or special tokens such as `<cls>` or `<sep>`)
  - `seq_pair_mask`: 1s for tokens belonging to the first sequence, 0s to the tokens belonging to the second sequence, -1s for the padding tokens. In case the input is a single sequence and not a sequence pair, there will be no 0s.
- `bracketing.py`: file containing **Bracketing** classes. As mentioned before, they're all instances of `nn.Module` to facilitate integration within the whole model in case this step is parametrized. The bracketing module takes as input the contextualized embedding, along with the masks dictionary, and returns a list of lists of tuples, containing the indices that will be compressed. For instance, for a batch of two sentences of length 5 and 6 respectively, it could return something like: `[[(0), (1, 2, 4), (2)], [(0, 1), (2, 3, 5), (4)]]`
- `generators.py`: file containing **Generator** classes. Again, they're all instances of `nn.Module`; he `forward()` method gets the original batch of embeddings to compress, the indices from bracketing in the mentioned forward. Also, a dictionary  with the masks must be passed. There are two main classes: 
  - `EmbeddingGenerator` is for non-parametrized generation. It needs to be passed a **pooling function** as an argument (`pool_function` such as *mean* or *max pooling*).
  - `ParamEmbeddingGenerator` is for parametrized generation. It needs to get a **generator network object** as an argument (`gen_net` such as an instance of *ConvAtt*).
- `classifiers.py`: file containing **Classifier** networks, either for single sequence classification or for sequence pair classification. Again, they should all inherit from `nn.Module` and implement a `forward()` pass.
- `model.py`: contains the main model class **Model** that combines all other parts of the network. It also contains the `MultitaskNet` class, which combines any given classifiers into a single network that implements the classifiers *in parallel* (conceptually in parallel, in reality the implementation is sequential for simplicity reasons).

### `model/builder/`
This directory contains files with functions that build each part of the network, as a useful abstraction to make more flexible and reusable code. Each file and function is quite self explanatory; whenever a part of the network is build the corresponding `make_module()` function should be called. The idea is that this funcitons can be modified and played around with while maintaining main scripts clean.

## `scripts/`
Scripts to run training or evaluations.
- `train.py`: train the model. It collects the command line arguments from `args.py`.
- `eval.py`: evaluate the model on a test set. It collects the command line arguments from `args.py`.

# Examples
Add examples of command line executions of the model.

