# Introduction

Welcome to the code for compressing semantic representations and evaluating them.

# Model Architecture
The high level architecture of the problem here is shown in the following figure.

![picture](https://user-images.githubusercontent.com/33417180/76959485-f7294f80-6919-11ea-9877-503cbd04b17f.png)

# Code Architecture
## `config/`
Contains configuration files for the datasets (and possible future additions).
- `datasets.yml`: yml configuration file for datasets paths, batch sizes, choice of classifier for each task, etc.
## `assets/`
- `checkpoints/`: directory where the checkpoints from training will be saved under the name of `--run-identifier` passed as a command line argument.
- `datasets/`: directory to save the datasets used under subdirectories and as a `.tsv` file for train, test and ev sets. Each dataset path must be specified in `config/datasets/yml` file, so any other path will also work.
## `pre-trainings/`
Store any Transformer pre-training checkpoints from Huggingface here.

## `model/`
The core of the project lives here. The file system is split accordingly to the architecture of the model, where each part is a submodel (instance of `nn.Module` which means that it's treated as a Neural Network by default). `utils.py` and `data_utils.py` contain generic auxiliary functions that are used in other parts of the model.
- `transformers.py`: wrapper around the Huggingface Transformer model. It implements `batch_encode_plus` with the addition of returning a mask for the special tokens, which, at the time of development was a lacking feature from the original Transformers library.
- `bracketing.py`: this file contains the classes to build different bracketing strategies. As mentioned before, they're all instances of `nn.Module` to facilitate integration within the whole model in case this step is parametrized. The bracketing module takes as input the contextualized embedding, along with the masks dictionary, and returns a list of lists of tuples, containing the indices that will be compressed. For instance, for a batch of two sentences of length 5 and 6 respectively, it could return something like: `[[(0), (1, 2, 4), (2)], [(0, 1), (2, 3, 5), (4)]]`
- `generators.py`:
- `classifiers.py`:
- `model.py`:

### `model/builder/`

## `scripts/`
