import sys
sys.path.append(".")
import argparse
from model.utils import str2bool, str2list

LOGGING_PATH = './logging'

parser = argparse.ArgumentParser(description="Model Options")
parser.add_argument(
    "--run-identifier",
    "-id",
    dest="run_id",
    type=str,
    required=True,
    help="Add an identifier that will be used to store the run in tensorboard.",
)
parser.add_argument(
    "--similarity-threshold",
    "-thr",
    dest="sim_threshold",
    type=float,
    default=None,
    required=False,
    help="Similarity threshold used for NNsim chunking in the embedding space.",
)
parser.add_argument(
    "--dist-threshold",
    "-distthr",
    dest="dist_threshold",
    type=float,
    default=None,
    required=False,
    help="Distance threshold used for agglomerative chunking in the embedding space.",
)
parser.add_argument(
    "--agg-layer",
    "-agglayer",
    dest="agg_layer",
    type=int,
    default=None,
    required=False,
    help="Layer of features to use for the agglomerative. If empty, the same trf-out-layer is used.",
)
parser.add_argument(
    "--log-threshold",
    "-logthr",
    dest="log_threshold",
    type=float,
    default=None,
    required=False,
    help="Log likelihood threshold for the frequency based bracketing.",
)
parser.add_argument(
    "--hard-span",
    "-span",
    dest="span",
    type=int,
    default=None,
    required=False,
    help="Hard span used for chunking naively.",
)
parser.add_argument(
    "--max-skip",
    "-skip",
    dest="max_skip",
    type=int,
    default=None,
    required=False,
    help="Max skip for Agglomerative Clustering.",
)
parser.add_argument(
    "--out-num",
    "-out",
    dest="out_num",
    type=int,
    default=None,
    required=False,
    help="Number of fixed output size for the fixed out size chunker.",
)
parser.add_argument(
    "--chunker",
    dest="chunker",
    type=str,
    required=False,
    default=None,
    choices=["NNSimilarity", "agglomerative", "hard", "fixed", "freq", "rand"],
    help="Specify the chunker part of the net",
)
parser.add_argument(
    "--pooling",
    dest="pooling",
    type=str,
    default=None,
    required=False,
    choices=["abs_max_pooling", "mean_pooling", "freq_pooling", "rnd_pooling","conv_att", "lstm"],
    help="function to do the generation"
)
parser.add_argument(
    "--trf-out-layer",
    "-layer",
    dest="trf_out_layer",
    type=int,
    required=True,
    help="Layer used from the transformer as embeddings.",
)
parser.add_argument(
    "--tensorboard-dir",
    "-tbdir",
    dest="log_dir",
    type=str,
    required=False,
    default="./tensorboard",
    help="rood directory where tensorboard logs are stored. ./tensorboard by default",
)
parser.add_argument(
    "--eval-periodicity",
    "-evalperiod",
    type=int,
    required=False,
    default=50,
    dest="evalperiod",
    help="How often in iterations the model is evaluated",
)
parser.add_argument(
    "--checkpoint-id",
    "-checkid",
    dest='checkpoint_id',
    default=None,
    type=str,
    required=False,
    help="ID of the checkpoint to load."
)
parser.add_argument(
    "--load-modules",
    "-load",
    dest="modules_to_load",
    default='[]',
    required=False,
    type=str2list,
    help="What modules need to be loaded from checkpoint?"
)
parser.add_argument(
    "--save-modules",
    "-save",
    dest="modules_to_save",
    default="[multitasknet]",
    type=str2list,
    required=False,
    help="What modules need to be saved in checkpoint?"
)
parser.add_argument(
    "--train-modules",
    "-train",
    dest="modules_to_train",
    default="[multitasknet]",
    type=str2list,
    required=False,
    help="What modules need to be trained by the optimizer?"
)
parser.add_argument(
    "--wall-time",
    "-wt",
    dest="walltime",
    required=False,
    type=int,
    default=3600,
    help="Walltime for training",
)
parser.add_argument(
    "--wall-steps",
    "-ws",
    dest="wallsteps",
    required=False,
    type=int,
    default=500000,
    help="Wall steps for training",
)
parser.add_argument(
    "--train-compression",
    "-tc",
    required=False,
    default='False',
    type=str2bool,
    dest="train_comp",
    help="set if compression happens during training, True or False",
)
parser.add_argument(
    "--eval-compression",
    "-ec",
    required=False,
    default='False',
    type=str2bool,
    dest="eval_comp",
    help="set if compression happens during evaluation, True or False",
)
parser.add_argument(
    "--full-test-eval",
    "-fev",
    required=False,
    default='False',
    type=str2bool,
    dest="full_test_eval",
    help="Set if an evaluation on the full test set is made at the end.",
)
parser.add_argument(
    "--datasets",
    "-dts",
    required=False,
    default=None,
    type=str2list,
    dest="datasets",
    help="Set the datasets to train on.",
)
parser.add_argument(
    "--model-config",
    "-mconfig",
    required=False,
    default='model',
    type=str,
    dest="model_config",
    help="Name of the model config within the config folder without extension",
)
parser.add_argument(
    "--datasets-config",
    "-dconfig",
    required=False,
    default='datasets',
    type=str,
    dest="datasets_config",
    help="Name of the model config within the config folder without extension",
)
parser.add_argument(
    "--optimizer-config",
    "-oconfig",
    required=False,
    default='optimizer',
    type=str,
    dest="optimizer_config",
    help="Name of the model config within the config folder without extension",
)
parser.add_argument(
    "--write-google-sheet",
    "-GS",
    required=False,
    action='store_true',
    dest="write_google_sheet",
    help="Flag to write on Google Sheet",
)
parser.add_argument(
    "--log-level",
    "-log",
    default='info',
    type=str,
    choices=['debug', 'info', 'warning'],
    required=False,
    help="Level used for debugging."
)

args = parser.parse_args()
