import sys
sys.path.append(".")
import argparse
from model.utils import str2bool, str2list

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
    default=666,
    required=False,
    help="Similarity threshold used for chunking in the embedding space.",
)

parser.add_argument(
    "--hard-span",
    "-span",
    dest="span",
    type=int,
    default=0,
    required=False,
    help="Hard span used for chunking naively.",
)

parser.add_argument(
    "--chunker",
    dest="chunker",
    type=str,
    required=True,
    choices=["NNSimilarity", "agglomerative", "hard"],
    help="Specify the bracketing part of the net",
)

parser.add_argument(
    "--pooling",
    dest="pooling",
    type=str,
    default="mean_pooling",
    required=False,
    choices=["abs_max_pooling", "mean_pooling"],
    help="function to do the generation"
)
parser.add_argument(
    "--learning-rate",
    "-lr",
    dest="lr",
    type=float,
    required=False,
    default=0.0001,
    help="Learning rate for Adam optimizer",
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
    "--tensorboard-comment",
    "-tbcomment",
    dest="tensorboard_comment",
    type=str,
    required=False,
    default=None,
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
    "--load-checkpoint",
    "-load",
    dest="load_checkpoint",
    required=False,
    action="store_true",
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
    "--train-compression",
    "-tc",
    required=True,
    type=str2bool,
    dest="train_comp",
    help="set if compression happens during training, True or False",
)
parser.add_argument(
    "--eval-compression",
    "-ec",
    required=True,
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
    default='[SST2, QQP]',
    type=str2list,
    dest="datasets",
    help="Set the datasets to train on.",
)

args = parser.parse_args()
