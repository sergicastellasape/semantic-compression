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
    required=True,
    help="Similarity threshold used for chunking in the embedding space.",
)

parser.add_argument(
    "--chunker",
    dest="chunker",
    type=str,
    required=True,
    choices=["NNSimilarity", "agglomerative"],
    help="Specify the bracketing part of the net",
)

parser.add_argument(
    "--eval-compression",
    "-ec",
    required=True,
    type=str2bool,
    dest="eval_comp",
    help="set if compression happens during evaluation, True or False",
)

args = parser.parse_args()
