# Own wrapper of transformers
from ..transformer import Transformer

# Transformers from huggingface
from transformers import BertModel, BertTokenizer

def make_transformer(args, device=None):
    """Auxiliary funciton to initialize a part of the model, to minimize
    boilerplate code and improve modularity.
    """
    assert args.trf_out_layer is not None
    assert device is not None
    transformer_net = Transformer(model_class=BertModel,
                                  tokenizer_class=BertTokenizer,
                                  pre_trained_weights="bert-base-uncased",
                                  output_layer=args.trf_out_layer,
                                  agg_layer=args.agg_layer,
                                  device=device)
    return transformer_net
