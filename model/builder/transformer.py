# Own wrapper of transformers
from ..transformer import Transformer

# Transformers from huggingface
from transformers import BertModel, BertTokenizer

def make_transformer(output_layer=None, device=None):
    assert output_layer is not None
    assert device is not None
    transformer_net = Transformer(model_class=BertModel,
                                  tokenizer_class=BertTokenizer,
                                  pre_trained_weights="bert-base-uncased",
                                  output_layer=output_layer,
                                  device=device)
    return transformer_net
