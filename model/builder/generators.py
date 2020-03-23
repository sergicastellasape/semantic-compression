from ..generators import EmbeddingGenerator, ParamEmbeddingGenerator, ConvAtt
from ..utils import abs_max_pooling, mean_pooling

def make_generator(pooling=None, device=None):
    assert pooling is not None
    assert device is not None
    embedding_dim = 768  # hardcoded for now
    pooling_fn = {
        'abs_max_pooling': abs_max_pooling,
        'mean_pooling': mean_pooling
    }
    pooling_nn = {
        'conv_att': ConvAtt
    }
    if pooling in pooling_fn.keys():
        return EmbeddingGenerator(pool_function=pooling_fn[pooling],
                                  device=device)
    elif pooling in pooling_nn.keys():
        return ParamEmbeddingGenerator(embedding_dim=embedding_dim,
                                       gen_net=pooling_nn[pooling],
                                       device=device)
    else:
        raise Exception("The provided pooling argument is not valid")
