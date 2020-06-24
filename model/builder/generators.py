from ..generators import (
    EmbeddingGenerator,
    ParamEmbeddingGenerator,
    ConvAtt,
    LSTM,
    IdentityGenerator
)
from ..utils import abs_max_pooling, mean_pooling, freq_pooling, rnd_pooling

def make_generator(args, device=None):
    """Auxiliary funciton to initialize a part of the model, to minimize
    boilerplate code and improve modularity.
    """
    assert device is not None

    embedding_dim = 768  # hardcoded for now
    pooling_fn = {
        'abs_max_pooling': abs_max_pooling,
        'mean_pooling': mean_pooling,
        'freq_pooling': freq_pooling,
        'rnd_pooling': rnd_pooling,
    }
    pooling_nn = {
        'conv_att': ConvAtt,
        'lstm': LSTM,
    }
    if args.pooling is None:
        return IdentityGenerator(device=device)

    if args.pooling in pooling_fn.keys():
        return EmbeddingGenerator(pool_function=pooling_fn[args.pooling],
                                  device=device)
    elif args.pooling in pooling_nn.keys():
        return ParamEmbeddingGenerator(embedding_dim=embedding_dim,
                                       gen_net=pooling_nn[args.pooling],
                                       device=device)
    else:
        raise Exception("The provided pooling argument is not valid")
