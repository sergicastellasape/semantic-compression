from ..generators import EmbeddingGenerator
from ..utils import abs_max_pooling, mean_pooling

def make_generator(pooling=None, device=None):
    assert pooling is not None
    assert device is not None
    pooling_fn = {
        'abs_max_pooling': abs_max_pooling,
        'mean_pooling': mean_pooling
    }
    return EmbeddingGenerator(pool_function=pooling_fn[pooling],
                              device=device)
