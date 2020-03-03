from ..generators import EmbeddingGenerator
from ..utils import abs_max_pooling, mean_pooling

pooling_fn = {
    'abs_max_pooling': abs_max_pooling,
    'mean_pooling': mean_pooling
}

def make_generator(pool_function=None, device=None):
    assert pool_function is not None
    assert device is not None
    return EmbeddingGenerator(pool_function=pool_function,
                              device=device)
