from ..bracketing import (
    NNSimilarityChunker,
    AgglomerativeClusteringChunker,
    HardSpanChunker,
    cos
)

sim_threshold = None
span = None

def make_bracketer(name=None, device=None, **kwargs):
    assert device is not None
    assert name is not None

    if name == 'NNSimilarity':
        print("Using NNsimilarity chunker")
        assert sim_threshold is not None, "Provide a valid threshold!"
        bracketing_net = NNSimilarityChunker(sim_function=cos,
                                             threshold=sim_threshold,
                                             exclude_special_tokens=False,
                                             combinatorics='sequential',
                                             chunk_size_limit=60,
                                             device=device)
    elif name == 'agglomerative':
        print("Using AGGLOMERATIVE chunker")
        assert sim_threshold is not None, "Provide a valid threshold!"
        bracketing_net = AgglomerativeClusteringChunker(threshold=sim_threshold,
                                                        device=device)
    elif name == 'hard':
        print("Using HARD SPAN chunker")
        assert span != 0, "Provide a valid span!"
        bracketing_net = HardSpanChunker(span=span,
                                         device=device)
    else:
        raise ValueError("You must pass a valid chunker as an argument!")

    return bracketing_net
