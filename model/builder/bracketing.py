import logging
from ..bracketing import (
    NNSimilarityChunker,
    AgglomerativeClusteringChunker,
    HardSpanChunker,
    FixedOutChunker,
    FreqChunker,
    IdentityChunker,
    cos
)

def make_bracketer(name=None,
                   sim_threshold=None,
                   dist_threshold=None,
                   max_skip=None,
                   span=None,
                   out_num=None,
                   log_threshold=None,
                   device=None):

    assert device is not None
    assert name is not None

    if name == 'NNSimilarity':
        logging.info("BRACKETER: NNSimilarity")
        assert sim_threshold is not None, "Provide a valid threshold!"
        bracketing_net = NNSimilarityChunker(sim_function=cos,
                                             threshold=sim_threshold,
                                             exclude_special_tokens=False,
                                             combinatorics='sequential',
                                             chunk_size_limit=60,
                                             device=device)
    elif name == 'agglomerative':
        logging.info("BRACKETER: Agglomerative")
        assert dist_threshold is not None, "Provide a valid threshold!"
        assert max_skip is not None, "Provide a max skip as an argument in --max-skip!"
        bracketing_net = AgglomerativeClusteringChunker(threshold=dist_threshold,
                                                        max_skip=max_skip,
                                                        device=device)
    elif name == 'hard':
        logging.info("BRACKETER: Hard Span")
        assert span != 0, "Provide a valid span!"
        bracketing_net = HardSpanChunker(span=span,
                                         device=device)
    elif name == 'fixed':
        logging.info("BRACKETER: Fixed Output Size")
        assert out_num is not None
        bracketing_net = FixedOutChunker(out_num=out_num,
                                         device=device)
    elif name ==  'freq':
        logging.info("BRACKETER: Frequency based bracketer")
        assert log_threshold is not None
        bracketing_net = FreqChunker(alpha=1.0,
                                     log_threshold=log_threshold,
                                     device=torch.device('cpu'))
    elif name == 'none':
        logging.info("BRACKETER: NO bracketer being used")
        bracketing_net = IdentityChunker()

    else:
        raise ValueError("You must pass a valid chunker as an argument!")

    return bracketing_net
