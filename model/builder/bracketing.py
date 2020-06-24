import logging
from ..bracketing import (
    NNSimilarityChunker,
    AgglomerativeClusteringChunker,
    HardSpanChunker,
    FixedOutChunker,
    FreqChunker,
    RndSpanChunker,
    IdentityChunker,
    cos
)

def make_bracketer(args, device=None):
    """Auxiliary funciton to initialize a part of the model, to minimize
    boilerplate code and improve modularity.
    """
    assert device is not None

    if args.chunker == 'NNSimilarity':
        logging.info("BRACKETER: NNSimilarity")
        assert args.sim_threshold is not None, "Provide a valid threshold!"
        bracketing_net = NNSimilarityChunker(sim_function=cos,
                                             threshold=args.sim_threshold,
                                             exclude_special_tokens=False,
                                             combinatorics='sequential',
                                             chunk_size_limit=60,
                                             device=device)
    elif args.chunker == 'agglomerative':
        logging.info("BRACKETER: Agglomerative")
        assert args.dist_threshold is not None, "Provide a valid threshold!"
        assert args.max_skip is not None, "Provide a max skip as an argument in --max-skip!"
        bracketing_net = AgglomerativeClusteringChunker(threshold=args.dist_threshold,
                                                        max_skip=args.max_skip,
                                                        device=device)
    elif args.chunker == 'hard':
        logging.info("BRACKETER: Hard Span")
        assert args.span is not None, "Provide a valid span!"
        bracketing_net = HardSpanChunker(span=args.span,
                                         device=device)
    elif args.chunker == 'fixed':
        logging.info("BRACKETER: Fixed Output Size")
        assert args.out_num is not None
        bracketing_net = FixedOutChunker(out_num=args.out_num,
                                         device=device)
    elif args.chunker == 'freq':
        logging.info("BRACKETER: Frequency based bracketer")
        assert args.log_threshold is not None
        bracketing_net = FreqChunker(alpha=1.0,
                                     log_threshold=args.log_threshold,
                                     device=device)
    elif args.chunker == 'rand':
        logging.info("BRACKETER: Random span bracketer")
        assert args.span is not None, "Provide a valid base span!"
        bracketing_net = RndSpanChunker(span=args.span,
                                        device=device)
    elif args.chunker is None:
        logging.info("BRACKETER: NO bracketer being used")
        bracketing_net = IdentityChunker()
    else:
        raise ValueError("You must pass a valid chunker as an argument!")

    return bracketing_net
