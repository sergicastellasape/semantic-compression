from ..classifiers import (
    BiLSTMClassifier,
    AttentionClassifier,
    SeqPairAttentionClassifier,
    SeqPairFancyClassifier,
    DecAttClassifiter,
    DecAttClassifiter_v2,
    ConvAttClassifier
)
from ..utils import abs_max_pooling


def make_BiLSTMClassifier(num_classes=None, task=None, device=None):
    """Auxiliary funciton to initialize a part of the model, to minimize
    boilerplate code and improve modularity.
    """
    assert task is not None
    assert device is not None
    assert num_classes is not None
    return BiLSTMClassifier(768,
                            hidden_dim=768,
                            num_classes=num_classes,
                            num_layers=2,
                            pooling="abs_max_pooling",
                            task=task,
                            bidirectional=True,
                            dropout=0.3,
                            device=device)


def make_AttentionClassifier(num_classes=None, task=None, device=None):
    """Auxiliary funciton to initialize a part of the model, to minimize
    boilerplate code and improve modularity.
    """
    assert task is not None
    assert device is not None
    assert num_classes is not None
    return AttentionClassifier(768,
                               num_classes,
                               dropout=0.3,
                               n_sentiments=2,
                               task=task,
                               pool_mode="concat",
                               device=device)


def make_SeqPairAttentionClassifier(num_classes=None, task=None, device=None):
    """Auxiliary funciton to initialize a part of the model, to minimize
    boilerplate code and improve modularity.
    """
    assert task is not None
    assert device is not None
    assert num_classes is not None
    return SeqPairAttentionClassifier(768,
                                      num_classes,
                                      dropout=0.0,
                                      n_attention_vecs=4,
                                      task=task,
                                      pool_mode="concat",
                                      device=device)


def make_SeqPairFancyClassifier(num_classes=None, task=None, device=None):
    """Auxiliary funciton to initialize a part of the model, to minimize
    boilerplate code and improve modularity.
    """
    assert task is not None
    assert device is not None
    assert num_classes is not None
    return SeqPairFancyClassifier(768,
                                  num_classes,
                                  dropout=0.0,
                                  n_attention_vecs=4,
                                  task=task,
                                  device=device)


def make_DecAttClassifier(num_classes=None, task=None, device=None):
    """Auxiliary funciton to initialize a part of the model, to minimize
    boilerplate code and improve modularity.
    """
    assert task is not None
    assert device is not None
    assert num_classes is not None
    return DecAttClassifiter(768,
                             num_classes,
                             dropout=0.3,
                             task=task,
                             pool_func=abs_max_pooling,
                             mask_special_tokens=True,
                             device=device)

def make_DecAttClassifier_v2(num_classes=None, task=None, device=None):
    """Auxiliary funciton to initialize a part of the model, to minimize
    boilerplate code and improve modularity.
    """
    assert task is not None
    assert device is not None
    assert num_classes is not None
    return DecAttClassifiter_v2(768,
                                num_classes,
                                num_heads=4,
                                dropout=0.3,
                                task=task,
                                pool_func=abs_max_pooling,
                                mask_special_tokens=True,
                                device=device)


def make_ConvAttClassifier(num_classes=None, task=None, device=None):
    """Auxiliary funciton to initialize a part of the model, to minimize
    boilerplate code and improve modularity.
    """
    assert task is not None
    assert device is not None
    assert num_classes is not None
    return ConvAttClassifier(768,
                             num_classes,
                             dropout=0.3,
                             task=task,
                             n_attention_vecs=3,
                             mask_special_tokens=True,
                             device=device)


# If you want to make a new classifier, add the function and the reference here
# with the corresponding reference in the config/datasets.yml
classifiers_dict = {
    'BiLSTMClassifier': make_BiLSTMClassifier,
    'AttentionClassifier': make_AttentionClassifier,
    'SeqPairAttentionClassifier': make_SeqPairAttentionClassifier,
    'SeqPairFancyClassifier': make_SeqPairFancyClassifier,
    'DecAttClassifier': make_DecAttClassifier,
    'DecAttClassifier-v2': make_DecAttClassifier_v2,
    'ConvAttClassifier': make_ConvAttClassifier,
}
