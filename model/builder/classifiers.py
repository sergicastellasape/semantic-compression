from ..classifiers import (
    BiLSTMClassifier,
    AttentionClassifier,
    SeqPairAttentionClassifier,
    SeqPairFancyClassifier
)


def make_BiLSTMClassifier(*args, device, **kwargs):
    raise NotImplementedError()
    return BiLSTMClassifier(768,
                            hidden_dim=768,
                            sentset_size=2,
                            num_layers=2,
                            batch_size=16,
                            bidirectional=True,
                            dropout=0.0,
                            device=device)


def make_AttentionClassifier(num_classes=None, task=None, device=None):
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
    assert task is not None
    assert device is not None
    assert num_classes is not None
    return SeqPairFancyClassifier(768,
                                  num_classes,
                                  dropout=0.0,
                                  n_attention_vecs=4,
                                  task=task,
                                  device=device)


# If you want to make a new classifier, add the function and the reference here with the
# corresponding reference in the config/datasets.yml
classifiers_dict = {
    'BiLSTMClassifier': make_BiLSTMClassifier,
    'AttentionClassifier': make_AttentionClassifier,
    'SeqPairAttentionClassifier': make_SeqPairAttentionClassifier,
    'SeqPairFancyClassifier': make_SeqPairFancyClassifier
}