"""
The idea is to have functions that build the models according to the configuration
"""
import torch
from .classifiers import classifiers_dict
from ..model import MultiTaskNet

def make_multitask_net(args, dataset_config, model_config, device=None):
    """Auxiliary funciton to initialize a part of the model, to minimize
    boilerplate code and improve modularity.
    """
    assert device is not None
    network_list = []
    for dataset in args.datasets:
        classifier_name = model_config['classifiers'][dataset]
        num_classes = dataset_config[dataset]['num_classes']
        make_classifier = classifiers_dict[classifier_name]
        classifier_net = make_classifier(num_classes=num_classes,
                                         task=dataset,
                                         device=device)
        network_list.append(classifier_net)

    return MultiTaskNet(*network_list, device=device).to(device)
