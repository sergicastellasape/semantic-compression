"""
The idea is to have functions that build the models according to the configuration
"""
import torch
from classifiers import make_classifiers_dict
from ..model import MultiTaskNet

def make_multitask_net(datasets, config, device=torch.device('cpu')):
    network_list = []
    for dataset in datasets:
        classifier_name = config[dataset]['classifier']
        num_classes = config[dataset]['num_classes']
        make_classifier = make_classifiers_dict[classifier_name]
        classifier_net = make_classifier(num_classes=num_classes,
                                         task=dataset,
                                         device=device)
        network_list.append(classifier_net)

    return MultiTaskNet(*network_list, device=device).to(device)
