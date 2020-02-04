"""
Main wrappers that implement the rest of the parts of the model: the initial common pipeline
sequentially (transformer + bracketer + generator) plus the multitask networks that
operate "in parallel"
"""
import torch.nn as nn
import torch


class End2EndModel(nn.Module):
    """
    End2EndModel docstring
    """
    def __init__(self, 
                 transformer, 
                 bracketer, 
                 generator, 
                 multitasknet, 
                 trainable_modules = ['multitasknet', 'generator'],
                 device = torch.device('cpu')):
        super().__init__()

        self.device = device
        self.transformer = transformer
        self.bracketer = bracketer
        self.generator = generator
        self.multitasknet = multitasknet
        self.trainable_modules = trainable_modules  # not implemented yet
     
    def forward(self, sequences_batch, batch_splits=None):
        assert batch_splits is not None
        context_representation, masks_dict = self.transformer.forward(sequences_batch, return_masks=True)
        # print('output of transformer size:', context_representation.size())
        indices = self.bracketer.forward(context_representation, masks_dict=masks_dict)
        compact_representation, compact_masks_dict = self.generator.forward(context_representation, 
                                                                            indices, 
                                                                            masks_dict=masks_dict)
        #print("'compact' representation size:", compact_representation.size())
        #print('difference between original and compact', context_representation - compact_representation)
        output = self.multitasknet.forward(compact_representation,
                                           batch_splits=batch_splits,
                                           masks_dict=compact_masks_dict)
        #print('output multitask net first tensor of list size', output[0].size())
        return output # list of tensors, each tensor is the model prediction for each task

    def loss(self, batch_prediction, batch_targets, weights=None):
        return self.multitasknet.loss(batch_prediction, batch_targets, weights=weights)

    def metrics(self, predictions, targets):
        return self.multitasknet.metrics(predictions, targets)

class MultiTaskNet(nn.Module):
    """
    MultiTaskNet docstring
    """
    def __init__(self,
                 *task_networks,
                 device=torch.device('cpu'),
                 config={}):
        super(MultiTaskNet, self).__init__()

        self.device = device
        self.parallel_net_list = nn.ModuleList(task_networks)
        self.config = config

    def forward(self, input, batch_splits=None, masks_dict=None):
        assert batch_splits is not None
        assert masks_dict is not None
        output = []
        # eventually this could be parallelized because there's no sequential
        # dependency, but makes the implementation more complex, given the 
        # input for each parallel net is different
        for i, task_net in enumerate(self.parallel_net_list):
            inp_split = input[batch_splits[i]:batch_splits[i+1], :, :]
            seq_pair_mask = masks_dict['seq_pair_mask'][batch_splits[i]:batch_splits[i+1], :]
            output.append(task_net.forward(inp_split, seq_pair_mask=seq_pair_mask))
        return output

    def loss(self, predictions, targets, weights=None):
        losses = []
        for network, prediction, target in zip(self.parallel_net_list, predictions, targets):
            # unsqeeze so the size is (1,) and they can be concatenated, 
            # otherwise torch.cat doesn't work for scalar tensors (zero dimensions)
            losses.append(network.loss(prediction, target).unsqueeze(0)) 
        loss = torch.cat(losses, dim=0)
        if weights is None:
            multi_task_loss = torch.mean(loss)
        else:
            weights = torch.tensor(weights, device=self.device)
            multi_task_loss = torch.dot(loss, weights)

        return multi_task_loss

    def metrics(self, predictions, targets):
        metrics = []
        for network, prediction, target in zip(self.parallel_net_list, predictions, targets):
            correct = torch.argmax(prediction, dim=1) == target
            metrics.append(int(correct.sum()) / len(correct))

        return metrics

