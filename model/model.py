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

        self.trainable_modules = trainable_modules
     
    def forward(self, sequences_batch, batch_splits=None):
        assert batch_splits is not None
        context_representation, masks_dict = self.transformer.forward(sequences_batch, return_masks=True)
        indices = self.bracketer.forward(context_representation, masks_dict=masks_dict)
        compact_representation, compact_masks_dict = self.generator.forward(context_representation, 
                                                                            indices, 
                                                                            masks_dict=masks_dict)
        output = self.multitasknet.forward(compact_representation, 
                                           batch_splits=batch_splits, 
                                           masks_dict=masks_dict)
        return output

    def loss(self, batch_prediction, batch_targets, weights=None):
        return self.multitasknet.loss(batch_prediction, batch_targets, weights=weights)


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
        for network in task_networks:
            self.parameters


    def forward(self, inp, batch_splits=None, masks_dict=None):
        assert batch_splits is not None
        assert masks_dict is not None
        output = []
        # eventually this could be parallelized because there's no sequential
        # dependency, but that introduces much complexity in the implementation
        for i, task_net in enumerate(self.parallel_net_list):
            inp_split = inp[batch_splits[i]:batch_splits[i+1], :, :]
            seq_pair_mask = masks_dict['seq_pair_mask'][batch_splits[i]:batch_splits[i+1]]
            output.append(task_net.forward(inp_split, seq_pair_mask=seq_pair_mask))
        return output


    def loss(self, predictions, targets, weights=None):
        losses = []
        for network, prediction, target in zip(self.parallel_net_list, predictions, targets):
            losses.append(network.loss(prediction, target))
        print(losses)
        loss = torch.tensor(losses, device=self.device, requires_grad=True)
        if weights is None:
            multi_task_loss = torch.mean(loss)
        else:
            weights = torch.tensor(weights, device=self.device)
            multi_task_loss = torch.sum(loss * weights)

        return multi_task_loss

