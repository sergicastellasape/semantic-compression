import torch.nn as nn
import torch


class End2EndModel(nn.Module):
    def __init__(self, 
                 transformer, 
                 bracketing, 
                 generator, 
                 multitasknet, 
                 trainable_modules = ['multitasknet', 'generator']
                 device = torch.device('cpu')):
        super(PlaceholderName, self).__init__()

        self.device = device
        self.transformer = transformer
        self.bracketer = bracketer
        self.generator = generator
        self.multitasknet = multitasknet

    

    def forward(self, sequences_batch):
        context_representation = self.transformer.forward(input)
        indices = self.bracketer.forward(context_representation)
        compact_representation = self.generator.forward(context_representation, indices)
        output = self.multitasknet(compact_representation)
        return output


class MultiTaskNet(nn.Module):
    def __init__(self,
                 *task_networks,
                 device=torch.device('cpu'),
                 config={}):
        super(MultiTaskNet, self).__init__()

        self.device = device
        self.parallel_net_list = [task_networks]


    def forward(self, input):
        output = []
        # eventually this could be parallelized because there's no sequential
        # dependency, but that introduces much complexity in the implementation
        for task_net in self.parallel_net_list:
            output.append(task_net.forward(input))
        return output


    def loss(self, predictions, targets, weights=None):
        losses = []
        for network, prediction, target in zip(self.parallel_net_list, predictions, targets):
            losses.append(network.loss(prediction, target))
        
        loss = torch.tensor(losses, device=self.device, requires_grad=True)
        if scaling == None:
            multi_task_loss = torch.mean(loss)
        else:
            weights = torch.tensor(weights, device=self.device)
            multi_task_loss = torch.sum(loss * weights)

        return multi_task_loss
