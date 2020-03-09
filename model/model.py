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

    def __init__(
        self,
        transformer,
        bracketer,
        generator,
        multitasknet,
        trainable_modules=["multitasknet", "generator"],
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.device = device
        self.transformer = transformer
        self.bracketer = bracketer
        self.generator = generator
        self.multitasknet = multitasknet
        self.trainable_modules = trainable_modules  # not implemented yet

    def forward(
        self,
        sequences_batch,
        batch_slices=None,
        compression=None,
        return_comp_rate=False,
    ):

        assert compression in [True, False]
        assert batch_slices is not None
        context_representation, masks_dict = self.transformer.forward(
            sequences_batch, return_masks=True
        )

        if compression:
            indices = self.bracketer.forward(
                context_representation, masks_dict=masks_dict
            )
            (
                compact_representation,
                compact_masks_dict,
                comp_rate,
            ) = self.generator.forward(
                context_representation, indices, masks_dict=masks_dict
            )
        else:
            comp_rate = 1
            compact_representation, compact_masks_dict = (
                context_representation,
                masks_dict,
            )

        output = self.multitasknet.forward(
            compact_representation,
            batch_slices=batch_slices,
            masks_dict=compact_masks_dict,
        )
        if return_comp_rate:
            return output, comp_rate
        else:
            return output  # list of tensors, each tensor is the model prediction for each task

    def loss(self, batch_prediction, batch_targets, weights=None):
        return self.multitasknet.loss(batch_prediction, batch_targets, weights=weights)

    def metrics(self, predictions, targets):
        return self.multitasknet.metrics(predictions, targets)


class MultiTaskNet(nn.Module):
    """
    MultiTaskNet docstring
    """

    def __init__(self, *task_networks, device=torch.device("cpu"), config={}):
        super(MultiTaskNet, self).__init__()

        self.device = device
        self.parallel_net_dict = nn.ModuleDict({net.task: net for net in task_networks})
        self.config = config

    def forward(self, inp, batch_slices=None, masks_dict=None):
        assert batch_slices is not None
        assert masks_dict is not None
        output = {}
        # eventually this could be parallelized because there's no sequential
        # dependency, but makes the implementation more complex, given that the
        # input for each parallel net is different
        for dataset, task_net in self.parallel_net_dict.items():
            inp_split = inp[batch_slices[dataset], :, :]
            masks_dict_sliced = {
                key: value[batch_slices[dataset], :]
                for key, value in masks_dict.items()
            }
            if inp_split.size(0) > 0:
                output[dataset] = task_net.forward(inp_split,
                                                   masks_dict=masks_dict_sliced)
        return output

    def loss(self, predictions, targets, weights=None):
        losses, w = [], []
        for dataset in targets.keys():
            # unsqeeze so the size is (1,) and they can be concatenated,
            # otherwise torch.cat doesn't work for scalar tensors (zero dimensions)
            losses.append(self.parallel_net_dict[dataset].loss(predictions[dataset],
                                                               targets[dataset]).unsqueeze(0))
            if weights is not None:
                w.append(weights[dataset])

        loss = torch.cat(losses, dim=0)
        if weights is None:
            multi_task_loss = torch.mean(loss)
        else:
            weights = torch.tensor(w, device=self.device)
            multi_task_loss = torch.dot(loss, weights)

        return multi_task_loss

    def metrics(self, predictions, targets):
        metrics = {}
        for dataset in targets.keys():
            correct = torch.argmax(predictions[dataset], dim=1) == targets[dataset]
            metrics[dataset] = int(correct.sum()) / len(correct)
        return metrics
