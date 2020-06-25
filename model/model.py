"""
Main wrappers that implement the rest of the parts of the model: the initial common pipeline
sequentially (transformer + bracketer + generator) plus the multitask networks that
operate "in parallel"
"""
import os
import torch.nn as nn
import torch


class End2EndModel(nn.Module):
    """Parent model that combines all elements of the pipeline into a single
    class iheriting from nn.Module that implements forwad() and loss() methods
    along with convenient methods such as loading and saving parts of the model
    from checkpoints.
    Args:
        transformer: transformer network from Transformer() class.
        bracketer: chunking network from any of the classes in bracketing.py
        generator: generator network from from any of the classes in generators.py
        multitasknet: multitask network from MultiTaskNet() class.
        device: `torch.device` to use, for cpu or cuda.
    """

    def __init__(
        self,
        transformer,
        bracketer,
        generator,
        multitasknet,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.device = device
        self.transformer = transformer
        self.bracketer = bracketer
        self.generator = generator
        self.multitasknet = multitasknet
        self.allowed_modules = ['transformer', 'bracketer',
                                'generator', 'multitasknet']

    def forward(
        self,
        sequences_batch,
        batch_slices=None,
        compression=None,
        return_comp_rate=False,
        max_length=256,
    ):
        """Implements a forward pass for the whole end to end model.
        Args:
            sequences_batch: list of strings or pairs of strings to encode.
            batch_slices: dictionary with dataset names as keys and slice
                objects that indicate what portion of the batch belongs to what task
            compression: boolean indicating if compression should be preformed
                during the forward pass.
            return_comp_rate: boolean indicating wether you want the method to
                return the average compression for each task in the batch.
            max_length: length above which the tokenizer will truncate the input
                It helps prevent memory errors.
        Returns:
            output: the logits for each output in the batch in a dict of tensors,
                with keys for each dataset name, given that the number of
                classes for each task can be different.
        """
        assert compression in [True, False]
        assert batch_slices is not None
        context_representation, masks_dict, batch_input_ids, bracketer_representation = self.transformer.forward(
            sequences_batch, return_extras=True, max_length=max_length,
        )

        if compression:
            indices = self.bracketer.forward(bracketer_representation,
                                             masks_dict=masks_dict,
                                             mask_special_tokens=True,
                                             token_ids=batch_input_ids)
            (
                compact_representation,
                compact_masks_dict,
                comp_rate,
            ) = self.generator.forward(context_representation,
                                       indices,
                                       masks_dict=masks_dict,
                                       token_ids=batch_input_ids)
        else:
            comp_rate = 1
            compact_representation = context_representation
            compact_masks_dict = masks_dict

        output = self.multitasknet.forward(
            compact_representation,
            batch_slices=batch_slices,
            masks_dict=compact_masks_dict,
        )
        if return_comp_rate:
            return output, comp_rate
        else:
            # dict of tensors, each tensor is the model prediction for each task
            return output

    def loss(self, batch_prediction, batch_targets, weights=None):
        """Runs the loss() method for the multitask net. See MultiTaskNet loss
        docstring.
        """
        return self.multitasknet.loss(batch_prediction, batch_targets, weights=weights)

    def metrics(self, predictions, targets):
        """Runs the metrics() method for the multitask net. See MultitaskNet loss
        docstring.
        """
        return self.multitasknet.metrics(predictions, targets)

    def save_modules(self,
                     checkpoint_id,
                     modules=['generator', 'multitasknet'],
                     parent_path='./assets/checkpoints/'):
        """Similarly as the `load_modules()` method, this saves modules'
        `state_dict` of parameters into a checkpoint as a .pt file in a directory
        named after the module, within the `parent_path`.
        Args:
            checkpoint_id: name of the checkpoint (i.e. the name of the run)
                *without* the file extension, which is added internally.
            modules: list of module names to save (i.e. [generator, multitasknet])
            parent_path: absolute or relative path to the parent path to save
                checkpoints. Inside that directory, a directory with the module
                name will be created and the checkpoint for that module will be
                saved in there. This allows us to save only parts of the model,
                minimizing the memory usage.
        Returns:
            None
        """
        assert all(module in self.allowed_modules for module in modules),\
            "Invalid module provided"

        if not os.path.exists(parent_path):
            os.makedirs(parent_path)

        for module in modules:
            path = os.path.join(parent_path, module)
            if not os.path.exists(path):
                os.makedirs(path)
            module_state_dict = getattr(self, module).state_dict()
            checkpoint_path = os.path.join(path, f'{checkpoint_id}.pt')
            torch.save(module_state_dict, checkpoint_path)

    def load_modules(self,
                     checkpoint_id,
                     modules=[],
                     parent_path='./assets/checkpoints'):
        """Similarly as the `save_modules()` method, this loads a set of modules
        from a checkpoint id.
        Args:
            checkpoint_id: name of the checkpoint (i.e. the name of the run)
                *without* the file extension, which is added internally. If
                we load more than one part of the network, they need to have the
                same `checkpoint_id`, which is a limitation that should be fixed
                but that implies some overhead with changing the `argparse` args
                passed in the command line interface.
            modules: list of module names to save (i.e. [generator, multitasknet])
            parent_path: absolute or relative path to the parent path to save
                checkpoints. Inside that directory, a directory with the module
                name will be created and the checkpoint for that module will be
                saved in there. This allows us to save only parts of the model,
                minimizing the memory usage.
        Returns:
            None
        """
        assert all(module in self.allowed_modules for module in modules),\
            "Invalid module provided"

        for module in modules:
            path = os.path.join(parent_path, module)
            checkpoint_path = os.path.join(path, f'{checkpoint_id}.pt')
            assert os.path.exists(checkpoint_path),\
                f"Checkpoint for {module}: {checkpoint_id} does not exist!"
            getattr(self, module).load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )


class MultiTaskNet(nn.Module):
    """Module that combines different classifier networks into a single class
    that inherits from nn.Module and implements a `forward()` and `loss()` methods.
    Args:
        *task_networks: unzipped list of classfier networks.
        device: `torch.device` to use, for cpu or cuda.
    Returns:
        MultiTaskNet module
    """

    def __init__(self, *task_networks, device=torch.device("cpu")):
        super(MultiTaskNet, self).__init__()

        self.device = device
        self.parallel_net_dict = nn.ModuleDict({net.task: net for net in task_networks})

    def forward(self, inp, batch_slices=None, masks_dict=None):
        """Performs a forward pass for all subnetworks sequentially. This could
        be theoretically done in parallel, as they are independent, but it is not
        implemented.
        Args:
            inp: `torch.tensor` of size (batch, length, embedding_dim)
            batch_slices: dictionary with dataset names as keys and slice
                objects that indicate what portion of the batch belongs to what task.
            masks_dict: dictionary of masks (torch.tensor of uint8) for the
                output with keys `padding_mask`, `seq_pair_mask` and
                `regular_tokens_mask`.
        Returns:
            output: dictionary where keys are dataset/task names and values are
            `torch.tensors` of size (batch_split,) with logits for the
            classifiers' predictions.
        """
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
        """Calculates the loss function from each classifier network and combines
        it with a weighted sum or just an average if weights are not provided.
        The loss is calculated in each classifier network, and all of them are
        Negative Log Likelihood losses over C classes using the log-probabilities
        that the network outputs, which are equivalent to a Cross Entropy Loss.
        Args:
            predictions: output of the forward pass of the multitask network
                (or the end to end model, which is equivalent). This is a dictionary
                with keys as dataset/task names and values as logit tensors.
            targets: dictionary with the same keys as predictions with values of
                `torch.tensor` of size (dataset_batch_split,) with the class label
                as an integer from [0,... , C - 1].
            weights: dictionary of weights to fo the weighted sum of losses.
        Returns:
            multitask_loss: combined loss for all tasks.
        """
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
            multitask_loss = torch.mean(loss)
        else:
            weights = torch.tensor(w, device=self.device)
            multitask_loss = torch.dot(loss, weights)

        return multitask_loss

    def metrics(self, predictions, targets):
        """Returns a dictionary of metrics (accuracy) for each task given a set
        of predictions and targets.
        Args:
            predictions: output of the forward pass of the multitask network
                (or the end to end model, which is equivalent). This is a dictionary
                with keys as dataset/task names and values as logit tensors.
            targets: dictionary with the same keys as predictions with values of
                `torch.tensor` of size (dataset_batch_split,) with the class label
                as an integer from [0,... , C - 1].
        Returns:
            metrics: dictionary with keys as dataset/task names and metrics
                (accuracies) for them.
        """
        metrics = {}
        for dataset in targets.keys():
            correct = torch.argmax(predictions[dataset], dim=1) == targets[dataset]
            metrics[dataset] = int(correct.sum()) / len(correct)
        return metrics
