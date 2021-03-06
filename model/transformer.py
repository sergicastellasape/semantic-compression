from transformers import BertModel, BertTokenizer
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """This class is a wrapper that implements a Transformer model from the
    @huggingface transformers library, but with convenient options of output and
    arguments for this application.
    Args:
        model_class: model class object, i.e. `BertModel` from transformers.
        tokenizer_class: tokenizer class object, i.e. `BertTokenizer`.
        pre_trained_weights: name of the pretrained weights to load, or the path
            to custom pretrained weights.
        output_layer: layer from the transformer that we will use as features.
        agg_layer: layer from the transformer that will be also kept for later
            use as features for agglomerative chunking. If None is provided, it
            defaults to the same as output_layer.
        device: torch.device to use, cpu or cuda.
    """
    def __init__(
        self,
        model_class=BertModel,
        tokenizer_class=BertTokenizer,
        pre_trained_weights="bert-base-uncased",
        output_layer=-1,
        agg_layer=None,
        device=torch.device("cpu"),
    ):
        super(Transformer, self).__init__()
        self.device = device
        self.model = model_class.from_pretrained(
            pre_trained_weights, output_hidden_states=True
        )
        self.model.to(device)
        self.tokenizer = tokenizer_class.from_pretrained(pre_trained_weights)
        self.output_layer = output_layer
        self.agg_layer = agg_layer if agg_layer is not None else output_layer

    def forward(self, batch_sequences, return_extras=False, max_length=256):
        """Performs a forward pass of the Transformer encoder.
        Args:
            batch_sequences: list of strings or pairs of strings to encode.
            return_extras: Flag which if set to `True`, returns not only the
                Transformer output layer but also masks, input ids and agg_layer
            max_length: length above which the tokenizer will truncate the input
                It helps prevent memory errors.
        Returns:
            output: transformer output layer of size (batch, length, dim)
            masks_dict: if return_extras=True, dictionary of masks (torch.tensor
                of uint8) for the output with keys `padding_mask`,
                `seq_pair_mask` and `regular_tokens_mask`.
                See https://huggingface.co/transformers/ for more info.
            batch_input_ids: returns a list of lists with integers which are the
                input ids returned by the tokenizer. Useful for later Zipf law
                calculation and estimating token.
            agg_layer_output: transformer output for agg layer of
                size (batch, length, dim)
        """
        # return_extras: add returning the masks_dict and the token_ids
        batch_input_ids, masks_dict = self.batch_encode(batch_sequences, max_length=max_length)
        hidden_states_tup = self.model(
            batch_input_ids, attention_mask=masks_dict["padding_mask"]
        )[-1]
        if return_extras:
            return hidden_states_tup[self.output_layer], masks_dict, batch_input_ids, hidden_states_tup[self.agg_layer]
        else:
            return hidden_states_tup[self.output_layer]

    def batch_encode(self, batch_sequences, max_length=256):
        """Encodes a batch of sequences. At the time of developing, huggingface's
        implementation of `batch_encode_plus` did not implement the pad_to_max_length
        method, so this method implements it (although not extremely efficiently).
        Args:
            batch_sequences: list of strings or pairs of strings to encode.
            max_length: length above which the tokenizer will truncate the input.
        Returns:
            input_tensor: `torch.tensor` to be passed to the transformer model.
            masks_dict: dictionary with masks as `torch.tensor` of `dtype=uint8`,
                keys from the dict being `padding_mask`, `seq_pair_mask` and
                `regular_tokens_mask`.
        """
        encoded_inputs_dict = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=batch_sequences,
            add_special_tokens=True,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            max_length=max_length,
        )

        batch_input_ids = encoded_inputs_dict["input_ids"]

        # get maximum sequence length in the batch to add extra padding
        seq_max_length = max([len(seq) for seq in batch_input_ids])

        # add -1 to identify the padding part
        batch_padded_token_type_ids = [
            L + [-1] * (seq_max_length - len(L))
            for L in encoded_inputs_dict["token_type_ids"]
        ]

        padded_batch_input_ids, batch_regular_tokens_mask, batch_padding_mask = (
            [],
            [],
            [],
        )
        for input_ids in batch_input_ids:
            padded_input_dict = self.tokenizer.prepare_for_model(
                input_ids,
                max_length=seq_max_length,
                pad_to_max_length=True,
                truncation_strategy="do_not_truncate",
                add_special_tokens=False,
                return_special_tokens_mask=True,
                return_attention_mask=True,
                return_token_type_ids=True,
            )

            # Construct mask that's 0s for <cls>, <sep> and <pad> tokens and 1s for the rest
            padding_mask = padded_input_dict["attention_mask"]
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(
                padded_input_dict["input_ids"], already_has_special_tokens=True
            )
            inverse_special_tokens_mask = [1 - m for m in special_tokens_mask]
            regular_tokens_mask = [
                m1 * m2 for m1, m2 in zip(inverse_special_tokens_mask, padding_mask)
            ]

            padded_batch_input_ids.append(padded_input_dict["input_ids"])
            batch_regular_tokens_mask.append(regular_tokens_mask)
            batch_padding_mask.append(padding_mask)

        input_tensor = torch.tensor(padded_batch_input_ids, device=self.device)
        regular_tokens_mask_tensor = torch.tensor(
            batch_regular_tokens_mask, device=self.device
        )
        padding_mask_tensor = torch.tensor(batch_padding_mask, device=self.device)
        seq_pair_mask_tensor = torch.tensor(
            batch_padded_token_type_ids, device=self.device
        )

        # regular_tokens_mask: 1s=regular token, 0s=special token
        # padding_mask: 1s=beefy token, 0s=padding tokens
        # seq_pair_mask: 0s=first sequence, 1s=second sequence, -1s=padding part
        masks_dict = {
            "regular_tokens_mask": regular_tokens_mask_tensor,
            "padding_mask": padding_mask_tensor,
            "seq_pair_mask": seq_pair_mask_tensor,
        }

        return input_tensor, masks_dict
