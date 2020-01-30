from transformers import BertModel, BertTokenizer
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self,
                 model_class=BertModel,
                 tokenizer_class=BertTokenizer,
                 pre_trained_weights='bert-base-uncased',
                 device=torch.device('cpu')):
        super(Transformer, self).__init__()
        self.device = device #stupid comment
        self.model = model_class.from_pretrained(pre_trained_weights, output_hidden_states=True)
        self.tokenizer = tokenizer_class.from_pretrained(pre_trained_weights)


    def forward(self, batch_sequences, output_layer=-1, return_masks=False):
        batch_input_ids, masks_dict = self.batch_encode(batch_sequences)
        print('input ids device:', batch_input_ids.device)
        print('reg mask device:', masks_dict['regular_tokens_mask'].device)
        print('padding mask device:', masks_dict['padding_mask'].device)
        print('seq pair mask device:', masks_dict['seq_pair_mask'].device)
        hidden_states_tup = self.model(batch_input_ids, 
                                       attention_mask=masks_dict['padding_mask'])[-1]
        
        if return_masks:
            return hidden_states_tup[output_layer], masks_dict
        else:
            return hidden_states_tup[output_layer]


    def batch_encode(self, batch_sequences):                             
        encoded_inputs_dict = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=batch_sequences,
                                                               add_special_tokens=True,
                                                               return_special_tokens_mask=True,
                                                               return_token_type_ids=True)
        
        batch_input_ids = encoded_inputs_dict['input_ids']
        
        # get maximum sequence length for padding, the +3 is to account for the [CLS] and [SEP] tokens added
        max_length = max([len(seq) for seq in batch_input_ids])
        
        # add -1 to identify the padding part
        batch_padded_token_type_ids = [L + [-1]*(max_length - len(L)) for L 
                                       in encoded_inputs_dict['token_type_ids']]
        
        padded_batch_input_ids, batch_regular_tokens_mask, batch_padding_mask = [], [], []
        for input_ids in batch_input_ids:
            padded_input_dict = self.tokenizer.prepare_for_model(input_ids,
                                                                 max_length=max_length, 
                                                                 pad_to_max_length=True,
                                                                 truncation_strategy='do_not_truncate',
                                                                 add_special_tokens=False,
                                                                 return_special_tokens_mask=True,
                                                                 return_attention_mask=True,
                                                                 return_token_type_ids=True)

            # Construct mask that's 0s for <cls>, <sep> and <pad> tokens and 1s for the rest
            padding_mask = padded_input_dict['attention_mask']
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(padded_input_dict['input_ids'],
                                                                         already_has_special_tokens=True)
            inverse_special_tokens_mask = [1-m for m in special_tokens_mask]
            regular_tokens_mask = [m1*m2 for m1, m2
                                   in zip(inverse_special_tokens_mask, padding_mask)]
            
            padded_batch_input_ids.append(padded_input_dict['input_ids'])
            batch_regular_tokens_mask.append(regular_tokens_mask)
            batch_padding_mask.append(padding_mask)
            
        input_tensor = torch.tensor(padded_batch_input_ids, device=self.device)
        regular_tokens_mask_tensor = torch.tensor(batch_regular_tokens_mask, device=self.device)
        padding_mask_tensor = torch.tensor(batch_padding_mask, device=self.device)
        seq_pair_mask_tensor = torch.tensor(batch_padded_token_type_ids, device=self.device)

        # regular_tokens_mask: 1s=regular token, 0s=special token
        # padding_mask: 1s=beefy token, 0s=padding tokens
        # seq_pair_mask: 0s=first sequence, 1s=second sequence, -1s=padding part
        masks_dict = {'regular_tokens_mask' : regular_tokens_mask_tensor,
                      'padding_mask'        : padding_mask_tensor,
                      'seq_pair_mask'       : seq_pair_mask_tensor}
        ########print("masks_dict:", masks_dict)
        return input_tensor, masks_dict
        