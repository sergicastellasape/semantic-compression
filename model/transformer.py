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
        self.device = device
        self.model = model_class.from_pretrained(pre_trained_weights, output_hidden_states=True)
        self.tokenizer = tokenizer_class.from_pretrained(pre_trained_weights)


    def forward(self, sequences_batch, output_layer=-1):
        input_tensor = self.batch_encode(sequences_batch)

        # hidden_states_tup[i].size() = (batch, layers, embedding_dim)
        hidden_states_tup = self.model(input_tensor)[-1]
        
        return hidden_states_tup[output_layer].detach()


    def batch_encode(self, sequences_batch):

        encoded_inputs_dict = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=sequences_batch)
        batch_input_ids = encoded_inputs_dict['input_ids']
        max_length = max([len(seq) for seq in batch_input_ids]) + 2 # get maximum sequence length for padding, the +2 is to account for the [CLS] and [SEP] tokens added
        padded_batch_input_ids = []
        for input_ids in batch_input_ids:
            padded_input_dict = self.tokenizer.prepare_for_model(input_ids, max_length=max_length, pad_to_max_length=True)
            padded_batch_input_ids.append(padded_input_dict['input_ids'])
        input_tensor = torch.tensor(padded_batch_input_ids).to(self.device)

        return input_tensor


        