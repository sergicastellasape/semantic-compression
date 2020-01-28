"""
Collection of utilities related to manipulating data
"""
import torch

# Given a list of indices and a Quora Question Pairs dataset this returns
# a list of tuples (question1, question2, is_duplicate)
def get_batch_QQP_from_indices(dataframe, batch_indices):
    
    q1 = dataframe.iloc[batch_indices]['question1'].tolist()
    q2 = dataframe.iloc[batch_indices]['question2'].tolist()
    is_duplicate = dataframe.iloc[batch_indices]['is_duplicate'].tolist()

    batch_question_pair_label = []
    for i in range(len(batch_indices)):
        batch_question_pair_label.append((q1[i], q2[i], is_duplicate[i]))
            
    return batch_question_pair_label


# THIS FOR IMDB DATASET: returns a list of
def sentiment2tensorIMDB(sent_list):
    logits = []
    for sent in sent_list:
        if sent == 'positive':
            logits.append(1)
        elif sent == 'negative':
            logits.append(0)
        else:
            raise ValueError("A sentiment wasn't positive or negative!")
    return torch.tensor(logits)


# For SST2 the sentiment column is already numbers so no
# need to do fancy stuff, just to tensor
def sentiment2tensorSST(sent_list):
    return torch.tensor(sent_list)