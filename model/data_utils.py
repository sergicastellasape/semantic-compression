"""
Collection of utilities related to manipulating data
"""
import torch

# Given a list of indices and a Quora Question Pairs dataset this returns
# a list of tuples (question1, question2, is_duplicate)
def get_batch_QQP_from_indices(dataframe, batch_indices, max_char_length=300):

    q1 = dataframe.iloc[batch_indices]['question1'].str.slice(0, max_char_length).tolist()
    q2 = dataframe.iloc[batch_indices]['question2'].str.slice(0, max_char_length).tolist()
    is_duplicate = dataframe.iloc[batch_indices]['is_duplicate'].tolist()

    batch_question_pair_label = []
    for i in range(len(batch_indices)):
        batch_question_pair_label.append(((q1[i], q2[i]), is_duplicate[i]))

    return batch_question_pair_label


def get_batch_SST2_from_indices(dataframe, batch_indices, max_char_length=None):

    reviews = dataframe.iloc[batch_indices]['review'].str.slice(0, max_char_length).tolist()
    sentiments = dataframe.iloc[batch_indices]['sentiment'].tolist()
    batch_review_sentiment = []
    for i in range(len(batch_indices)):
        batch_review_sentiment.append((reviews[i], sentiments[i]))

    return batch_review_sentiment

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