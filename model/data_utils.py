"""
Collection of utilities related to manipulating data
"""
import torch

# Given a list of indices and a Quora Question Pairs dataset this returns
# a list of tuples (question1, question2, is_duplicate)
def get_batch_QQP_from_indices(dataframe, batch_indices, max_char_length=1000):
    """Function to load a batch from a the Quora Question Pairs dataframe.
    Args:
        dataframe: `pandas.DataFrame` object containing the QQP dataset.
        batch_indices: list of indices from the dataset that we want to
            load in the batch
        max_char_length: limit on the character length when loading samples.
            It should not be nessessary because the model already has a max_length
            argument to truncate too long sequences, but it is a safeguard. It can
            be set to None and it will have no effect.
    Returns:
        batch_question_pair_label: list (length=batch) of tuples with
            (question1, question2, is_duplicate).
    """
    q1 = (
        dataframe.iloc[batch_indices]["question1"]
        .str.slice(0, max_char_length)
        .tolist()
    )
    q2 = (
        dataframe.iloc[batch_indices]["question2"]
        .str.slice(0, max_char_length)
        .tolist()
    )
    is_duplicate = dataframe.iloc[batch_indices]["is_duplicate"].tolist()

    batch_question_pair_label = []
    for i in range(len(batch_indices)):
        batch_question_pair_label.append(((q1[i], q2[i]), is_duplicate[i]))

    return batch_question_pair_label


def get_batch_SST2_from_indices(dataframe, batch_indices, max_char_length=1000):
    """Function to load a batch from a the Stanford Sentiment Treebank v2 dataframe.
    Args:
        dataframe: `pandas.DataFrame` object containing the QQP dataset.
        batch_indices: list of indices from the dataset that we want to
            load in the batch
        max_char_length: limit on the character length when loading samples.
            It should not be nessessary because the model already has a max_length
            argument to truncate too long sequences, but it is a safeguard. It can
            be set to None and it will have no effect.
    Returns:
        batch_review_sentiment: list (length=batch) of tuples with
            (review, is_duplicate).
    """
    reviews = (
        dataframe.iloc[batch_indices]["review"].str.slice(0, max_char_length).tolist()
    )
    sentiments = dataframe.iloc[batch_indices]["sentiment"].tolist()
    batch_review_sentiment = []
    for i in range(len(batch_indices)):
        batch_review_sentiment.append((reviews[i], sentiments[i]))

    return batch_review_sentiment


def get_batch_MNLI_from_indices(dataframe, batch_indices, max_char_length=1000):
    """Function to load a batch from a the Matched Natural Language Inference dataframe.
    Args:
        dataframe: `pandas.DataFrame` object containing the QQP dataset.
        batch_indices: list of indices from the dataset that we want to
            load in the batch
        max_char_length: limit on the character length when loading samples.
            It should not be nessessary because the model already has a max_length
            argument to truncate too long sequences, but it is a safeguard. It can
            be set to None and it will have no effect.
    Returns:
        batch_sentence_pair_label: list (length=batch) of tuples with
            (sentence1, sentence2, is_duplicate).
    """
    s1 = (
        dataframe.iloc[batch_indices]["sentence1"]
        .str.slice(0, max_char_length)
        .tolist()
    )
    s2 = (
        dataframe.iloc[batch_indices]["sentence2"]
        .str.slice(0, max_char_length)
        .tolist()
    )
    label = dataframe.iloc[batch_indices]["gold_label"].tolist()

    batch_sentence_pair_label = []
    for i in range(len(batch_indices)):
        batch_sentence_pair_label.append(((s1[i], s2[i]), label[i]))

    return batch_sentence_pair_label

def get_batch_WNLI_from_indices(dataframe, batch_indices, max_char_length=1000):
    """Function to load a batch from a the wNLI dataframe.
    Args:
        dataframe: `pandas.DataFrame` object containing the QQP dataset.
        batch_indices: list of indices from the dataset that we want to
            load in the batch
        max_char_length: limit on the character length when loading samples.
            It should not be nessessary because the model already has a max_length
            argument to truncate too long sequences, but it is a safeguard. It can
            be set to None and it will have no effect.
    Returns:
        batch_question_pair_label: list (length=batch) of tuples with
            (question1, question2, is_duplicate).
    """
    s1 = (
        dataframe.iloc[batch_indices]["sentence1"]
        .str.slice(0, max_char_length)
        .tolist()
    )
    s2 = (
        dataframe.iloc[batch_indices]["sentence2"]
        .str.slice(0, max_char_length)
        .tolist()
    )
    label = dataframe.iloc[batch_indices]["label"].tolist()

    batch_question_pair_label = []
    for i in range(len(batch_indices)):
        batch_question_pair_label.append(((s1[i], s2[i]), label[i]))

    return batch_question_pair_label


def sentiment2tensorIMDB(sent_list):
    """Convert 'positive' and 'negative' labels to 1s and 0s"""
    logits = []
    for sent in sent_list:
        if sent == "positive":
            logits.append(1)
        elif sent == "negative":
            logits.append(0)
        else:
            raise ValueError("A sentiment wasn't positive or negative!")
    return torch.tensor(logits)


def sentiment2tensorSST(sent_list):
    """Convert list to tensor. Done for consistency with the IMBD dataset"""
    return torch.tensor(sent_list)
