import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import abs_max_pooling, hotfix_pack_padded_sequence
from model.customlayers import Attention

# this class is heavily based on the one implemented in the tutorial from pytorch
# and forum https://discuss.pytorch.org/t/lstm-to-bi-lstm/12967


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        sentset_size,
        num_layers,
        batch_size,
        bidirectional=True,
        dropout=0.0,
        device=torch.device("cpu"),
    ):
        super(BiLSTMClassifier, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        if bidirectional:
            self.directions = 2
        else:
            self.directions = 1

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            bidirectional=bidirectional,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bias=True,
        )

        # attend to LSTM outputs w/ sentiment queries
        # self.attend = nlpnn.Attention(embedding_dim, attention_type='dot')

        # The linear layer that maps from hidden state space to sentiment classification space
        self.hidden2sent = nn.Linear(hidden_dim * self.directions, sentset_size)
        self.hidden = self.init_hidden()
        self.loss_fn = nn.NLLLoss(reduction="mean")

    def init_hidden(self):

        # As of the documentation from nn.LSTM in pytorch, the input to the lstm cell is
        # the input and a tuple of (h, c) hidden state and memory state. We initialize that
        # tuple with the proper shape: num_layers*directions, batch_size, hidden_dim. Don't worry
        # that the batch here is second, this is dealt with internally if the lstm is created with
        # batch_first=True
        shape = (self.num_layers * self.directions, self.batch_size, self.hidden_dim)
        return (
            torch.zeros(shape, requires_grad=True, device=self.device),
            torch.zeros(shape, requires_grad=True, device=self.device),
        )

    def loss(self, predicted, target):
        return self.loss_fn(predicted, target)

    def forward(self, input, pooling=None, **kwargs):

        # Calculate original lengths
        collapse_embedding_dim_tensors = input.sum(
            dim=2
        )  # all elements in the embedding need to be 0
        B = (
            torch.zeros_like(input[:, :, 0]) != collapse_embedding_dim_tensors
        )  # this gives a boolean matrix of size (batch, max_seq_length)
        lengths = B.sum(dim=1).to(
            device=self.device
        )  # summing the dimension of sequence length gives the original length

        packed_tensors = hotfix_pack_padded_sequence(
            input, lengths, enforce_sorted=False, batch_first=True
        )

        # detach to make the computation graph for the backward pass only for 1 sequence
        self.init_hidden()
        h, c = self.hidden[0].detach(), self.hidden[1].detach()
        lstm_out, self.hidden = self.lstm(packed_tensors, (h, c))

        # we unpack and use the last lstm output for classification
        unpacked_output = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True
        )[0]

        # max pooling input needs to be (batch, seq_lengh, embedding)
        if pooling == "max_pooling":
            sent_space = self.hidden2sent(abs_max_pooling(unpacked_output))
        elif pooling == "mean_pooling":
            sent_space = self.hidden2sent(unpacked_output.mean(dim=1))
        else:
            sent_space = self.hidden2sent(unpacked_output[:, -1, :])

        sent_scores = F.log_softmax(sent_space, dim=1)

        return sent_scores


class AttentionClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim,
        sentset_size,
        dropout=0.0,
        n_sentiments=2,
        task=None,
        pool_mode="concat",
        device=torch.device("cpu"),
    ):
        super(AttentionClassifier, self).__init__()
        assert task is not None
        self.task = task
        self.device = device
        self.pool_mode = pool_mode

        # Define attention layers:
        self.self_attend = Attention(embedding_dim, attention_type="general").to(device)

        self.attend = Attention(embedding_dim, attention_type="dot").to(device)

        # The linear layer that maps from embedding state space to sentiment classification space
        if self.pool_mode == "concat":
            # self.classifier = nn.Sequential(nn.Linear(embedding_dim*n_sentiments, 100),
            #                                nn.ELU(),
            #                                nn.Linear(100, sentset_size))
            self.classifier = nn.Linear(embedding_dim * n_sentiments, sentset_size).to(
                device
            )
        else:
            # self.classifier = nn.Sequential(nn.Linear(embedding_dim, 100),
            #                                       nn.ELU,
            #                                       nn.Linear(100, sentset_size))
            self.classifier = nn.Linear(embedding_dim, sentset_size).to(device)

        # Loss function as negative log likelihood, which needs a logsoftmax input
        self.loss_fn = nn.NLLLoss(reduction="mean")

        # initialize the vectors that represent sentiments characteristics,
        # and add them as model parameters so that they're trained by backprop
        init_normal = torch.empty(1, n_sentiments, embedding_dim).normal_(
            mean=0, std=0.3
        )
        self.sentiment_queries = nn.Parameter(
            init_normal.clone().detach().requires_grad_(True).to(device)
        )

    def loss(self, prediction, target):
        return self.loss_fn(prediction, target)

    def forward(self, input, pooling=None, **kwargs):
        # concatenate along sequence length dimension so attention is calculated
        # along both positive and negative sentiments
        query = self.sentiment_queries.repeat(
            input.size(0), 1, 1
        )  # expand for batch size

        # sequence is the input
        self_attended_context, _ = self.self_attend(input, input)

        # self_attended_context = self.feedforward(self_attended_context)

        attention_seq, _ = self.attend(query, self_attended_context)

        if self.pool_mode == "concat":
            sentiment = self.classifier(attention_seq.flatten(1))
        elif self.pool_mode == "mean_pooling":
            sentiment = self.classifier(attention_seq.mean(1))
        elif self.pool_mode == "max_pooling":
            sentiment = self.classifier(abs_max_pooling(attention_seq, dim=1))

        sentiment_logscore = F.log_softmax(sentiment, dim=1)
        return sentiment_logscore

"""
class ConvAttentionClassifier(nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_classes,
                 dropout=0.0,
                 n_attention_vecs=2,
                 device=torch.device("cpu")):
        super()__init__()
        self.device = device

        self.conv1d = nn.Conv2d(1, 1, (embedding_dim, 3), )
"""


class SeqPairAttentionClassifier(nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_classes,
                 dropout=0.0,
                 n_attention_vecs=4,
                 task=None,
                 pool_mode="concat",
                 device=torch.device("cpu")):
        super(SeqPairAttentionClassifier, self).__init__()
        assert task is not None
        self.task = task
        self.device = device
        self.pool_mode = pool_mode

        # Define attention layers:
        self.self_attend = Attention(embedding_dim, attention_type="general").to(device)

        self.attend = Attention(embedding_dim, attention_type="dot").to(device)

        # The linear layer that maps from embedding state space to sentiment classification space
        layer_multiplier = n_attention_vecs if self.pool_mode == "concat" else 1
        self.classifier = nn.Linear(
            2 * embedding_dim * layer_multiplier, num_classes
        ).to(device)

        # Loss function as negative log likelihood, which needs a logsoftmax input
        self.loss_fn = nn.NLLLoss(reduction="mean")

        # initialize the vectors that represent attention characteristics,
        # and add them as model parameters so that they're trained by backprop
        init_normal1 = torch.empty(1, n_attention_vecs, embedding_dim).normal_(
            mean=0, std=0.3
        )
        init_normal2 = torch.empty(1, n_attention_vecs, embedding_dim).normal_(
            mean=0, std=0.3
        )
        self.attention_vecs1 = nn.Parameter(
            init_normal1.clone().detach().requires_grad_(True).to(device)
        )
        self.attention_vecs2 = nn.Parameter(
            init_normal2.clone().detach().requires_grad_(True).to(device)
        )

    def loss(self, prediction, target):
        return self.loss_fn(prediction, target)

    def forward(self, input, seq_pair_mask=None):
        """
        Input_tup should be a tuple of two tensors, containing the batches
        """
        # seq_pair_mask looks like [00000000001111111111] so for the second sentence
        # one needs to multiply by the mask and for the first one it's the negation
        mask_2 = (seq_pair_mask == 1).unsqueeze(-1).expand(input.size()).to(self.device)
        mask_1 = (seq_pair_mask == 0).unsqueeze(-1).expand(input.size()).to(self.device)

        inp_tensor1 = input * mask_1
        inp_tensor2 = input * mask_2

        # concatenate along sequence length dimension so attention is calculated
        # along both positive and negative sentiments
        query1 = self.attention_vecs1.repeat(
            inp_tensor1.size(0), 1, 1
        )  # expand for batch size
        query2 = self.attention_vecs2.repeat(inp_tensor2.size(0), 1, 1)

        # sequence is the input
        self_attended1, _ = self.self_attend(inp_tensor1, inp_tensor1)
        self_attended2, _ = self.self_attend(inp_tensor2, inp_tensor2)

        # attention in between two sentences
        combined_seq, _ = self.self_attend(inp_tensor1, inp_tensor2)

        # attention to the attention_vecs
        attention_seq, _ = self.attend(query1, combined_seq)

        if self.pool_mode == "concat":
            class_score = self.classifier(attention_seq.flatten(1))
        elif self.pool_mode == "mean_pooling":
            class_score = self.classifier(attention_seq.mean(1))
        elif self.pool_mode == "max_pooling":
            class_score = self.classifier(abs_max_pooling(attention_seq, dim=1))

        class_logscore = F.log_softmax(class_score, dim=1)
        return class_logscore


class NaivePoolingClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_classes,
        task=None,
        dropout=0.0,
        pool_mode="max_pooling",
        device=torch.device("cpu"),
    ):
        super().__init__()
        assert task is not None

        self.task = task
        self.device = device
        self.pre_pooling_linear = nn.Linear(embedding_dim, embedding_dim, bias=True)
        # network
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
        # loss func
        self.loss_fn = nn.NLLLoss(reduction="mean")

    def loss(self, prediction, target):
        return self.loss_fn(prediction, target)

    def forward(self, input, **kwargs):
        # print('input classifier size', input.size())
        # input is size (batch, max_seq_length, embedding_dim)
        inp = self.pre_pooling_linear(input)
        pooled_features = abs_max_pooling(inp, dim=1)
        class_score = self.classifier(pooled_features)
        class_log_score = self.log_softmax(class_score)
        return class_log_score


class SeqPairFancyClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_classes,
        task=None,
        dropout=0.0,
        n_attention_vecs=4,
        device=torch.device("cpu"),
    ):
        super(SeqPairFancyClassifier, self).__init__()
        assert task is not None

        self.task = task
        self.device = device

        # Define attention layers:
        self.self_attend = Attention(embedding_dim, attention_type="general").to(device)

        self.attend = Attention(embedding_dim, attention_type="dot").to(device)

        # Convolutional layer to filter attention weights
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=4, stride=1, padding=(0, 0)
        )
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.conv2 = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=5, stride=3, padding=(0, 0)
        )

        # The linear layer that maps from embedding state space to sentiment classification space
        self.seq_classifier = nn.Linear(
            embedding_dim * n_attention_vecs, num_classes
        ).to(device)
        self.weights_classifier = nn.Linear(
            486, num_classes
        )  # size of the conv. features
        self.join_classifiers = nn.Linear(num_classes * 2, num_classes)

        # Loss function as negative log likelihood, which needs a logsoftmax input
        self.loss_fn = nn.NLLLoss(reduction="mean")

        # initialize the vectors that represent attention characteristics,
        # and add them as model parameters so that they're trained by backprop
        init_normal1 = torch.empty(1, n_attention_vecs, embedding_dim).normal_(
            mean=0, std=0.3
        )
        init_normal2 = torch.empty(1, n_attention_vecs, embedding_dim).normal_(
            mean=0, std=0.3
        )
        self.attention_vecs1 = nn.Parameter(
            init_normal1.clone().detach().requires_grad_(True).to(device)
        )
        self.attention_vecs2 = nn.Parameter(
            init_normal2.clone().detach().requires_grad_(True).to(device)
        )

    def loss(self, prediction, target):
        return self.loss_fn(prediction, target)

    def forward(self, input, seq_pair_mask=None):
        """
        Input_tup should be a tuple of two tensors, containing the batches
        """
        batch_size = input.size(0)
        # seq_pair_mask looks like [00000000001111111111] so for the second sentence
        # one needs to multiply by the mask and for the first one it's the negation
        # print('seq pair original mask:', seq_pair_mask[0, :])
        mask_2 = (seq_pair_mask == 1).unsqueeze(-1).expand(input.size()).to(self.device)
        mask_1 = (seq_pair_mask == 0).unsqueeze(-1).expand(input.size()).to(self.device)

        inp_tensor1 = input * mask_1
        inp_tensor2 = input * mask_2

        # concatenate along sequence length dimension so attention is calculated
        # along both positive and negative sentiments
        query1 = self.attention_vecs1.repeat(
            inp_tensor1.size(0), 1, 1
        )  # expand for batch size
        query2 = self.attention_vecs2.repeat(inp_tensor2.size(0), 1, 1)

        # attention in between two sentences. att_weights size (batch, seq1, seq2)
        combined_seq, att_weights = self.self_attend(inp_tensor1, inp_tensor2)

        convoluted_weights = self.conv1(
            att_weights.unsqueeze(1)
        )  # batch, channels, height, width
        # interpolate to known shape of 30 by 30
        weights3x30x30 = F.interpolate(
            convoluted_weights, size=(30, 30), mode="bilinear", align_corners=True
        )

        # further extract convolutional features and make them deeper
        weights6x10x10 = self.conv2(self.leakyrelu(weights3x30x30))  # batch, 6, 10, 10

        # attention to the attention_vecs
        attention_seq, _ = self.attend(query1, combined_seq)

        class_score_seq = self.seq_classifier(attention_seq.flatten(1))
        class_score_weights = self.weights_classifier(
            weights6x10x10.view(batch_size, -1)
        )
        score_cat = torch.cat([class_score_seq, class_score_weights], dim=1)
        joint_score = self.join_classifiers(score_cat)
        class_logscore = F.log_softmax(joint_score, dim=1)

        return class_logscore
