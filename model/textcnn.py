import torch
from torch import nn
import torch.nn.functional as F
from training.utils.register import registry


class TextCNN(nn.Module):

    def __init__(self,
                 input_size,
                 seq_length,
                 output_size,
                 embedding_dim,
                 dropout,
                 embedding_dropout,
                 num_filter,
                 embedding_weights,
                 filters=None):
        super(TextCNN, self).__init__()

        if filters is None:
            filters = [2, 3, 4, 5]

        self.input_size = input_size
        self.seq_length = seq_length
        self.output_size = output_size
        self.embedding_length = embedding_dim
        self.embedding_dropout = embedding_dropout
        if embedding_weights is not None:
            self.word_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(embedding_weights),
                freeze=False)
        else:
            self.word_embeddings = nn.Embedding(self.input_size, self.embedding_length)
        self.embedding_dropout_layer = nn.Dropout(p=self.embedding_dropout)

        # 卷积层
        _convs = [
            nn.Sequential(
                nn.Conv1d(embedding_dim, num_filter, filter_size),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=self.seq_length - filter_size + 1)
            )
            for filter_size in filters]
        self.convs = nn.ModuleList(_convs)

        # 正则化处理
        self.dropout = nn.Dropout(dropout, inplace=True)

        # 分类层
        self.fc = nn.Linear(num_filter * len(filters), output_size)

    def forward(self, x, **kwargs):
        x = self.word_embeddings(x)
        x = x.transpose(1, 2).contiguous()
        x = torch.cat([conv_relu_pool(x) for conv_relu_pool in self.convs], dim=1).squeeze(2)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def __repr__(self):
        main_string = super(TextCNN, self).__repr__()
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        registry.register("model_total_num", round(total_num / 1e6, 2))
        registry.register("model_trainable_num", round(trainable_num / 1e6, 2))
        main_string += "\nNumber of Total Parameter: %.2f M\n" % (total_num / 1e6)
        main_string += "Number of Trainable Parameter: %.2f M" % (trainable_num / 1e6)
        return main_string


class TextCnnWithoutEmbed(nn.Module):

    def __init__(self,
                 seq_length,
                 output_size,
                 embedding_dim,
                 dropout,
                 num_filter,
                 filters=None):
        super(TextCnnWithoutEmbed, self).__init__()

        if filters is None:
            filters = [2, 3, 4, 5]

        self.seq_length = seq_length
        self.output_size = output_size
        self.embedding_length = embedding_dim

        # 卷积层
        _convs = [
            nn.Sequential(
                nn.Conv1d(embedding_dim, num_filter, filter_size),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=self.seq_length - filter_size + 1)
            )
            for filter_size in filters]
        self.convs = nn.ModuleList(_convs)

        # 正则化处理
        self.dropout = nn.Dropout(dropout, inplace=True)

        # 分类层
        self.fc = nn.Linear(num_filter * len(filters), output_size)

    def forward(self, x, **kwargs):
        x = x.transpose(1, 2).contiguous()
        x = torch.cat([conv_relu_pool(x) for conv_relu_pool in self.convs], dim=1).squeeze(2)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def __repr__(self):
        main_string = super(TextCnnWithoutEmbed, self).__repr__()
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        registry.register("model_total_num", round(total_num / 1e6, 2))
        registry.register("model_trainable_num", round(trainable_num / 1e6, 2))
        main_string += "\nNumber of Total Parameter: %.2f M\n" % (total_num / 1e6)
        main_string += "Number of Trainable Parameter: %.2f M" % (trainable_num / 1e6)
        return main_string
