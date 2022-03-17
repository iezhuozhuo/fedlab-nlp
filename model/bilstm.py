import torch
from torch import nn
from training.utils.register import registry


class BiLSTM_TextClassification(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, embedding_dropout, lstm_dropout,
                 attention_dropout, embedding_length, attention=False, embedding_weights=None):
        super(BiLSTM_TextClassification, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.embedding_dropout = embedding_dropout
        self.lstm_dropout = lstm_dropout
        self.attention_dropout = attention_dropout
        self.attention = attention
        self.embedding_length = embedding_length

        if embedding_weights is not None:
            self.word_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(embedding_weights),
                freeze=False)
        else:
            self.word_embeddings = nn.Embedding(self.input_size, self.embedding_length)
        self.embedding_dropout_layer = nn.Dropout(p=self.embedding_dropout)
        if self.attention:
            self.attention_layer = nn.Linear(self.hidden_size * 4, self.hidden_size * 2)
            self.attention_dropout_layer = nn.Dropout(p=self.attention_dropout)

        self.lstm_layer = nn.LSTM(self.embedding_length, self.hidden_size, self.num_layers, dropout=lstm_dropout,
            bidirectional=True)
        self.lstm_dropout_layer = nn.Dropout(p=self.lstm_dropout)
        self.output_layer = nn.Linear(self.hidden_size * 2, self.output_size)

    def attention_forward(self, lstm_output, state, seq_lens):
        # We implement Luong attention here, the attention range should be less or equal than original sequence length
        # lstm_output -> [batch_size, seq_len, num_directions*hidden_size]
        # state -> [batch_size, num_directions*hidden_size]

        hidden = state.unsqueeze(2)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        # attn_weights -> [batch_size, seq_len]
        new_hiddens = []
        for i, seq_len in enumerate(seq_lens):
            soft_attn_weights = torch.softmax(attn_weights[i][:seq_len], 0)
            # soft_attn_weights -> [seq_len]
            new_hidden = torch.matmul(soft_attn_weights.unsqueeze(0), lstm_output[i, :seq_len, :])
            # new_hidden ->[1, num_directions*hidden_size]
            new_hiddens.append(new_hidden)
        concat_hidden = torch.cat((torch.cat(new_hiddens, 0), state), 1)
        # concat_hidden ->[batch_size, 2*num_directions*hidden_size]
        output_hidden = self.attention_layer(concat_hidden)
        # output_hidden ->[batch_size, num_directions*hidden_size]
        output_hidden = self.attention_dropout_layer(output_hidden)
        return output_hidden

    def forward(self, input_seq, batch_size, seq_lens, device):
        # input_seq -> [batch_size, seq_len]
        input_seq = self.word_embeddings(input_seq)
        # input -> [batch_size, seq_len, embedding_len]

        input_seq = self.embedding_dropout_layer(input_seq)

        h_0 = torch.zeros((self.num_layers * 2, batch_size, self.hidden_size)).to(device=device)
        c_0 = torch.zeros((self.num_layers * 2, batch_size, self.hidden_size)).to(device=device)

        input_seq = input_seq.permute(1, 0, 2)
        output, (final_hidden_state, final_cell_state) = self.lstm_layer(input_seq, (h_0, c_0))
        # output -> [seq_len, batch_size, num_directions*hidden_size]

        output = output.permute(1, 0, 2)
        # the final state is constructed based on original sequence lengths
        state = torch.cat([output[i, seq_len - 1, :].unsqueeze(0) for i, seq_len in enumerate(seq_lens)], dim=0)

        state = self.lstm_dropout_layer(state)

        if self.attention:
            output = self.attention_forward(output, state, seq_lens)
        else:
            output = state

        logits = self.output_layer(output)

        return logits

    def __repr__(self):
        main_string = super(BiLSTM_TextClassification, self).__repr__()
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        registry.register("model_total_num", round(total_num / 1e6, 2))
        registry.register("model_trainable_num", round(trainable_num / 1e6, 2))
        main_string += "\nNumber of Total Parameter: %.2f M\n" % (total_num / 1e6)
        main_string += "Number of Trainable Parameter: %.2f M" % (trainable_num / 1e6)
        return main_string


class Embedding(nn.Module):
    def __init__(self, vocb_size, embedding_length, embedding_dropout,
                 embedding_weights=None):
        super().__init__()
        self.vocb_size = vocb_size
        self.embedding_length = embedding_length
        self.embedding_dropout = embedding_dropout

        if embedding_weights is not None:
            self.word_embeddings = nn.Embedding.from_pretrained(torch.tensor(embedding_weights))
        else:
            self.word_embeddings = nn.Embedding(self.vocb_size, self.embedding_length)
        self.embedding_dropout_layer = nn.Dropout(p=self.embedding_dropout)

    def forward(self, input_seq):
        input_seq = self.word_embeddings(input_seq)
        input_seq = self.embedding_dropout_layer(input_seq)
        return input_seq

    def __repr__(self):
        main_string = super(Embedding, self).__repr__()
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        registry.register("model_total_num", round(total_num / 1e6, 2))
        registry.register("model_trainable_num", round(trainable_num / 1e6, 2))
        main_string += "\nNumber of Total Parameter: %.2f M\n" % (total_num / 1e6)
        main_string += "Number of Trainable Parameter: %.2f M" % (trainable_num / 1e6)
        return main_string

    def weight(self):
        return self.word_embeddings.weight

    def from_pretrained(self, embedding_weights):
        self.word_embeddings = nn.Embedding.from_pretrained(embedding_weights)


class BiLSTM_TextClassification_WithoutEmbed(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, lstm_dropout,
                 embedding_length):
        super(BiLSTM_TextClassification_WithoutEmbed, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm_dropout = lstm_dropout
        self.embedding_length = embedding_length

        self.lstm_layer = nn.LSTM(self.embedding_length, self.hidden_size, self.num_layers, dropout=lstm_dropout,
            bidirectional=True)
        self.lstm_dropout_layer = nn.Dropout(p=self.lstm_dropout)
        self.output_layer = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, input_seq, batch_size, seq_lens, device):
        # input -> [batch_size, seq_len, embedding_len]
        h_0 = torch.zeros((self.num_layers * 2, batch_size, self.hidden_size)).to(device=device)
        c_0 = torch.zeros((self.num_layers * 2, batch_size, self.hidden_size)).to(device=device)

        input_seq = input_seq.permute(1, 0, 2)
        output, (final_hidden_state, final_cell_state) = self.lstm_layer(input_seq, (h_0, c_0))
        # output -> [seq_len, batch_size, num_directions*hidden_size]

        output = output.permute(1, 0, 2)
        # the final state is constructed based on original sequence lengths
        state = torch.cat([output[i, seq_len - 1, :].unsqueeze(0) for i, seq_len in enumerate(seq_lens)], dim=0)

        state = self.lstm_dropout_layer(state)

        logits = self.output_layer(state)

        return logits

    def __repr__(self):
        main_string = super(BiLSTM_TextClassification_WithoutEmbed, self).__repr__()
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        registry.register("model_total_num", round(total_num / 1e6, 2))
        registry.register("model_trainable_num", round(trainable_num / 1e6, 2))
        main_string += "\nNumber of Total Parameter: %.2f M\n" % (total_num / 1e6)
        main_string += "Number of Trainable Parameter: %.2f M" % (trainable_num / 1e6)
        return main_string
