import numpy as np
import torch
import torch.nn as nn

class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, word_embedding_size, word_vec_size, hidden_size, bidirectional=False,
                 input_dropout_p=0, dropout_p=0, n_layers=1, rnn_type='lstm', variable_lengths=True):
        super(RNNEncoder, self).__init__()
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, word_embedding_size)
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.mlp = nn.Sequential(nn.Linear(word_embedding_size, word_vec_size),
                                 nn.ReLU())
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type.upper())(word_vec_size, hidden_size, n_layers,
                                                 batch_first=True,
                                                 bidirectional=bidirectional,
                                                 dropout=dropout_p)
        self.num_dirs = 2 if bidirectional else 1

    def forward(self, input_labels,embedded_weight=None):
        """
        Inputs:
        - input_labels: Variable long (batch, seq_len)
        Outputs:
        - output  : Variable float (batch, max_len, hidden_size * num_dirs)
        - hidden  : Variable float (batch, num_layers * num_dirs * hidden_size)
        - embedded: Variable float (batch, max_len, word_vec_size)
        """

        input_labels=torch.cat(input_labels,dim=0)
        max_len = (input_labels != 0).sum(1).max()
        input_labels = input_labels[:, :max_len]

        if max_len  == 0:
            input_labels = torch.tensor(1).cuda().unsqueeze(0).unsqueeze(0)

        if self.variable_lengths:
            input_lengths = (input_labels != 0).sum(1)  # Variable (batch, )

            # make ixs
            input_lengths_list = input_lengths.data.cpu().numpy().tolist()
            sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist()  # list of sorted input_lengths
            sort_ixs = np.argsort(input_lengths_list)[::-1].tolist()  # list of int sort_ixs, descending
            s2r = {s: r for r, s in enumerate(sort_ixs)}  # O(n)
            recover_ixs = [s2r[s] for s in range(len(input_lengths_list))]  # list of int recover ixs
            # print ('input_length_list:', max(input_lengths_list))
            # print('input_labels:',input_labels.size(1))
            assert max(input_lengths_list) == input_labels.size(1)

            # move to long tensor
            sort_ixs = input_labels.new(sort_ixs).long()  # Variable long
            recover_ixs = input_labels.new(recover_ixs).long()  # Variable long

            # sort input_labels by descending order
            input_labels = input_labels[sort_ixs]

        # embed
        embedded = self.embedding(input_labels.long())  # (n, seq_len, word_embedding_size)
        embedded = self.input_dropout(embedded)  # (n, seq_len, word_embedding_size)
        embedded = self.mlp(embedded)  # (n, seq_len, word_vec_size)
        if embedded_weight != None:
            embedded = embedded * embedded_weight.unsqueeze(2)

        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_input_lengths_list, batch_first=True)

        # forward rnn
        output, hidden = self.rnn(embedded)

        # recover
        if self.variable_lengths:

            # embedded (batch, seq_len, word_vec_size)
            embedded, _ = nn.utils.rnn.pad_packed_sequence(embedded, batch_first=True)
            embedded = embedded[recover_ixs]

            # recover rnn
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)  # (batch, max_len, hidden)
            output = output[recover_ixs]

            # recover hidden
            if self.rnn_type == 'lstm':
                hidden = hidden[0]  # we only use hidden states for the final hidden representation

            hidden = hidden[:, recover_ixs, :]  # (num_layers * num_dirs, batch, hidden_size)
            hidden = hidden.transpose(0, 1).contiguous()  # (batch, num_layers * num_dirs, hidden_size)
            hidden = hidden.view(hidden.size(0), -1)  # (batch, num_layers * num_dirs * hidden_size)

        return output, hidden, embedded

class RNNEncoder_ATT(nn.Module):
    def __init__(self, vocab_size, word_embedding_size, word_vec_size, hidden_size, bidirectional=False,
                 input_dropout_p=0, dropout_p=0, n_layers=1, rnn_type='lstm', variable_lengths=True):
        super(RNNEncoder_ATT, self).__init__()
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, word_embedding_size)
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.mlp = nn.Sequential(nn.Linear(word_embedding_size, word_vec_size),
                                 nn.ReLU())
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type.upper())(word_vec_size, hidden_size, n_layers,
                                                 batch_first=True,
                                                 bidirectional=bidirectional,
                                                 dropout=dropout_p)
        self.num_dirs = 2 if bidirectional else 1
        self.fc_att = nn.Linear(20,20,bias=False)

    def forward(self, input_labels):
        """
        Inputs:
        - input_labels: Variable long (batch, seq_len)
        Outputs:
        - output  : Variable float (batch, max_len, hidden_size * num_dirs)
        - hidden  : Variable float (batch, num_layers * num_dirs * hidden_size)
        - embedded: Variable float (batch, max_len, word_vec_size)
        """
        input_labels=torch.cat(input_labels,dim=0)
        max_len = (input_labels != 0).sum(1).max()
        input_labels = input_labels[:, :max_len]
        if self.variable_lengths:
            input_lengths = (input_labels != 0).sum(1)  # Variable (batch, )

            # make ixs
            input_lengths_list = input_lengths.data.cpu().numpy().tolist()
            sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist()  # list of sorted input_lengths
            sort_ixs = np.argsort(input_lengths_list)[::-1].tolist()  # list of int sort_ixs, descending
            s2r = {s: r for r, s in enumerate(sort_ixs)}  # O(n)
            recover_ixs = [s2r[s] for s in range(len(input_lengths_list))]  # list of int recover ixs
            # print ('input_length_list:', max(input_lengths_list))
            # print('input_labels:',input_labels.size(1))
            assert max(input_lengths_list) == input_labels.size(1)

            # move to long tensor
            sort_ixs = input_labels.new(sort_ixs).long()  # Variable long
            recover_ixs = input_labels.new(recover_ixs).long()  # Variable long

            # sort input_labels by descending order
            input_labels = input_labels[sort_ixs]

        # embed
        embedded = self.embedding(input_labels.long())  # (n, seq_len, word_embedding_size)
        embedded = self.input_dropout(embedded)  # (n, seq_len, word_embedding_size)
        embedded = self.mlp(embedded)  # (n, seq_len, word_vec_size)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_input_lengths_list, batch_first=True)

        # forward rnn

        output, hidden = self.rnn(embedded)

        # recover
        if self.variable_lengths:

            # embedded (batch, seq_len, word_vec_size)
            embedded, _ = nn.utils.rnn.pad_packed_sequence(embedded, batch_first=True)
            embedded = embedded[recover_ixs]

            # recover rnn
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)  # (batch, max_len, hidden)
            output = output[recover_ixs]

            # recover hidden
            if self.rnn_type == 'lstm':
                hidden = hidden[0]  # we only use hidden states for the final hidden representation
            hidden = hidden[:, recover_ixs, :]  # (num_layers * num_dirs, batch, hidden_size)
            hidden = hidden.transpose(0, 1).contiguous()  # (batch, num_layers * num_dirs, hidden_size)
            hidden = hidden.view(hidden.size(0), -1)  # (batch, num_layers * num_dirs * hidden_size)

        return output, hidden, embedded


