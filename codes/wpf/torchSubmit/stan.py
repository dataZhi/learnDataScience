import torch
from torch import nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Encoder(nn.Module):
    """
    capture relevant information from the history input,
    and generate an encoder_outputs and hidden vector
    """

    def __init__(self, enc_size, embedding=None, enc_hidden=64, dec_hidden=64, mode='GRU',
                 bidirectional=True, n_neighbors=10, dim_head=4, num_heads=8, withSA=True):
        super().__init__()
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden

        self.mode = mode
        self.bidirectional = bidirectional
        self.num_heads = num_heads

        self.withSA = withSA

        self.embedding = embedding
        embedding_dim = embedding.embedding_dim if embedding else 0
        self.enc_size = embedding_dim + enc_size

        if withSA:
            self.spatio_att = SpatioAttention(n_neighbors, dim_head, num_heads)
            self.enc_size -= (n_neighbors - 1) * dim_head

        if mode == "GRU":
            self.rnn = nn.GRU(self.enc_size, self.enc_hidden, bidirectional=bidirectional)
        elif mode == "LSTM":
            self.rnn = nn.LSTM(self.enc_size, self.dec_hidden, bidirectional=bidirectional)
        elif mode == "TCN":
            self.rnn = TemporalConvNet(num_inputs=self.enc_size, num_channels=[64] * 10)
        else:
            raise Exception(f"mode should be one of ('GRU', 'LSTM', 'TCN'), and got {mode}!")

        output_size = enc_hidden * 2 if self.bidirectional else enc_hidden
        self.fc_output = nn.Sequential(
            nn.Linear(output_size, dec_hidden),
            nn.ReLU()
        )
        self.fc_hidden = nn.Sequential(
            nn.Linear(output_size, dec_hidden),
            nn.ReLU()
        )

    def forward(self, data):
        """
        :param data: [[turbID, cur_period], x_enc, x_dec, y]
                x_enc: (batch, seq_len, enc_size), default enc_size=54
                x_dec: (batch, seq_len, dec_size), default dec_size=29
                y: (batch, output_len, out_var)
        """
        turb_id = data[0][:, 0]
        if self.withSA:
            data = self.spatio_att(data)
        x = data[1].permute(1, 0, 2).float()  # (batch, seq_len, enc_size) => (seq_len, batch, enc_size)
        # x = x.nan_to_num()  # fill na with 0
        seq_len = x.size(0)

        if self.embedding is not None:
            turb_embedding = self.embedding(turb_id).unsqueeze(0).repeat(seq_len, 1, 1)
            x = torch.cat((turb_embedding, x), dim=2).float()

        if self.mode in ("GRU", "LSTM"):
            output, hidden = self.rnn(x)  # hidden in LSTM: (hn, cn), hidden in GRU: hn
            if self.mode == 'LSTM':
                hidden = hidden[0]

            if self.bidirectional:
                hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=-1)
                hidden = hidden.unsqueeze(0)  # (1, batch, 2*hidden_size)
        else:
            x = x.permute(1, 2, 0)  # (seq_len, batch, enc_size) => (batch, enc_size, seq_len)
            output = self.rnn(x)
            output = output.permute(2, 0, 1)  # (batch, feature, seq_len) => (seq_len, batch, feature)
            hidden = output[-1:, :, :]  # (1, batch, feature)

        output = self.fc_output(output)  # (seq_len, batch, dec_hidden)
        hidden = self.fc_hidden(hidden)  # (1, batch, dec_hidden)

        return output, hidden


class Attention(nn.Module):
    """
    one-head attention mechanism
    """

    def __init__(self, n_dim):
        super().__init__()
        self.n_dim = n_dim
        self.linear_q = nn.Linear(n_dim, n_dim)
        self.linear_k = nn.Linear(n_dim, n_dim)
        self.linear_v = nn.Linear(n_dim, n_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        """
        multi-head attention
        :param query: (batch, seq_len, feature)
        :param key: (batch, seq_len, feature)
        :param value: (batch, seq_len, feature)
        :return: (batch, seq_len, feature)
        """
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)
        att_weight = torch.bmm(query, key.transpose(1, 2))  # (batch, seq_len, seq_len)
        att_weight = self.softmax(att_weight)
        return torch.bmm(att_weight, value)  # (batch, seq_len, feature)


class SpatioAttention(nn.Module):
    """
    fusion neighbor turbine's info with multi-head attention
    """

    def __init__(self, n_neighbor=10, dim_head=4, num_heads=8):
        super().__init__()
        self.n_neighbor = n_neighbor
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.attentions = nn.ModuleList([Attention(dim_head) for _ in range(num_heads)])
        self.linear_out = nn.Linear(dim_head * num_heads, dim_head)

    def forward(self, data):
        """
        :param data: [[turbID, cur_period], x_enc, x_dec, y]
                x_enc: (batch, seq_len, enc_size), default enc_size=54
                x_dec: (batch, seq_len, dec_size), default dec_size=29
                y: (batch, output_len, out_var)
        """
        x_enc = data[1]
        bs, seq_len, input_size = x_enc.shape
        split_idx = input_size - self.n_neighbor * self.dim_head
        # features of each turb own
        x_left = x_enc[:, :, :split_idx]
        x = x_enc[:, :, split_idx:]
        key = x.view(bs * seq_len, self.n_neighbor, self.dim_head)  # (bs*seq_len, 10, dim_head)
        query = key[:, :1, :]  # (bs*seq_len, 1, dim_head)
        x = [att(query, key, key) for att in self.attentions]
        x = self.linear_out(torch.dstack(x))  # (bs*seq_len, 1, dim_head)
        x = x.view(bs, seq_len, -1)  # (batch, seq_len, feature)
        data[1] = torch.dstack((x_left, x))
        return data


class Decoder(nn.Module):
    """
    forecast wind power one-by-one
    """

    def __init__(self, dec_size, embedding=None, enc_hidden=64, dec_hidden=64, out_var=1, mode='GRU',
                 output_len=288, num_heads=8, withTA=2, hidden_as_input=False, hidden_as_output=True):
        super().__init__()
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden
        self.output_len = output_len
        self.out_var = out_var
        self.mode = mode

        self.embedding = embedding
        embedding_dim = embedding.embedding_dim if embedding else 0
        self.dec_size = embedding_dim + dec_size + (dec_hidden if hidden_as_input else 0)

        self.withTA = withTA
        self.hidden_as_input = hidden_as_input
        self.hidden_as_output = hidden_as_output

        if mode == 'GRU':
            self.rnn = nn.GRU(self.dec_size, dec_hidden)
        if mode == 'LSTM':
            self.rnn = nn.LSTM(self.dec_size, dec_hidden)

        if withTA == 2:
            self.output = TemporalAttention2(dec_hidden=dec_hidden, out_var=out_var, num_heads=num_heads)
        elif withTA == 1:
            self.output = TemporalAttention(dec_hidden=dec_hidden, out_var=out_var, num_heads=num_heads)
        elif self.hidden_as_output:
            self.output = nn.Linear(dec_hidden * 2, out_var)
        else:
            self.output = nn.Linear(dec_hidden, out_var)

    def forward(self, data, encoder_output, hidden):
        """
        :param data: [[turbID, cur_period], x_enc, x_dec, y]
                x_enc: (batch, seq_len, enc_size), default enc_size=54
                x_dec: (batch, seq_len, dec_size), default dec_size=29
                y: (batch, output_len)
        :param encoder_output: (seq_len, batch, dec_hidden)
        :param hidden: (1, bs, dec_hidden)
        """
        turb_id = data[0][:, 0]

        x_dec = data[2].permute(1, 0, 2)  # (batch, output_len+1, dec_size) => (output_len+1, batch, dec_size)
        if self.hidden_as_input:
            x_hidden = hidden.repeat(self.output_len+1, 1, 1)  # (1, bs, dec_hidden) => (output_len+1, bs, dec_hidden)
            x_dec = torch.cat((x_hidden, x_dec), dim=2)  # (batch, output_len+1, dec_size+dec_hidden)
        # x_dec[x_dec != x_dec] = 0  # fill na with 0
        x = x_dec[[0]]  # initial status of decoder: [Hour, Distance, Patv]
        cell_state = torch.zeros_like(hidden).unsqueeze(0)
        hidden_encoder = hidden[:]

        output_list = []
        for i in range(1, self.output_len + 1):
            if self.embedding is not None:
                turb_embedding = self.embedding(turb_id).unsqueeze(0)
                x = torch.cat((turb_embedding, x), dim=2).float()

            if self.mode == 'GRU':
                output, hidden = self.rnn(x, hidden)
            elif self.mode == 'LSTM':
                output, (hidden, cell_state) = self.rnn(x, (hidden, cell_state))

            if self.withTA > 0:
                output = self.output(encoder_output, hidden)  # output: (bs, out_var)
            elif self.hidden_as_output:
                hidden_aug = torch.cat((hidden, hidden_encoder), dim=2)  # (1, bs, dec_hidden * 2)
                output = self.output(hidden_aug.squeeze(0))  # output: (bs, out_var)
            else:
                output = self.output(hidden.squeeze(0))  # output: (bs, out_var)

            output = output.unsqueeze(1)  # output: (bs, out_var) => (bs, 1, out_var)
            output_list.append(output)

            x = x_dec[i, :, :-self.out_var].unsqueeze(1)  # (bs, st_idx) => (bs, 1, st_idx) [Hour0~23, Distance1~4]
            x = torch.cat((x, output), dim=2)  # (bs, 1, st_idx+out_var) [Hour0~23, Distance1~4, ~Patv]
            x = x.permute(1, 0, 2)  # (1, batch, dec_size)

        y_pred = torch.cat(output_list, dim=1).float()  # (batch, output_len, out_var)
        return y_pred


class TemporalAttention(nn.Module):
    """
    capture relevant information from the outputs of encoder,
    and then concat with its own hidden to project the final forecast
    i.e.:
        context = attention(encoder_outputs, hidden)
        y_pred = f([context, hidden])
    """

    def __init__(self, dec_hidden=64, out_var=1, num_heads=8):
        super().__init__()
        self.dec_hidden = dec_hidden
        self.num_heads = num_heads
        self.attentions = nn.ModuleList([Attention(dec_hidden) for _ in range(num_heads)])
        self.linear_out = nn.Sequential(
            nn.Linear(dec_hidden * num_heads, 32),
            nn.ReLU(),
            nn.Linear(32, out_var)
        )

    def forward(self, encoder_output, hidden):
        """
        query: hidden, key: output, value: encoder_output
        :param encoder_output: (seq_len, bs, dec_hidden)
        :param hidden: (1, bs, dec_hidden)
        :return: final forecast output, (bs, 1)
        """
        # hidden: (1, bs, dec_hidden) => (bs, 1, dec_hidden)
        hidden = hidden.permute(1, 0, 2)
        # encoder_output: (seq_len, bs, dec_hidden) => (bs, seq_len, dec_hidden)
        encoder_output = encoder_output.permute(1, 0, 2)
        # process with each head
        x = [att(hidden, encoder_output, encoder_output) for att in self.attentions]
        x = torch.dstack(x).squeeze(1)  # (batch, num_heads*dec_hidden)
        x = self.linear_out(x)  # (batch, out_var)

        return x


class TemporalAttention2(nn.Module):
    """
    capture relevant information from the outputs of encoder,
    and then concat with its own hidden to project the final forecast
    i.e.:
        context = attention(encoder_outputs, hidden)
        y_pred = f([context, hidden])
    """

    def __init__(self, dec_hidden=64, out_var=1, num_heads=8):
        super().__init__()
        self.dec_hidden = dec_hidden
        self.num_heads = num_heads
        assert dec_hidden % num_heads == 0, "dec_hidden // num_heads should be 0"
        self.dim_head = dec_hidden // num_heads
        self.linear_k = nn.Linear(self.dim_head, self.dim_head)
        self.softmax = nn.Softmax(dim=-1)
        self.linear_out = nn.Sequential(
            nn.Linear(dec_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, out_var)
        )

    def forward(self, encoder_output, hidden):
        """
        query: hidden, key: output, value: encoder_output
        :param encoder_output: (seq_len, bs, dec_hidden)
        :param hidden: (1, bs, dec_hidden)
        :return: final forecast output, (bs, 1)
        """
        seq_len, bs, n_feature = encoder_output.shape
        assert n_feature == self.dec_hidden, "feature dimension dont match"
        # query: (1, bs, dec_hidden) => (1, bs*num_heads, dim_head) => (bs*num_heads, 1, dim_head)
        query = hidden.reshape(1, bs * self.num_heads, self.dim_head).permute(1, 0, 2)
        # value: (seq_len, bs, dec_hidden) => (seq_len, bs*num_heads, dim_head) => (bs*num_heads, seq_len, dim_head)
        value = encoder_output.reshape(seq_len, bs * self.num_heads, self.dim_head).permute(1, 0, 2)
        # key: (bs*num_heads, seq_len, dim_head) => (bs*num_heads, dim_head, seq_len)
        key = self.linear_k(value).permute(0, 2, 1)
        # multi-head attention
        att_weight = torch.bmm(query, key)  # (bs*num_heads, 1, seq_len)
        att_weight = self.softmax(att_weight)
        output = torch.bmm(att_weight, value)  # (bs*num_heads, 1, dim_head)
        output = output.squeeze(1).reshape(bs, self.dec_hidden)

        return self.linear_out(output)  # (bs, 1)


class STAN(nn.Module):
    def __init__(self, enc_size, dec_size, embedding_dim=4, enc_hidden=64, dec_hidden=64, output_len=288, out_var=1,
                 n_turb=134, n_neighbors=10, num_heads=8, encoder_mode='GRU', decoder_mode='GRU', bidirectional=False,
                 withSA=False, withTA=False, hidden_as_input=False, hidden_as_output=True):
        super().__init__()
        self.enc_size = enc_size
        self.dec_size = dec_size
        self.embedding = nn.Embedding(n_turb, embedding_dim) if embedding_dim and n_turb > 1 else None
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden
        self.output_len = output_len
        self.out_var = out_var

        self.encoder = Encoder(
            enc_size=enc_size,
            embedding=self.embedding,
            enc_hidden=enc_hidden,
            dec_hidden=dec_hidden,
            n_neighbors=n_neighbors,
            mode=encoder_mode,
            bidirectional=bidirectional,
            num_heads=num_heads,
            withSA=withSA
        )
        self.decoder = Decoder(
            dec_size=dec_size,
            embedding=self.embedding,
            enc_hidden=enc_hidden,
            dec_hidden=dec_hidden,
            out_var=out_var,
            mode=decoder_mode,
            output_len=output_len,
            num_heads=num_heads,
            withTA=withTA,
            hidden_as_input=hidden_as_input,
            hidden_as_output=hidden_as_output
        )

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)

    def forward(self, data):
        """
        :param data: [[turbID, cur_period], x_enc, x_dec, y]
                x_enc: (batch, seq_len, enc_size), default enc_size=54
                x_dec: (batch, seq_len, dec_size), default dec_size=29
                y: (batch, output_len, out_var)
        """
        data[1] = data[1].nan_to_num(0)
        data[2] = data[2].nan_to_num(0)

        output, hidden = self.encoder(data)
        y_pred = self.decoder(data, output, hidden)

        return y_pred
