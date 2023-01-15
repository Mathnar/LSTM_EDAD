import torch
import torch.nn as nn
from easydict import EasyDict as edict
from functools import partial


class Encoder(nn.Module):
    def __init__(self, n_features, latent_dim, rnn):
        super().__init__()

        self.rec_enc_f = rnn(n_features, latent_dim, batch_first=True)

    def forward(self, x):
        _, h_n = self.rec_enc_f(x)

        return h_n


class Decoder(nn.Module):
    def __init__(self, latent_dim, n_features, rnn_cell, device):
        super().__init__()
        self.n_features = n_features
        self.device = device
        self.rec_dec1 = rnn_cell(n_features, latent_dim)
        self.dense_dec1 = nn.Linear(latent_dim, n_features)

    def forward(self, h_0, seq_len):
        x = torch.tensor([], device = self.device)

        h_i = h_0.squeeze()
        x_i = self.dense_dec1(h_i)
        for i in range(0, seq_len):
            h_i = self.rec_dec1(x_i, h_i)
            x_i = self.dense_dec1(h_i)
            x = torch.cat([x, x_i], axis=0)
        return x.view(-1, seq_len, self.n_features)


class RecurrentDecoderLSTM(nn.Module):
    def __init__(self, latent_dim, n_features, rnn_cell, device):
        super().__init__()

        self.n_features = n_features
        self.device = device
        self.rec_dec1 = rnn_cell(n_features, latent_dim)
        self.dense_dec1 = nn.Linear(latent_dim, n_features)

    def forward(self, h_0, seq_len):
        # output var
        x = torch.tensor([], device = self.device)
        h_i = [h.squeeze() for h in h_0]

        # Reconstruct first el (ENC OUT)
        x_i = self.dense_dec1(h_i[0])
        print(x_i)

        # Rec the rest
        for i in range(0, seq_len):
            h_i = self.rec_dec1(x_i, h_i)
            x_i = self.dense_dec1(h_i[0])
            x = torch.cat([x, x_i], axis = 1)
        print('\nOutput of forward decoder: ', x)

        return x.view(-1, seq_len, self.n_features)


class RecurrentAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        ########## config
        self.config = config
        self.rnn, self.rnn_cell = self.get_rnn_type(self.config.rnn_type, self.config.rnn_act)
        self.decoder = self.get_decoder(self.config.rnn_type)
        self.latent_dim = self.config.latent_dim
        self.n_features = self.config.n_features
        self.device = self.config.device

        self.encoder = Encoder(self.n_features, self.latent_dim, self.rnn)
        self.decoder = self.decoder(self.latent_dim, self.n_features, self.rnn_cell, self.device)

    def forward(self, x):
        seq_len = x.shape[1]
        h_n = self.encoder(x)
        out = self.decoder(h_n, seq_len)

        return torch.flip(out, [1])

    @staticmethod
    def get_rnn_type(rnn_type, rnn_act=None):
        """Get recurrent layer and cell type"""
        if rnn_type == 'RNN':
            rnn = partial(nn.RNN, nonlinearity=rnn_act)
            rnn_cell = partial(nn.RNNCell, nonlinearity=rnn_act)

        else:
            rnn = getattr(nn, rnn_type)
            rnn_cell = getattr(nn, rnn_type + 'Cell')

        return rnn, rnn_cell

    @staticmethod
    def get_decoder(rnn_type):
        """Get recurrent decoder type"""
        if rnn_type == 'LSTM':
            decoder = RecurrentDecoderLSTM
        else:
            decoder = Decoder
        return decoder

