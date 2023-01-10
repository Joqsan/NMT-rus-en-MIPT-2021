import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=emb_dim)

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src):

        embedded = self.embedding(src)

        embedded = self.dropout(embedded)

        output, enc_hidden = self.rnn(embedded)

        return output, enc_hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim

        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(num_embeddings=output_dim, embedding_dim=emb_dim)

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
        )

        self.out = nn.Linear(in_features=2 * hid_dim, out_features=output_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, dec_input, dec_hidden, enc_output):

        dec_input = dec_input.unsqueeze(0)

        dec_emb = self.dropout(self.embedding(dec_input))

        dec_output, dec_hidden = self.rnn(dec_emb, dec_hidden)

        ## Luong's attention
        enc_output_p = enc_output.permute(1, 2, 0)
        dec_output_p = dec_output.permute(1, 0, 2)

        # Alligment scores
        scores = dec_output_p.bmm(enc_output_p)

        # Attention scores
        alpha = F.softmax(scores, dim=-1)

        enc_output_pp = enc_output.permute(1, 0, 2)

        # Context vector
        context = alpha.bmm(enc_output_pp).permute(1, 0, 2)

        # Concatenation
        concat = torch.cat([context, dec_output], dim=-1)

        # Linear layer
        prediction = self.out(concat.squeeze(0))

        return prediction, dec_hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert (
            encoder.hid_dim == decoder.hid_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):

        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        enc_output, enc_hidden = self.encoder(src)

        input = trg[0, :]

        dec_hidden = enc_hidden

        for t in range(1, max_len):

            output, dec_hidden = self.decoder(input, dec_hidden, enc_output)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = trg[t] if teacher_force else top1

        return outputs
