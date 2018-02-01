''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Modules import BottleLinear as Linear
from transformer.Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang"

def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(Constants.PAD).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
    return pad_attn_mask

def get_attn_subsequent_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.'''
    assert seq.dim() == 2
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):

        super(Encoder, self).__init__()

        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos):
        # Word embedding look up
        bs, L = src_seq.size()
        enc_input = torch.zeros(bs, L, self.d_model)

        if src_seq.is_cuda:
            enc_input = enc_input.cuda()
        enc_input = enc_input.scatter_(
                2, torch.unsqueeze(src_seq.data, 2), 1.0)
        enc_input = torch.autograd.Variable(enc_input)
            # Position Encoding addition
        enc_input += self.position_enc(src_pos)
        enc_outputs, enc_slf_attns = [], []

        enc_output = enc_input
        enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask)
            enc_outputs += [enc_output]
            enc_slf_attns += [enc_slf_attn]

        return enc_outputs, enc_slf_attns

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(
            self, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):

        super(Decoder, self).__init__()
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(
            n_position, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_outputs):
        # Word embedding look up
        bs, L = tgt_seq.size()
        dec_input = torch.zeros(bs, L, self.d_model)
        if tgt_seq.is_cuda:
            dec_input = dec_input.cuda()
        dec_input = dec_input.scatter_(
                        2, torch.unsqueeze(tgt_seq.data, 2), 1.0)
        dec_input = torch.autograd.Variable(dec_input)
        # Position Encoding addition
        dec_input += self.position_enc(tgt_pos)

        dec_outputs, dec_slf_attns, dec_enc_attns = [], [], []

        # Decode
        dec_slf_attn_pad_mask = get_attn_padding_mask(tgt_seq, tgt_seq)
        dec_slf_attn_sub_mask = get_attn_subsequent_mask(tgt_seq)
        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)

        dec_enc_attn_pad_mask = get_attn_padding_mask(tgt_seq, src_seq)

        dec_output = dec_input
        for dec_layer, enc_output in zip(self.layer_stack, enc_outputs):
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=dec_slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_pad_mask)

            dec_outputs += [dec_output]
            dec_slf_attns += [dec_slf_attn]
            dec_enc_attns += [dec_enc_attn]

        return dec_outputs, dec_slf_attns, dec_enc_attns

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_tgt_vocab, n_max_seq, n_organisms,
            n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, d_k=64, d_v=64,
            dropout=0.1, d_organism=64,
            proj_share_weight=True, embs_share_weight=True):

        super(Transformer, self).__init__()
        self.encoder = Encoder(
            n_src_vocab, n_max_seq, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout)
        self.decoder = Decoder(
            n_tgt_vocab, n_max_seq, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout)
        self.tgt_word_proj = Linear(d_model + d_organism, n_tgt_vocab, bias=False)
        if d_organism > 0:
            self.organism_enc = nn.Embedding(n_organisms, d_organism)
        else:
            self.organism_enc = None

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, the dimensions of all module output shall be the same.'


    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        enc_freezed_param_ids = set(map(id, self.encoder.position_enc.parameters()))
        dec_freezed_param_ids = set(map(id, self.decoder.position_enc.parameters()))
        freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def forward(self, src, tgt, org):
        src_seq, src_pos = src
        tgt_seq, tgt_pos = tgt

        tgt_seq = tgt_seq[:, :-1]
        tgt_pos = tgt_pos[:, :-1]
        enc_outputs, enc_slf_attns = self.encoder(src_seq, src_pos)
        dec_outputs, dec_slf_attns, dec_enc_attns = self.decoder(
            tgt_seq, tgt_pos, src_seq, enc_outputs)
        dec_output = dec_outputs[-1]
        if self.organism_enc is not None:
            org_input = self.organism_enc(org)
            org_input = org_input.repeat(1, dec_output.size()[1], 1)
            dec_output = torch.cat((dec_output, org_input), dim=-1)
        seq_logit = self.tgt_word_proj(dec_output)

        return seq_logit.view(-1, seq_logit.size(2))
