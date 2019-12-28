# This contains a critic function with self attention function which
# takes a seq and image feature as input and output estimate Q value.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils

import copy
import math
import numpy as np

from .CaptionModel import CaptionModel
from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper
from misc.rewards import get_reward

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask, xt):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask, xt)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, xt):
        tgt_embedding = self.tgt_embed(tgt)
        tgt_embedding[:, 0, :] = xt
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class AttCriticModel(nn.Module):

    def make_model(self, src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            lambda x:x,
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab)
        )
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt):
        super(AttCriticModel, self).__init__()
        self.opt = opt
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.att_feat_size = opt.att_feat_size
        self.use_bn = getattr(opt, 'use_bn', 0)
        self.ss_prob = 0.0
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.input_encoding_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn==2 else ())))
        tgt_vocab = self.vocab_size + 1
        self.model = self.make_model(0, tgt_vocab,
                                     N=opt.num_layers,
                                     d_model=opt.input_encoding_size,
                                     d_ff=opt.rnn_size)

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, att_feats, crop, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            if crop:
                seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += 1

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def forward(self, seq, xt, att_feats, crop, opt, att_masks=None):
        #TODO: maybe include target sentences as input as well
        att_feats, seq, att_masks, seq_mask = self._prepare_feature(att_feats, crop, att_masks, seq)

        out = self.model(att_feats, seq, att_masks, seq_mask, xt)
        outputs = self.model.generator(out)

        return outputs


def critic_loss_fun(fc_feats, att_feats, att_masks, dp_model, critic_model, opt, data):
    gen_result, sample_logprobs_total = dp_model(fc_feats, att_feats, att_masks, opt={'sample_max': 0},
                                                 total_probs=True, mode='sample')
    gen_result_pad = torch.cat([gen_result.new_zeros(gen_result.size(0), 1, dtype=torch.long), gen_result], 1)
    critic_value = critic_model(gen_result_pad, fc_feats, att_feats, True, opt,
                                att_masks)  # batch, length, vocab
    critic_value_keep = critic_value
    critic_mask = critic_value.gather(2, gen_result.unsqueeze(2)).squeeze(2)  # batch, length
    critic_value = torch.cat(
        [torch.sum(critic_value * F.softmax(sample_logprobs_total, 2).detach(), 2)[:, 0].unsqueeze(1),
         critic_mask], 1)

    bellman_loss = bellman_loss_fun(critic_mask, critic_value_keep.detach(),
                                    F.softmax(sample_logprobs_total, 2))
    # TODO: target.
    reward, std = get_reward(data, gen_result, opt, critic=True)
    reward_cuda = torch.from_numpy(reward).float().cuda()
    mask = (gen_result > 0).float()
    mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
    crit_loss = (torch.sum((critic_value[:, 1:] - reward_cuda).pow(2) * mask) +
                 torch.sum((critic_value[:, 0] - reward_cuda[:, 0]).pow(2))) / (torch.sum(mask) + mask.size(0))
    crit_loss += opt.bl_weight * bellman_loss
    if opt.critic_var_penalty != 0:
        crit_loss += opt.critic_var_penalty * (
                critic_value_keep -
                critic_value_keep.mean(2).unsqueeze(2).repeat(1, 1, critic_value_keep.size()[2])).pow(2).mean()
    return crit_loss, reward, std


def target_critic_loss_fun(fc_feats, att_feats, att_masks, dp_model, critic_model, opt, data, target_critic,
                           target_actor, gen_result=None, sample_logprobs_total=None, reward=None):
    if gen_result is None:
        gen_result, sample_logprobs_total = target_actor(fc_feats, att_feats, att_masks, opt={'sample_max': 0},
                                                     total_probs=True, mode='sample')
    gen_result_pad = torch.cat([gen_result.new_zeros(gen_result.size(0), 1, dtype=torch.long), gen_result], 1)
    critic_value = critic_model(gen_result_pad, fc_feats, att_feats, True, opt,
                                att_masks)  # batch, length, vocab
    critic_value_keep = critic_value
    target_critic_value = target_critic(gen_result_pad, fc_feats, att_feats, True, opt, att_masks)
    critic_mask = critic_value.gather(2, gen_result.unsqueeze(2)).squeeze(2)  # batch, length
    critic_value = torch.cat(
        [torch.sum(critic_value * F.softmax(sample_logprobs_total, 2).detach(), 2)[:, 0].unsqueeze(1),
         critic_mask], 1)

    bellman_loss = bellman_loss_fun(critic_mask, target_critic_value.detach(),
                                    F.softmax(sample_logprobs_total, 2))
    if reward is None:
        reward, std = get_reward(data, gen_result, opt, critic=True)
    else:
        std = np.std(reward)

    reward_cuda = torch.from_numpy(reward).float().cuda()
    mask = (gen_result > 0).float()
    mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
    #TODO: Bellman loss + mask
    crit_loss = (torch.sum((critic_value[:, 1:] - reward_cuda).pow(2) * mask) +
                 torch.sum((critic_value[:, 0] - reward_cuda[:, 0]).pow(2))) / (torch.sum(mask) + mask.size(0))

    crit_loss += opt.bl_weight * bellman_loss

    if opt.critic_var_penalty != 0:
        crit_loss += opt.critic_var_penalty * (
                critic_value_keep -
                critic_value_keep.mean(2).unsqueeze(2).repeat(1, 1, critic_value_keep.size()[2])).pow(2).mean()
    return crit_loss, reward, std


def target_critic_loss_fun_mask(fc_feats, att_feats, att_masks, dp_model, critic_model, opt, data, target_critic,
                           target_actor, gen_result=None, sample_logprobs_total=None, reward=None):
    att_feats = torch.zeros_like(att_feats)
    if gen_result is None:
        gen_result, sample_logprobs_total = target_actor(fc_feats, att_feats, att_masks, opt={'sample_max': 0},
                                                     total_probs=True, mode='sample')
    gen_result_pad = torch.cat([gen_result.new_zeros(gen_result.size(0), 1, dtype=torch.long), gen_result], 1)
    xt = target_actor.img_embed(fc_feats)
    critic_value = critic_model(gen_result_pad, xt, att_feats, True, opt,
                                att_masks)  # batch, length, vocab
    critic_value_keep = critic_value
    target_critic_value = target_critic(gen_result_pad, xt, att_feats, True, opt, att_masks).detach()
    critic_mask = critic_value.gather(2, gen_result.unsqueeze(2)).squeeze(2)  # batch, length
    critic_value = torch.cat(
        [torch.sum(critic_value * F.softmax(sample_logprobs_total, 2).detach(), 2)[:, 0].unsqueeze(1),
         critic_mask], 1)
    mask = (gen_result > 0).float()
    mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
    bellman_loss = bellman_loss_fun(critic_mask, target_critic_value.detach(),
                                    F.softmax(sample_logprobs_total, 2), gen_result)
    if reward is None:
        reward, std = get_reward(data, gen_result, opt, critic=True)
    else:
        std = np.std(reward)
    reward_cuda = torch.from_numpy(reward).float().cuda()
    #print('critic', torch.max(critic_value_keep[:, 10, :], 1)[0])
    #print('reward', torch.max(reward_cuda, 1)[0])
    max_index = max_nonzero_index(gen_result)
    last_word_index = torch.min(torch.cat([max_index.unsqueeze(1) + 1, torch.ones_like(max_index).unsqueeze(1).cuda() * (gen_result.size()[1]-1)], 1), 1)[0]
    crit_loss = (critic_mask.gather(1, last_word_index.long().unsqueeze(1)) - reward_cuda[:, 0]).pow(2).sum() / float(mask.size()[0])
    # crit_loss = (torch.sum((critic_value[:, 1:] - reward_cuda).pow(2) * mask) +
    #              torch.sum((critic_value[:, 0] - reward_cuda[:, 0]).pow(2))) / (torch.sum(mask) + mask.size(0))
    crit_loss += opt.bl_weight * bellman_loss
    if opt.critic_var_penalty != 0:
        vocab_mask = mask.unsqueeze(2).repeat(1, 1, critic_value_keep.size()[2])
        crit_loss += opt.critic_var_penalty * (
            (critic_value_keep -
                critic_value_keep.mean(2).unsqueeze(2).repeat(1, 1, critic_value_keep.size()[2])) * vocab_mask).pow(2).sum() / vocab_mask.sum()
    return crit_loss, reward, std


def bellman_loss_fun(critic_mask, critic_value_keep, probs, gen_result):
    mask = (gen_result > 0).float()
    return ((critic_mask[:, :-1] - (critic_value_keep * probs.detach()).sum(2)[:, 1:]) * mask[:, :-1]).pow(2).sum() / mask[:, :-1].sum()

def max_nonzero_index(sentences):
    mask = (sentences != 0).float()
    rows, cols = mask.size()
    max_index = torch.max(mask * torch.arange(cols).float().cuda(), dim=1)[0]
    return max_index