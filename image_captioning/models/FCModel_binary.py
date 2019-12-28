from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
import os
from six.moves import cPickle
import numpy as np
import copy

from time import time
from .CaptionModel import CaptionModel

from collections import OrderedDict

import sys
sys.path.append("/home1/06008/xf993/self-critical.pytorch/cider")
sys.path.append("/home/ziyu/self-critical.pytorch/cider")
from pyciderevalcap.ciderD.ciderD import CiderD
sys.path.append("/home1/06008/xf993/self-critical.pytorch/coco-caption")
sys.path.append("/home/ziyu/self-critical.pytorch/coco-caption")
from pycocoevalcap.bleu.bleu import Bleu

CiderD_scorer = None
Bleu_scorer = None
epsilon = 1e-18
def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

class LSTMCore(nn.Module):
    def __init__(self, opt):
        super(LSTMCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm

        # Build a LSTM
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

    def forward(self, xt, state):

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = torch.max(\
            all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size),
            all_input_sums.narrow(1, 4 * self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state

class FCModel_binary(CaptionModel):
    def __init__(self, opt):
        super(FCModel_binary, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size

        self.ss_prob = 0.0 # Schedule sampling probability

        self.img_embed = nn.Linear(self.fc_feat_size, self.input_encoding_size)
        self.core = LSTMCore(opt)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size)

        self.init_weights()
        with open(os.path.join(opt.binary_tree_coding_dir)) as f:
            binary_tree_coding = cPickle.load(f)
        self.depth = binary_tree_coding['depth']
        self.vocab2code = binary_tree_coding['vocab2code']
        self.phi_list = binary_tree_coding['phi_list']
        self.stop_list = binary_tree_coding['stop_list']
        self.code2vocab = binary_tree_coding['code2vocab']
        self.cluster_size = binary_tree_coding.get('cluster_size')

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'lstm':
            return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                    weight.new_zeros(self.num_layers, bsz, self.rnn_size))
        else:
            return weight.new_zeros(self.num_layers, bsz, self.rnn_size)

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        # sequence is padded with zero in the beginning
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []

        for i in range(seq.size(1)):
            if i == 0:
                xt = self.img_embed(fc_feats)
            else:
                it = seq[:, i-1].clone()
                # break if all the sequences end
                if i >= 2 and seq[:, i-1].sum() == 0:
                    break
                xt = self.embed(it)

            output, state = self.core(xt, state)
            output = self.logit(output)
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs[1:]], 1).contiguous()

    def get_logprobs_state(self, it, state):
        # 'it' is contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, state)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state


    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}, total_probs=False):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        seq = fc_feats.new_zeros(batch_size, self.seq_length, dtype=torch.long).cuda()
        output_logit = fc_feats.new_zeros(batch_size, self.seq_length, self.depth).cuda()
        unfinished = fc_feats.new_ones(batch_size, dtype=torch.uint8).cuda()
        binary_code = fc_feats.new_ones(batch_size, self.seq_length, self.depth, dtype=torch.uint8).cuda()
        for t in range(self.seq_length + 2):
            if t == 0:
                xt = self.img_embed(fc_feats)
            else:
                if t == 1: # input <bos>
                    it = fc_feats.data.new(batch_size).long().zero_()
                xt = self.embed(it)

            output, state = self.core(xt, state)
            phi = self.logit(output) # phi: batch, vocab-1
            # sample the next_word
            if t == self.seq_length + 1: # skip if we achieve maximum length
                break
            if t >= 1:
                mask_depth = unfinished.clone()
                code_sum = np.zeros(batch_size)
                for i in range(self.depth):
                    if i == 0: #two level mask ugly!
                        phi_index = torch.zeros(batch_size, 1).long().cuda()
                    if i > 0:
                        if mask_depth.sum() == 0:
                            break
                        phi_index = torch.from_numpy(map_phi(self.phi_list[i],
                                            np.expand_dims(code_sum * unfinished.cpu().numpy(), 1))).long().cuda()
                    phi_depth = phi.gather(1, phi_index)
                    if sample_max:
                        pi = 0.5
                    else:
                        pi = torch.from_numpy(np.random.uniform(size=[batch_size, 1])).float().cuda()
                    it_depth = (pi > binary_softmax(phi_depth))  # int8
                    binary_code[:, t - 1, i] = it_depth.squeeze(1) * mask_depth
                    output_logit[:, t - 1, i] = (phi_depth * it_depth.type_as(phi_depth) - LogOnePlusExp(phi_depth)).squeeze(1) * mask_depth
                    code_sum += (it_depth.squeeze(1) * mask_depth).cpu().numpy() * np.power(2, i)  # batch
                    if len(self.stop_list[i]) != 0:
                        mask_depth *= torch.from_numpy(unfinished_fun(code_sum, self.stop_list[i])).cuda().type_as(mask_depth)
                it = torch.from_numpy(code2vocab_fun(code_sum, self.code2vocab)).cuda().long()
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                it = it * unfinished.type_as(it)
                seq[:, t-1] = it
                if unfinished.sum() == 0:
                    break
        return seq, output_logit

    def word_completion(self, depth, phi, unfinished, code_sum, mask_depth, sample_max):
        batch_size = unfinished.size(0)
        for i in range(depth + 1, self.depth):
            if mask_depth.sum() == 0:
                break
            phi_index = torch.from_numpy(map_phi(self.phi_list[i],
                                                 np.expand_dims(code_sum * unfinished.cpu().numpy(), 1))).long().cuda()
            phi_depth = phi.gather(1, phi_index)
            if sample_max:
                pi = 0.5
            else:
                pi = torch.from_numpy(np.random.uniform(size=[batch_size, 1])).float().cuda()
            it_depth = (pi > binary_softmax(phi_depth))
            code_sum += (it_depth.squeeze(1) * mask_depth).cpu().numpy() * np.power(2, i)
            if len(self.stop_list[i]) != 0:
                mask_depth *= torch.from_numpy(unfinished_fun(code_sum, self.stop_list[i])).cuda().type_as(
                    mask_depth)
        return code_sum

    def sentence_completion(self, step, unfinished, it, state, seqs, sample_max):
        batch_size = seqs.size(0)
        seqs_cp = seqs.clone()
        unfinished_cp = unfinished.clone()
        for t in range(step + 1, self.seq_length + 1):
            if unfinished.sum() == 0:
                break
            xt = self.embed(it)
            output, state = self.core(xt, state)
            phi = self.logit(output)
            mask_depth = unfinished_cp.clone()
            code_sum = np.zeros(batch_size)
            for i in range(self.depth):
                if i == 0:
                    phi_index = torch.zeros(batch_size, 1).long().cuda()
                else:
                    if mask_depth.sum() == 0:
                        break
                    phi_index = torch.from_numpy(map_phi(self.phi_list[i],
                                                         np.expand_dims(code_sum * unfinished_cp.cpu().numpy(),
                                                                        1))).long().cuda()
                phi_depth = phi.gather(1, phi_index)
                if sample_max:
                    pi = 0.5
                else:
                    pi = torch.from_numpy(np.random.uniform(size=[batch_size, 1])).float().cuda()
                it_depth = (pi > binary_softmax(phi_depth))
                code_sum += (it_depth.squeeze(1) * mask_depth).cpu().numpy() * np.power(2, i)
                if len(self.stop_list[i]) != 0:
                    mask_depth *= torch.from_numpy(unfinished_fun(code_sum, self.stop_list[i])).cuda().type_as(
                        mask_depth)
            it = torch.from_numpy(code2vocab_fun(code_sum, self.code2vocab)).cuda().long()
            unfinished_cp = unfinished_cp * (it > 0)
            it = it * unfinished_cp.type_as(it)
            seqs_cp[:, t - 1] = it
        return seqs_cp


    def get_arm_loss_binary_fast(self, fc_feats, att_feats, att_masks, opt, data, loader):
        sample_max = 0
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        seq = fc_feats.new_zeros(batch_size, self.seq_length, dtype=torch.long).cuda()
        output_logit = fc_feats.new_zeros(batch_size, self.seq_length, self.depth).cuda()
        unfinished = fc_feats.new_ones(batch_size, dtype=torch.uint8).cuda()
        binary_code = fc_feats.new_ones(batch_size, self.seq_length, self.depth, dtype=torch.uint8).cuda()
        loss = torch.zeros([]).float().cuda()
        mask_sum = 0
        for t in range(self.seq_length + 2):
            if t == 0:
                xt = self.img_embed(fc_feats)
            else:
                if t == 1: # input <bos>
                    it = fc_feats.data.new(batch_size).long().zero_()
                xt = self.embed(it)

            output, state = self.core(xt, state)
            phi = self.logit(output) # phi: batch, vocab-1
            # sample the next_word
            if t == self.seq_length + 1: # skip if we achieve maximum length
                break
            if t >= 1:
                mask_depth = unfinished.clone() # batch,
                code_sum = np.zeros(batch_size)
                # things to concat across depths:
                seqs_arm_list = []
                state_arm_list = []
                it_arm_list = []
                unfinished_arm_list = []
                need_run_index_list = []
                pi_list = []
                phi_depth_list = []
                mask_depth_list = []
                pseudo_num_list = []
                tic = time()
                for i in range(self.depth):
                    if i == 0: #two level mask
                        phi_index = torch.zeros(batch_size, 1).long().cuda()
                    if i > 0:
                        if mask_depth.sum() == 0:
                            break
                        phi_index = torch.from_numpy(map_phi(self.phi_list[i],
                                            np.expand_dims(code_sum * unfinished.cpu().numpy(), 1))).long().cuda()
                    phi_depth = phi.gather(1, phi_index)
                    ## complete words, concat seqs, index, and phi_depth, pi, state, unfinished, it,
                    mask_depth_list.append(mask_depth.clone())
                    phi_depth_list.append(phi_depth.clone())
                    pre_seq = seq.clone()
                    # depth = i
                    pi_i = torch.from_numpy(np.random.uniform(size=[batch_size, 1])).float().cuda()
                    it_1 = (pi_i > binary_softmax(phi_depth))
                    it_0 = (pi_i < binary_softmax(-phi_depth))
                    pseudo_actions = torch.cat([it_1, it_0], 1)  # batch, 2
                    if mask_depth.sum() != batch_size:
                        pseudo_actions[(1 - mask_depth), :] = 0
                    need_run_index = (pseudo_actions[:, 1] != pseudo_actions[:, 0])
                    pseudo_num_list.append(need_run_index.sum() * 2)
                    if need_run_index.sum() == 0:
                        pi_list.append('')
                        need_run_index_list.append('')
                    else:
                        seqs_arm, phi_arm, unfinished_arm, state_arm, binary_code_arm, code_sum_arm, mask_depth_arm, it_depth_arm = \
                            concatenate_arm(need_run_index, pre_seq, phi, unfinished, state, binary_code, code_sum,
                                            mask_depth,
                                            it_1, it_0)
                        code_sum_arm += (it_depth_arm.squeeze(1) * mask_depth_arm).cpu().numpy() * np.power(2, i)
                        if len(self.stop_list[i]) != 0:
                            mask_depth_arm *= torch.from_numpy(
                                unfinished_fun(code_sum_arm, self.stop_list[i])).cuda().type_as(mask_depth_arm)
                        code_sum_arm = self.word_completion(i, phi_arm, unfinished_arm, code_sum_arm, mask_depth_arm,
                                                            sample_max)
                        it_arm = torch.from_numpy(code2vocab_fun(code_sum_arm, self.code2vocab)).cuda().long()
                        unfinished_arm = unfinished_arm * (it_arm > 0)
                        it_arm = it_arm * unfinished_arm.type_as(it_arm)
                        seqs_arm[:, t - 1] = it_arm
                        ## concat:
                        pi_list.append(pi_i.clone())
                        need_run_index_list.append(need_run_index.clone())
                        unfinished_arm_list.append(unfinished_arm.clone())
                        it_arm_list.append(it_arm.clone())
                        state_arm_list.append(state_arm) #TODO: figure out where copy is needed
                        seqs_arm_list.append(seqs_arm.clone())
                    mask_sum += mask_depth.sum()
                    if sample_max:
                        pi = 0.5
                    else:
                        pi = torch.from_numpy(np.random.uniform(size=[batch_size, 1])).float().cuda()
                    it_depth = (pi > binary_softmax(phi_depth))  # int8
                    code_sum += (it_depth.squeeze(1) * mask_depth).cpu().numpy() * np.power(2, i)  # batch
                    if len(self.stop_list[i]) != 0:
                        mask_depth *= torch.from_numpy(unfinished_fun(code_sum, self.stop_list[i])).cuda().type_as(mask_depth)
                # complete the seqs together, and compute the reward, and f_delta, and loss
                #print('word completion time ', time() - tic)
                #print('pseudo action num ', pseudo_num_list)
                if len(unfinished_arm_list) > 0:
                    unfinished_arm_straight = straight_fun(unfinished_arm_list)
                    it_arm_straight = straight_fun(it_arm_list)
                    seqs_arm_straight = straight_fun(seqs_arm_list)
                    for i, item in enumerate(state_arm_list):
                        if i == 0:
                            state_h_arm_straight = item[0]
                            state_c_arm_straight = item[1]
                        else:
                            state_h_arm_straight = torch.cat([state_h_arm_straight, item[0]], 1)
                            state_c_arm_straight = torch.cat([state_c_arm_straight, item[0]], 1)
                    state_arm_straight = (state_h_arm_straight, state_c_arm_straight)
                    tic = time()
                    seqs_arm_completed = self.sentence_completion(t, unfinished_arm_straight, it_arm_straight, state_arm_straight, seqs_arm_straight, sample_max)
                    #print('sentence completion time ', time() - tic)
                    tic = time()
                    start_index = 0
                    for i in range(len(mask_depth_list)):
                        pseudo_num = pseudo_num_list[i]
                        if pseudo_num != 0:
                            arm_metric_value = reward_function(data, batch_size,
                                                               seqs_arm_completed[start_index:(start_index+pseudo_num)],
                                                               need_run_index_list[i])
                            start_index += pseudo_num
                            f_delta = f_delta_fun(batch_size, need_run_index_list[i], pi_list[i], arm_metric_value)
                            loss -= (torch.from_numpy(f_delta).cuda().float() * phi_depth_list[i].squeeze(
                                1) * mask_depth_list[i].float()).sum()
                    #print('reward time ', time() - tic)
                it = torch.from_numpy(code2vocab_fun(code_sum, self.code2vocab)).cuda().long()
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                it = it * unfinished.type_as(it)
                seq[:, t-1] = it
                if unfinished.sum() == 0:
                    break
        return loss / mask_sum


def concatenate_arm(need_run_index, pre_seq, phi, unfinished, state, binary_code, code_sum, mask_depth,
                    it_1, it_0):
    seqs_arm = torch.cat([pre_seq[need_run_index, :], pre_seq[need_run_index, :]], 0) #batch, length
    phi_arm = torch.cat([phi[need_run_index, :], phi[need_run_index, :]], 0) #batch,
    unfinished_arm = torch.cat([unfinished[need_run_index], unfinished[need_run_index]], 0) #batch,
    state_h, state_c = state
    state_h_arm = torch.cat([state_h[:, need_run_index, :], state_h[:, need_run_index, :]], 1)
    state_c_arm = torch.cat([state_c[:, need_run_index, :], state_c[:, need_run_index, :]], 1)
    state_arm = (state_h_arm, state_c_arm)
    binary_code_arm = torch.cat([binary_code[need_run_index, :, :], binary_code[need_run_index, :, :]], 0)
    code_sum_arm = np.concatenate([code_sum[need_run_index.cpu().numpy().astype(bool)], code_sum[need_run_index.cpu().numpy().astype(bool)]], 0)
    mask_depth_arm = torch.cat([mask_depth[need_run_index], mask_depth[need_run_index]], 0)
    it_depth_arm = torch.cat([it_1[need_run_index, :], it_0[need_run_index, :]], 0)
    return seqs_arm, phi_arm, unfinished_arm, state_arm, binary_code_arm, code_sum_arm, mask_depth_arm, it_depth_arm


def reward_function(data, batch_size, seqs, target_index):
    seq_per_img = batch_size // len(data['gts'])
    gts = OrderedDict()
    seqs_size = seqs.size(0)

    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]
    seqs = seqs.data.cpu().numpy()
    res_ = []
    gts_new = {}
    for i in range(seqs_size):
        res_.append({'image_id':i, 'caption': [array_to_str(seqs[i])]})
        i_index = int(i % (seqs_size / 2))
        gts_new[i] = gts[np.arange(batch_size)[target_index.cpu().numpy().astype(bool)][i_index] // seq_per_img]
    _, reward = CiderD_scorer.compute_score(gts_new, res_)
    return reward

def f_delta_fun(batch_size, nonzero_index, pi, reward):
    f_delta = np.zeros(shape=(batch_size))
    f_delta[nonzero_index.cpu().numpy().astype(bool)] = reward[0:(nonzero_index.sum().cpu().numpy())] - \
                                                        reward[(nonzero_index.sum().cpu().numpy()):]
    f_delta = f_delta * (pi.squeeze(1).cpu().numpy() - 0.5)
    return f_delta



def map_phi(phi, code_sum):
    # TODO:
    # phi: a dictionary,
    # code_sum, a matrix containing code_sum: batch, length
    phi_index = np.zeros_like(code_sum)
    batch, length = np.shape(code_sum)
    if len(phi) < batch * length:
        for i in phi:
            phi_index[code_sum == i] = phi[i]
    else:
        for i in range(batch):
            for j in range(length):
                if phi.get(code_sum[i, j]) is not None:
                    phi_index[i, j] = phi[code_sum[i, j]]
                else:
                    phi_index[i, j] = 0
    return phi_index

def unfinished_fun(code_sum, stop_list_i):
    # code_sum: batch, stop_list_i: list of code_sum that should stop
    batch_size = np.shape(code_sum)[0]
    unfinished = np.zeros(batch_size)
    for i in range(batch_size):
        unfinished[i] = np.sum(code_sum[i] == np.array(stop_list_i)) == 0
    return unfinished

def code2vocab_fun(code_sum, code2vocab):
    batch_size = np.shape(code_sum)[0]
    vocab_return = np.zeros(batch_size)
    for i in range(batch_size):
        vocab_return[i] = code2vocab[code_sum[i]]
    return vocab_return

def straight_fun(input):
    for i, item in enumerate(input):
        if i == 0:
            output = item
        else:
            output = torch.cat([output, item], 0)
    return output

def binary_softmax(phi):
    result = torch.zeros_like(phi).float().cuda()
    result[phi > 0] = torch.exp(-phi[phi > 0]) / (1.0 + torch.exp(-phi[phi > 0]))
    result[phi <= 0] = 1.0 / (1.0 + torch.exp(phi[phi <= 0]))
    return result

def LogOnePlusExp(x):
    result = torch.zeros_like(x).float().cuda()
    result[x > 0] = (x[x > 0] + torch.log1p(torch.exp(-x[x > 0])))
    result[x <= 0] = torch.log1p(torch.exp(x[x <= 0]))
    return result