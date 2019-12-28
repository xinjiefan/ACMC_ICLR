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

class FCModel_two_layer(CaptionModel):
    def __init__(self, opt):
        super(FCModel_two_layer, self).__init__()
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
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        seq = fc_feats.new_zeros(batch_size, self.seq_length, dtype=torch.long).cuda()
        output_logit = fc_feats.new_zeros(batch_size, self.seq_length, self.depth).cuda()
        unfinished = fc_feats.new_ones(batch_size, dtype=torch.uint8).cuda()
        n_cluster = len(self.cluster_size)
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
                code_sum = torch.zeros(batch_size, 1).float().cuda()
                probs_step_1 = F.softmax(torch.cat([phi[:, :(n_cluster-1)], torch.zeros(batch_size, 1).float().cuda()], 1), 1)
                if sample_max:
                    it_1 = torch.max(probs_step_1.data, 1)[1].view(-1).cuda().long()
                else:
                    it_1 = torch.multinomial(probs_step_1.data, 1).cuda().squeeze(1)
                it_2 = torch.zeros_like(it_1).cuda()
                start = n_cluster - 1
                for i in range(n_cluster):
                    if self.cluster_size[i] != 1:
                        index = it_1 == i
                        if index.sum() != 0:
                            probs_step_2 = F.softmax(torch.cat([phi[index, start:(start+self.cluster_size[i]-1)], torch.zeros(index.sum(), 1).float().cuda()], 1), 1)
                            if sample_max:
                                it_2[index] = torch.max(probs_step_2.data, 1)[1].view(-1).cuda().long()
                            else:
                                it_2[index] = torch.multinomial(probs_step_2.data, 1).cuda().squeeze(1)
                    start = start + self.cluster_size[i]-1
                code_sum = it_1 * (self.vocab_size + 1) + it_2
                it = torch.from_numpy(code2vocab_fun(code_sum.cpu().numpy(), self.code2vocab)).cuda().long()
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                it = it * unfinished.type_as(it)
                seq[:, t-1] = it
                if unfinished.sum() == 0:
                    break
        return seq, output_logit

    def sentence_completion(self, step, unfinished, it, state, seqs, sample_max):
        batch_size = seqs.size(0)
        seqs_cp = seqs.clone()
        unfinished_cp = unfinished.clone()
        n_cluster = len(self.cluster_size)
        for t in range(step + 1, self.seq_length + 1):
            if unfinished.sum() == 0:
                break
            xt = self.embed(it)
            output, state = self.core(xt, state)
            phi = self.logit(output)

            probs_step_1 = F.softmax(
                torch.cat([phi[:, :(n_cluster - 1)], torch.zeros(batch_size, 1).float().cuda()], 1), 1)
            if sample_max:
                it_1 = torch.max(probs_step_1.data, 1)[1].view(-1).cuda().long()
            else:
                it_1 = torch.multinomial(probs_step_1.data, 1).cuda().squeeze(1)
            it_2 = torch.zeros_like(it_1).cuda()
            start = n_cluster - 1
            for i in range(n_cluster):
                if self.cluster_size[i] != 1:
                    index = it_1 == i
                    if index.sum() != 0:
                        probs_step_2 = F.softmax(torch.cat([phi[index, start:(start + self.cluster_size[i] - 1)],
                                                            torch.zeros(index.sum(), 1).float().cuda()], 1), 1)
                        if sample_max:
                            it_2[index] = torch.max(probs_step_2.data, 1)[1].view(-1).cuda().long()
                        else:
                            it_2[index] = torch.multinomial(probs_step_2.data, 1).cuda().squeeze(1)
                start = start + self.cluster_size[i] - 1
            code_sum = it_1 * (self.vocab_size + 1) + it_2

            it = torch.from_numpy(code2vocab_fun(code_sum.cpu().numpy(), self.code2vocab)).cuda().long()
            unfinished_cp = unfinished_cp * (it > 0)
            it = it * unfinished_cp.type_as(it)
            seqs_cp[:, t - 1] = it
        return seqs_cp


    def get_arm_loss_two_layer_fast(self, fc_feats, att_feats, att_masks, opt, data, loader):
        sample_max = 1
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        seq = fc_feats.new_zeros(batch_size, self.seq_length, dtype=torch.long).cuda()
        unfinished = fc_feats.new_ones(batch_size, dtype=torch.uint8).cuda()
        loss = torch.zeros([]).float().cuda()
        mask_sum = 0
        n_cluster = len(self.cluster_size)
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
                # things to concat across depths:
                seqs_arm_list = [] # length 2
                state_arm_list = [] # length 2
                it_arm_list = []
                unfinished_arm_list = []
                pi_list = []
                phi_list = []

                arm_pseudo_action_set_list = []
                arm_index_list = []
                arm_index_2_list = []
                arm_pseudo_counts_list = []
                arm_pseudo_index_list = []
                counts_per_sample_list_list = []
                batch_index_list = []

                ### first depth:
                unfinished_size = unfinished.sum()
                pi_step_1 = np.random.uniform(size=[unfinished_size, n_cluster])
                phi_pad_step_1 = torch.cat([phi[unfinished, :(n_cluster-1)].clone(), torch.zeros(unfinished_size, 1).float().cuda()], 1)
                pseudo_action_step_1 = pseudo_action_batch(pi_step_1, phi_pad_step_1.data.cpu().numpy()) #batch, n_cluster, n_cluster
                pseudo_action_step_1 = np.reshape(pseudo_action_step_1, [unfinished_size, -1])
                ## concate unique pseudo actions
                arm_pseudo_action_set, arm_index, arm_index_2, arm_pseudo_counts, arm_pseudo_index, \
                counts_per_sample_list = unique_function(pseudo_action_step_1)
                ## complete words
                if np.sum(arm_pseudo_counts) !=0: #TODO: what if it ==0
                    arm_pseudo_action_set_list.append(arm_pseudo_action_set)
                    pi_list.append(pi_step_1)
                    phi_list.append(phi_pad_step_1)
                    arm_index_list.append(arm_index)
                    arm_index_2_list.append(arm_index_2)
                    arm_pseudo_counts_list.append(arm_pseudo_counts)
                    arm_pseudo_index_list.append(arm_pseudo_index)
                    counts_per_sample_list_list.append(counts_per_sample_list)
                    seqs_arm_step_1 = seq[unfinished, :][arm_index_2, :].clone()
                    unfinished_arm_step_1 = unfinished[unfinished][arm_index_2]
                    it_step_1 = torch.from_numpy(arm_pseudo_action_set).long().cuda()
                    phi_arm_step_1 = phi[unfinished, :][arm_index_2, :].clone()
                    start = n_cluster - 1
                    it_step_2 = torch.zeros_like(it_step_1).cuda()
                    for i in range(n_cluster):
                        index = it_step_1 == i
                        if index.sum() != 0:
                            probs_step_2 = F.softmax(torch.cat(
                                [phi_arm_step_1[index, start:(start+self.cluster_size[i]-1)],
                                 torch.zeros(index.sum(), 1).float().cuda()], 1), 1)
                            if sample_max:
                                it_step_2[index] = torch.max(probs_step_2.data, 1)[1].view(-1).cuda().long()
                            else:
                                it_step_2[index] = torch.multinomial(probs_step_2.data, 1).cuda().squeeze(1)
                        start = start + self.cluster_size[i] - 1
                    code_sum = it_step_1 * (self.vocab_size + 1) + it_step_2
                    it_step = torch.from_numpy(code2vocab_fun(code_sum.cpu().numpy(), self.code2vocab)).cuda().long()
                    seqs_arm_step_1[:, t-1] = it_step * unfinished_arm_step_1.type_as(it)
                    unfinished_arm_step_1 = unfinished_arm_step_1 * (it_step > 0)
                    state_h, state_c = state
                    state_h_arm_step_1 = state_h[:, unfinished, :][:, arm_index_2, :]
                    state_c_arm_step_1 = state_c[:, unfinished, :][:, arm_index_2, :]
                    state_arm_step_1 = (state_h_arm_step_1, state_c_arm_step_1)
                    seqs_arm_list.append(seqs_arm_step_1)
                    state_arm_list.append(state_arm_step_1)
                    it_arm_list.append(it_step)
                    unfinished_arm_list.append(unfinished_arm_step_1)
                    batch_index_list.append(torch.arange(batch_size)[unfinished])


                ### second depth:
                probs_step_1 = F.softmax(
                    torch.cat([phi[:, :(n_cluster - 1)], torch.zeros(batch_size, 1).float().cuda()], 1), 1)
                if sample_max:
                    it_1 = torch.max(probs_step_1.data, 1)[1].view(-1).cuda().long()
                else:
                    it_1 = torch.multinomial(probs_step_1.data, 1).cuda().squeeze(1)
                it_2 = torch.zeros_like(it_1).cuda()
                start = n_cluster - 1
                for i in range(n_cluster):
                    index = it_1[unfinished] == i
                    if index.sum() != 0:
                        # pseudo actions
                        effect_batch = index.sum()
                        cluster_size = self.cluster_size[i]
                        pi_step_2 = np.random.uniform(size=[effect_batch, cluster_size])
                        phi_pad_step_2 = torch.cat(
                            [phi[unfinished, :][index, start:(start+cluster_size - 1)].clone(), torch.zeros(effect_batch, 1).float().cuda()], 1)
                        pseudo_action_step_2 = pseudo_action_batch(pi_step_2,
                                                                   phi_pad_step_2.data.cpu().numpy())  # batch, n_cluster, n_cluster
                        pseudo_action_step_2 = np.reshape(pseudo_action_step_2, [effect_batch, -1])
                        arm_pseudo_action_set, arm_index, arm_index_2, arm_pseudo_counts, arm_pseudo_index, \
                        counts_per_sample_list = unique_function(pseudo_action_step_2)
                        arm_pseudo_action_set_list.append(arm_pseudo_action_set)
                        arm_index_list.append(arm_index)
                        arm_index_2_list.append(arm_index_2)
                        arm_pseudo_counts_list.append(arm_pseudo_counts)
                        arm_pseudo_index_list.append(arm_pseudo_index)
                        counts_per_sample_list_list.append(counts_per_sample_list)
                        pi_list.append(pi_step_2)
                        phi_list.append(phi_pad_step_2)

                        code_sum = it_1[unfinished][index][arm_index_2].clone() * (self.vocab_size + 1) + torch.from_numpy(np.array(arm_pseudo_action_set)).long().cuda()
                        it_step = torch.from_numpy(
                            code2vocab_fun(code_sum.cpu().numpy(), self.code2vocab)).cuda().long()

                        seqs_arm_step_2 = seq[unfinished, :][index, :][arm_index_2, :].clone()
                        unfinished_arm_step_2 = unfinished[unfinished][index][arm_index_2]


                        seqs_arm_step_2[:, t - 1] = it_step * unfinished_arm_step_2.type_as(it)
                        unfinished_arm_step_2 = unfinished_arm_step_2 * (it_step > 0)
                        state_h_arm_step_2 = state_h[:, unfinished, :][:, index, :][:, arm_index_2, :]
                        state_c_arm_step_2 = state_c[:, unfinished, :][:, index, :][:, arm_index_2, :]
                        state_arm_step_2 = (state_h_arm_step_2, state_c_arm_step_2)
                        seqs_arm_list.append(seqs_arm_step_2)
                        state_arm_list.append(state_arm_step_2)
                        it_arm_list.append(it_step)
                        unfinished_arm_list.append(unfinished_arm_step_2)
                        batch_index_list.append(torch.arange(batch_size)[unfinished][index])
                    start = start + self.cluster_size[i] - 1
                start = n_cluster - 1
                for i in range(n_cluster):
                    if self.cluster_size[i] != 1:
                        index = it_1 == i
                        if index.sum() != 0:
                            probs_step_2 = F.softmax(torch.cat([phi[index, start:(start + self.cluster_size[i] - 1)],
                                                                torch.zeros(index.sum(), 1).float().cuda()], 1), 1)
                            if sample_max:
                                it_2[index] = torch.max(probs_step_2.data, 1)[1].view(-1).cuda().long()
                            else:
                                it_2[index] = torch.multinomial(probs_step_2.data, 1).cuda().squeeze(1)
                    start = start + self.cluster_size[i] - 1
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
                    seqs_arm_completed = self.sentence_completion(t, unfinished_arm_straight, it_arm_straight,
                                                                  state_arm_straight, seqs_arm_straight, sample_max)
                    gts = OrderedDict()
                    for i in range(len(data['gts'])):
                        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]
                    start_index = 0
                    for i in range(len(unfinished_arm_list)):
                        arm_pseudo_index = np.array(arm_pseudo_index_list[i])  # TODO: only run non-1 pseudo
                        batch_index = np.array(batch_index_list[i])
                        effect_batch = np.sum(arm_pseudo_index[arm_pseudo_index > 1])
                        if effect_batch > 1:
                            arm_metric_value = reward_function(data, batch_size,
                                                               seqs_arm_completed[start_index:(start_index+effect_batch)],
                                                               batch_index[np.array(arm_index_2_list[i]).astype(int)],
                                                               gts)
                            start_index = start_index + effect_batch
                            arm_index = arm_index_list[i]
                            arm_pseudo_counts = arm_pseudo_counts_list[i]
                            vocab_size = phi_list[i].size(1)
                            arm_index += np.repeat(
                                np.expand_dims(np.concatenate([[0], np.cumsum(arm_pseudo_counts)[0:-1]]), 1),
                                vocab_size * vocab_size, 1)
                            arm_index = np.reshape(arm_index, [-1])
                            #print(i, batch_index[np.array(arm_index_2_list[i]).astype(int)])
                            arm_metric_matrix = np.reshape(arm_metric_value[arm_index], [-1, vocab_size, vocab_size])
                            arm_metric_matrix_cuda = torch.from_numpy(arm_metric_matrix).float().cuda()
                            f_delta = (arm_metric_matrix_cuda - arm_metric_matrix_cuda.mean(1).unsqueeze(1).repeat(1,vocab_size,1))
                            f_delta = (f_delta * (1 / vocab_size - torch.from_numpy(pi_list[i][arm_pseudo_index > 1]).float().cuda().unsqueeze(1).repeat(1,vocab_size,1))).sum(2)
                            f_delta = f_delta - f_delta[:, -1].unsqueeze(1).repeat(1, vocab_size) #TODO: verify formulation
                            loss = loss - (f_delta.detach() * phi_list[i][torch.from_numpy(arm_pseudo_index).cuda() > 1]).sum()
                            if np.random.randint(200) == 1:
                                print('step', t, 'vocab', i, 'average reward',
                                      np.mean(arm_metric_value), 'ave pseudo num', np.mean(arm_pseudo_index))
                    assert start_index == seqs_arm_completed.size(0)
                code_sum = it_1 * (self.vocab_size + 1) + it_2
                it = torch.from_numpy(code2vocab_fun(code_sum.cpu().numpy(), self.code2vocab)).cuda().long()
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                it = it * unfinished.type_as(it)
                seq[:, t - 1] = it
                mask_sum += unfinished.sum()
                if unfinished.sum() == 0:
                    break
        # reward = reward_function(data, batch_size, seq, torch.arange(batch_size), gts)
        # print('ave reward', np.mean(reward))
        return loss / mask_sum


def concatenate_arm(need_run_index, pre_seq, phi, unfinished, state, code_sum, mask_depth,
                    it_1, it_0):
    seqs_arm = torch.cat([pre_seq[need_run_index, :], pre_seq[need_run_index, :]], 0) #batch, length
    phi_arm = torch.cat([phi[need_run_index, :], phi[need_run_index, :]], 0) #batch,
    unfinished_arm = torch.cat([unfinished[need_run_index], unfinished[need_run_index]], 0) #batch,
    state_h, state_c = state
    state_h_arm = torch.cat([state_h[:, need_run_index, :], state_h[:, need_run_index, :]], 1)
    state_c_arm = torch.cat([state_c[:, need_run_index, :], state_c[:, need_run_index, :]], 1)
    state_arm = (state_h_arm, state_c_arm)
    code_sum_arm = np.concatenate([code_sum[need_run_index.cpu().numpy().astype(bool)], code_sum[need_run_index.cpu().numpy().astype(bool)]], 0)
    mask_depth_arm = torch.cat([mask_depth[need_run_index], mask_depth[need_run_index]], 0)
    it_depth_arm = torch.cat([it_1[need_run_index, :], it_0[need_run_index, :]], 0)
    return seqs_arm, phi_arm, unfinished_arm, state_arm, code_sum_arm, mask_depth_arm, it_depth_arm


def reward_function(data, batch_size, seqs, target_index, gts):
    seq_per_img = batch_size // len(data['gts'])
    seqs_size = seqs.size(0)

    seqs = seqs.data.cpu().numpy()
    res_ = []
    gts_new = {}
    for i in range(seqs_size):
        res_.append({'image_id':i, 'caption': [array_to_str(seqs[i])]})
        i_index = int(target_index[i])
        gts_new[i] = gts[i_index // seq_per_img]
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


def pseudo_action_swap_matrix(pi, phi):
    C = len(pi)
    RaceAllSwap = np.log(pi[:, np.newaxis]) - phi[np.newaxis, :]
    Race = np.diag(RaceAllSwap)
    action_true = np.argmin(Race)
    Race_min = Race[action_true]

    if C < 7:
        # Slow version for large C
        pseudo_actions = np.full((C, C), action_true)
        for m in range(C):
            for jj in range(m):
                RaceSwap = Race.copy()
                RaceSwap[m], RaceSwap[jj] = RaceAllSwap[jj, m], RaceAllSwap[m, jj]
                s_action = np.argmin(RaceSwap)
                pseudo_actions[m, jj], pseudo_actions[jj, m] = s_action, s_action
    else:
        # Fast version for large C
        pseudo_actions = np.full((C, C), action_true)

        SwapSuccess = RaceAllSwap <= Race_min
        SwapSuccess[action_true, :] = True
        np.fill_diagonal(SwapSuccess, 0)
        m_idx, j_idx = np.where(SwapSuccess)

        for i in range(len(m_idx)):
            m, jj = m_idx[i], j_idx[i]
            RaceSwap = Race.copy()
            RaceSwap[m], RaceSwap[jj] = RaceAllSwap[jj, m], RaceAllSwap[m, jj]
            if m == action_true or jj == action_true:
                s_action = np.argmin(RaceSwap)
                pseudo_actions[m, jj], pseudo_actions[jj, m] = s_action, s_action
            else:
                if RaceSwap[m] < RaceSwap[jj]:
                    pseudo_actions[m, jj], pseudo_actions[jj, m] = m, m
                else:
                    pseudo_actions[m, jj], pseudo_actions[jj, m] = jj, jj

    return pseudo_actions

def pseudo_action_batch(pi_batch, phi_batch):
    batch, vocab = np.shape(pi_batch)
    result = np.zeros(shape=[batch, vocab, vocab])
    for i in range(batch):
        result[i, :, :] = pseudo_action_swap_matrix(pi_batch[i, :], phi_batch[i, :])
    return result


def unique_function(pseudo_action):
    batch_size = pseudo_action.shape[0]
    arm_pseudo_action_set = []
    arm_index = []
    arm_index_2 = np.zeros(0)
    arm_pseudo_counts = []
    counts_per_sample_list = []
    arm_pseudo_index = []
    for i in range(batch_size):
        set_per_sample, index_per_sample, counts_per_sample = np.unique(pseudo_action[i, :],
                                                                        return_inverse=True,
                                                                        return_counts=True)
        pseudo_count = len(set_per_sample)
        arm_pseudo_index.append(pseudo_count)
        if pseudo_count > 1:
            arm_pseudo_counts.append(pseudo_count)
            arm_pseudo_action_set = np.concatenate([arm_pseudo_action_set, set_per_sample], axis=0)
            arm_index.append(index_per_sample)
            arm_index_2 = np.concatenate([arm_index_2, (np.ones(pseudo_count) * i)], axis=0)
            counts_per_sample_list.append(counts_per_sample)
    return arm_pseudo_action_set, arm_index, arm_index_2, arm_pseudo_counts, arm_pseudo_index, counts_per_sample_list
