from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from time import time
import misc.utils as utils
from collections import OrderedDict
import torch
import torch.nn.functional as F
import misc.utils as utils

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

def get_self_critical_reward(model, fc_feats, att_feats, att_masks, data, gen_result, opt):
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])

    # get greedy decoding baseline
    model.eval()
    with torch.no_grad():
        greedy_res, _ = model(fc_feats, att_feats, att_masks=att_masks, mode='sample')
    model.train()

    res = OrderedDict()

    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards


def get_reward(data, gen_result, opt, critic=False):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])

    res = OrderedDict()

    gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(batch_size)]
    res__ = {i: res[i] for i in range(batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(batch_size)}
    #print(gen_result[0])
    #print(gts[0])
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)
    if critic:
        return rewards, np.std(cider_scores)
    if opt.rf_demean == 1:
        rewards = np.repeat(scores[:, np.newaxis] - np.mean(scores[:, np.newaxis]), gen_result.shape[1], 1)

    return rewards


def get_mct_loss(model, fc_feats, att_feats, att_masks, opt, true_model, critic=None):
    batch_size = fc_feats.size(0)
    vocab_size = opt.vocab_size + 1
    state = model.init_hidden(batch_size)
    seq = fc_feats.new_zeros(batch_size, model.seq_length, dtype=torch.long)
    mct_baseline = fc_feats.new_zeros(batch_size, model.seq_length)
    unfinished = fc_feats.new_ones(batch_size, dtype=torch.uint8)
    temperature = getattr(opt, 'temperature', 1.0)
    seqLogprobs = fc_feats.new_zeros(batch_size, model.seq_length)
    seqprobs = fc_feats.new_zeros(batch_size, model.seq_length)
    true_length = 0
    for t in range(model.seq_length + 1):
        if t == 0:
            xt = model.img_embed(fc_feats)
        else:
            if t == 1:
                it = fc_feats.data.new(batch_size).long().zero_()
            xt = model.embed(it)

        output, state = model.core(xt, state)
        if t >= 1:
            logprobs = F.log_softmax(model.logit(output), dim=1)
            probs = F.softmax(model.logit(output), dim=1)
            mct_baseline[:, t-1] = torch.from_numpy(complete_batch_fun(logprobs, seq, t, model, state, unfinished, opt, true_model, fc_feats)).float().cuda()
            if opt.arm_step_sample == 'greedy':
                it = torch.max(logprobs.data, 1)[1].unsqueeze(1)
            else:
                if temperature == 1.0:
                    it = torch.multinomial(torch.exp(logprobs.data).cpu(), 1).cuda()
                else:
                    it = torch.multinomial(torch.exp(torch.div(logprobs.data, temperature)).cpu(), 1).cuda()
            sampleLogprobs = logprobs.gather(1, it)
            sampleprobs = probs.gather(1, it)
            it = it.view(-1).long()

            if t == 1:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)

            it = it * unfinished.type_as(it)
            seq[:, t-1] = it
            seqLogprobs[:, t - 1] = sampleLogprobs.view(-1)
            seqprobs[:, t - 1] = sampleprobs.view(-1)
            true_length += 1
            if unfinished.sum() == 0:
                break

    return seq, seqLogprobs, seqprobs.detach(), mct_baseline.detach()

def complete_batch_fun(logits, pre_seq, step, model, state, unfinished, opt, true_model, fc_feats):
    mct_sample_num = getattr(opt, 'mct_sample_num', 1)
    batch_size, _ = logits.size()
    rewards = np.zeros([batch_size, mct_sample_num])
    arm_metric_matrix = np.ones([batch_size, mct_sample_num]) * -1
    pseudo_actions = torch.multinomial(torch.exp(logits.data).cpu(), mct_sample_num, replacement=True).cuda()
    arm_pseudo_action_set = []
    arm_index = []
    arm_index_2 = np.zeros(0)
    arm_pseudo_counts = []
    counts_per_sample_list = []
    temperature = getattr(opt, 'temperature', 1.0)
    arm_pseudo_index = []
    for i in range(batch_size):
        set_per_sample, index_per_sample, counts_per_sample = np.unique(pseudo_actions[i, :].cpu().numpy(),
                                                                        return_inverse=True, return_counts=True)
        pseudo_count = len(set_per_sample)
        arm_pseudo_index.append(pseudo_count)
        if pseudo_count > 1:
            arm_pseudo_counts.append(pseudo_count)
            arm_pseudo_action_set = np.concatenate([arm_pseudo_action_set, set_per_sample], axis=0)
            arm_index.append(index_per_sample)
            arm_index_2 = np.concatenate([arm_index_2, (np.ones(pseudo_count) * i)], axis=0)
            counts_per_sample_list.append(counts_per_sample)
    if np.sum(arm_pseudo_counts) == 0:
        return np.mean(arm_metric_matrix, 1)
    seqs_arm = pre_seq[arm_index_2, :]
    unfinished_arm = unfinished[arm_index_2]
    it = torch.from_numpy(arm_pseudo_action_set).long().cuda()
    seqs_arm[:, step - 1] = it * unfinished_arm.type_as(it)
    unfinished_arm = (it > 0) * unfinished_arm
    state_h, state_c = state
    state_h_arm = state_h[:, arm_index_2, :]
    state_c_arm = state_c[:, arm_index_2, :]
    state_arm = (state_h_arm, state_c_arm)
    for t in range(step + 1, model.seq_length + 1):
        if unfinished_arm.sum() == 0:
            break
        xt = model.embed(it)
        output, state_arm = model.core(xt, state_arm)
        logprobs = F.log_softmax(model.logit(output), dim=1)
        if opt.arm_sample == 'greedy':
            it = torch.max(logprobs, 1)[1]
        else:
            if temperature == 1.0:
                prob_prev = torch.exp(logprobs.data).cpu()
            else:
                prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
            it = torch.multinomial(prob_prev, 1).cuda()
        it = it.view(-1).long()
        unfinished_arm = (it > 0) * unfinished_arm
        seqs_arm[:, t - 1] = it * unfinished_arm.type_as(it)
    # print('time for completion: ' + str(time() - tic))
    ## evaluate reward
    labels = torch.cat([torch.zeros(seqs_arm.size(0), 1).cuda().long(), seqs_arm], 1)
    masks = (labels > 0).float()
    logit_reward = true_model(fc_feats[arm_index_2], None, labels, None)
    arm_metric_value = (logit_reward.gather(2, labels[:, 1:].unsqueeze(2)).squeeze(2) * masks[:, 1:]).sum(1)
    arm_metric_value = arm_metric_value.detach().cpu().numpy()

    arm_index = np.array(arm_index)
    arm_index += np.repeat(np.expand_dims(np.concatenate([[0], np.cumsum(arm_pseudo_counts)[0:-1]]), 1),
                           mct_sample_num, 1)
    arm_index = np.reshape(arm_index, [-1])
    arm_pseudo_index = np.array(arm_pseudo_index)
    arm_metric_matrix[arm_pseudo_index > 1, :] = np.reshape(arm_metric_value[arm_index], [-1, mct_sample_num])
    return np.mean(arm_metric_matrix, 1)



def get_arm_loss(model, fc_feats, att_feats, att_masks, true_model, opt, type='arsm', critic=None):
    batch_size = fc_feats.size(0)
    vocab_size = opt.vocab_size + 1
    state = model.init_hidden(batch_size)
    seq = fc_feats.new_zeros(batch_size, model.seq_length, dtype=torch.long)
    arm_baseline = fc_feats.new_zeros(batch_size, model.seq_length)
    loss = fc_feats.new_zeros([])
    unfinished = fc_feats.new_ones(batch_size, dtype=torch.uint8)
    temperature = getattr(opt, 'temperature', 1.0)
    pseudo_action_list = fc_feats.new_ones(batch_size, model.seq_length, vocab_size, dtype=torch.long)
    seqLogprobs = fc_feats.new_zeros(batch_size, model.seq_length)
    seqprobs = fc_feats.new_zeros(batch_size, model.seq_length)
    mask_sum = 0
    true_length = 0
    pi_list = []
    logprobs_list = []
    for t in range(model.seq_length + 1):
        if t == 0:
            xt = model.img_embed(fc_feats)
        else:
            if t == 1:
                it = fc_feats.data.new(batch_size).long().zero_()
            xt = model.embed(it)

        output, state = model.core(xt, state)
        #print(opt.seq_per_img)
        if t >= 1:
            logits = model.logit(output)
            logprobs = F.log_softmax(logits, dim=1)
            probs = F.softmax(model.logit(output), dim=1)
            pi = torch.from_numpy(np.random.dirichlet(np.ones(vocab_size), batch_size)).float().cuda()
            mask = unfinished.float()
            # if opt.arm_as_baseline == 1:
            #     data = None
            #     loader = None
            #     if opt.critic_model != 'att_critic_vocab' or critic == None:
            #         # TODO: log space.
            #         arm_baseline[:, t-1] = arsm_f_delta_fun_batch_torch(logits.data, pi, data, seq, t, model, state, unfinished, loader,
            #                                          opt, critic)
            #     elif opt.critic_model == 'att_critic_vocab' and critic is not None:
            #         pseudo_action, pi_R = arsm_f_delta_fun_batch_torch(logits.data, pi, data, seq, t, model, state,
            #                                                            unfinished, loader,
            #                                                            opt, critic)
            #         pseudo_action_list[:, t - 1, :] = pseudo_action
            #         pi_list.append(pi_R)

            if opt.critic_model != 'att_critic_vocab' or critic == None:
                f_delta = arsm_f_delta_fun_batch_torch(logits.data, pi, true_model, seq, t, model, state, unfinished,
                                                       fc_feats, att_feats, att_masks, opt, critic, type)
                f_delta = f_delta / temperature
                f_delta = (f_delta.transpose(0, 1) * mask).transpose(0, 1)
                mask_sum += torch.sum(mask)
                loss -= torch.sum(f_delta.detach() * model.logit(output))
            if opt.arm_step_sample == 'greedy':
                it = torch.max(logprobs.data, 1)[1].unsqueeze(1)
            else:
                if temperature == 1.0:
                    it = torch.multinomial(torch.exp(logprobs.data).cpu(), 1).cuda()
                    # it = torch.min(torch.log(pi) - logprobs_demin, 1)[1].unsqueeze(1)
                else:
                    it = torch.multinomial(torch.exp(torch.div(logprobs.data, temperature)).cpu(), 1).cuda()
                    # it = torch.min(torch.log(pi) - logprobs_demin / temperature, 1)[1].unsqueeze(1)
            sampleLogprobs = logprobs.gather(1, it)
            sampleprobs = probs.gather(1, it)
            it = it.view(-1).long()

            if t == 1:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)

            it = it * unfinished.type_as(it)
            seq[:, t-1] = it
            seqLogprobs[:, t - 1] = sampleLogprobs.view(-1)
            seqprobs[:, t - 1] = sampleprobs.view(-1)
            true_length += 1
            if unfinished.sum() == 0:
                break
    if opt.arm_as_baseline == 1 and (opt.critic_model != 'att_critic_vocab' or critic is None):
        return seq, seqLogprobs, seqprobs.detach(), arm_baseline.detach()
    elif opt.arm_as_baseline == 1 and opt.critic_model == 'att_critic_vocab' and critic is not None:
        seq_pad = torch.cat([seq.new_zeros(seq.size(0), 1, dtype=torch.long), seq], 1)
        critic_value = critic(seq_pad, fc_feats, att_feats, True, opt, att_masks).detach()
        for t in range(true_length):
            arm_baseline[:, t] = critic_value[:, t, :].gather(1, pseudo_action_list[:, t, :]).mean(1)
        return seq, seqLogprobs, seqprobs.detach(), arm_baseline.detach()
    if opt.critic_model == 'att_critic_vocab' and critic is not None:
        loss = fc_feats.new_zeros([])
        seq_pad = torch.cat([seq.new_zeros(seq.size(0), 1, dtype=torch.long), seq], 1)
        critic_value = critic(seq_pad, fc_feats, att_feats, True, opt, att_masks).detach()
        mask = fc_feats.new_ones(batch_size, dtype=torch.uint8)
        for t in range(true_length):
            f_delta = critic_value[:, t, :].gather(1, pseudo_action_list[:, t, :])
            f_delta = f_delta - torch.mean(f_delta, 1).unsqueeze(1).repeat(1, vocab_size)
            f_delta = f_delta * (1.0 - vocab_size * pi_list[t]).float().unsqueeze(1).repeat(1, vocab_size)
            if t > 0:
                mask *= seq_pad[:, t] > 0
            f_delta = (f_delta.transpose(0, 1) * mask.float()).transpose(0, 1)
            mask_sum += torch.sum(mask.float())
            loss -= torch.sum(f_delta.detach() * logprobs_list[t])
    loss = loss / mask_sum
    return loss



def get_ar_loss(model, fc_feats, att_feats, att_masks, true_model, opt, critic=None):
    batch_size = fc_feats.size(0)
    vocab_size = opt.vocab_size + 1
    state = model.init_hidden(batch_size)
    seq = fc_feats.new_zeros(batch_size, model.seq_length, dtype=torch.long)
    arm_baseline = fc_feats.new_zeros(batch_size, model.seq_length)
    loss = fc_feats.new_zeros([])
    unfinished = fc_feats.new_ones(batch_size, dtype=torch.uint8)
    temperature = getattr(opt, 'temperature', 1.0)
    seqLogprobs = fc_feats.new_zeros(batch_size, model.seq_length)
    seqprobs = fc_feats.new_zeros(batch_size, model.seq_length)
    mask_sum = 0
    true_length = 0
    pi_list = []
    logprobs_list = []
    for t in range(model.seq_length + 1):
        if t == 0:
            xt = model.img_embed(fc_feats)
        else:
            if t == 1:
                it = fc_feats.data.new(batch_size).long().zero_()
            xt = model.embed(it)

        output, state = model.core(xt, state)
        # print(opt.seq_per_img)
        if t >= 1:
            logprobs = F.log_softmax(model.logit(output), dim=1)
            probs = F.softmax(model.logit(output), dim=1)
            pi = torch.from_numpy(np.random.dirichlet(np.ones(vocab_size), batch_size)).float().cuda()
            mask = unfinished.float()
            if temperature == 1.0:
                it = torch.min(torch.log(pi) - logprobs.data, 1)[1].unsqueeze(1)
            else:
                it = torch.min(torch.log(pi) - logprobs.data / temperature, 1)[1].unsqueeze(1)
            sampleLogprobs = logprobs.gather(1, it)
            sampleprobs = probs.gather(1, it)
            it = it.view(-1).long()
            pi_list.append(pi)
            logprobs_list.append(model.logit(output))
            if t == 1:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)

            it = it * unfinished.type_as(it)
            seq[:, t - 1] = it
            seqLogprobs[:, t - 1] = sampleLogprobs.view(-1)
            seqprobs[:, t - 1] = sampleprobs.view(-1)
            true_length += 1
            if unfinished.sum() == 0:
                break
    loss = fc_feats.new_zeros([])

    ## evaluate reward
    labels = torch.cat([torch.zeros(seq.size(0), 1).cuda().long(), seq], 1)
    masks = (labels > 0).float()
    logit_reward = true_model(fc_feats, None, labels, None)
    arm_metric_value = (logit_reward.gather(2, labels[:, 1:].unsqueeze(2)).squeeze(2) * masks[:, 1:]).sum(1)
    arm_metric_value = arm_metric_value.detach().cpu().numpy()


    mask = fc_feats.new_ones(batch_size, dtype=torch.uint8)
    for t in range(true_length):
        f_delta = torch.from_numpy(np.repeat(np.expand_dims(arm_metric_value, 1), vocab_size, 1)).float().cuda() * (
                    1.0 - vocab_size * pi_list[t])
        if t > 0:
            mask *= seq[:, t - 1] > 0
        f_delta = (f_delta.transpose(0, 1) * mask.float()).transpose(0, 1)
        mask_sum += torch.sum(mask.float())
        loss -= torch.sum(f_delta.detach() * logprobs_list[t])
    loss = loss / mask_sum
    return loss


def get_rf_loss(model, fc_feats, att_feats, att_masks, data, opt, loader, critic=None, test_critic=False):
    batch_size = fc_feats.size(0)
    vocab_size = opt.vocab_size + 1
    state = model.init_hidden(batch_size)
    seq = fc_feats.new_zeros(batch_size, model.seq_length, dtype=torch.long)
    unfinished = fc_feats.new_ones(batch_size, dtype=torch.uint8)
    temperature = getattr(opt, 'temperature', 1.0)
    seqLogprobs = fc_feats.new_zeros(batch_size, model.seq_length)
    seqLogprobs_total = fc_feats.new_zeros(batch_size, model.seq_length, vocab_size)
    seqprobs = fc_feats.new_zeros(batch_size, model.seq_length)
    mask_sum = 0
    true_length = 0
    pi_list = []
    logprobs_list = []
    probs_list = []
    q_list = []
    for t in range(model.seq_length + 1):
        if t == 0:
            xt = model.img_embed(fc_feats)
        else:
            if t == 1:
                it = fc_feats.data.new(batch_size).long().zero_()
            xt = model.embed(it)

        output, state = model.core(xt, state)
        if t >= 1:
            logprobs1 = model.logit(output)
            logprobs = F.log_softmax(model.logit(output), dim=1)
            probs = F.softmax(model.logit(output), dim=1)
            pi = torch.from_numpy(np.random.dirichlet(np.ones(vocab_size), batch_size)).float().cuda()
            mask = unfinished.float()
            if opt.importance_sampling == 1:
                if critic == None:
                    f = fc_feats.new_ones(batch_size, vocab_size)
                else:
                    if t == 1:
                        seq_pad = seq.new_zeros(seq.size(0), 1, dtype=torch.long)
                    else:
                        seq_pad = torch.cat([seq.new_zeros(seq.size(0), 1, dtype=torch.long), seq[:, :t-1]], 1)
                    critic_value = critic(seq_pad, xt, att_feats, False, opt, att_masks).detach()
                    f = critic_value[:, t-1, :]
            else:
                logprobs_sample = logprobs
            if temperature == 1.0:
                it = torch.min(torch.log(pi) - logprobs_sample, 1)[1].unsqueeze(1)
            else:
                it = torch.min(torch.log(pi) - logprobs_sample / temperature, 1)[1].unsqueeze(1)
            if test_critic:
                it = torch.min(- logprobs_sample, 1)[1].unsqueeze(1)
            sampleLogprobs = logprobs.gather(1, it)
            sampleprobs = probs.gather(1, it)
            seqLogprobs_total[:, t - 1, :] = logprobs
            it = it.view(-1).long()
            pi_list.append(pi)
            logprobs_list.append(logprobs1)
            probs_list.append(probs)
            if t == 1:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)

            it = it * unfinished.type_as(it)
            seq[:, t-1] = it
            seqLogprobs[:, t - 1] = sampleLogprobs.view(-1)
            seqprobs[:, t - 1] = sampleprobs.view(-1)
            true_length += 1
            if unfinished.sum() == 0:
                break

    if test_critic:
        print('imageid', data['infos'][0]['id'],data['infos'][1]['id'],data['infos'][2]['id'],data['infos'][3]['id'])
        sents = utils.decode_sequence(loader.get_vocab(), seq[0:1,:])
        print(sents)
        sents = utils.decode_sequence(loader.get_vocab(), seq[20:21,:])
        print(sents)
    loss = fc_feats.new_zeros([])
    seq_per_img = batch_size // len(data['gts'])
    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]
    seqs = seq.data.cpu().numpy()
    res_ = []
    gts_arm = {}
    for i in range(batch_size):
        res_.append({'image_id': i, 'caption': [array_to_str(seqs[i])]})
        gts_arm[i] = gts[i // seq_per_img]
    _, arm_metric_value = CiderD_scorer.compute_score(gts_arm, res_)
    mask = fc_feats.new_ones(batch_size, dtype=torch.uint8)
    for t in range(true_length):
        indicator = torch.arange(vocab_size).unsqueeze(0).repeat(batch_size, 1).cuda().long() == (seq[:,t]).unsqueeze(1).repeat(1, vocab_size)
        f_delta = torch.from_numpy(np.repeat(np.expand_dims(arm_metric_value, 1), vocab_size, 1)).float().cuda() * (
                    indicator.float() - probs_list[t])
        if opt.importance_sampling == 1:
            f_delta = f_delta * (torch.div(probs_list[t].gather(1, seq[:, t].unsqueeze(1)), epsilon + q_list[t].gather(1, seq[:, t].unsqueeze(1))).detach().repeat(1, vocab_size))
        if t > 0:
            mask *= seq[:, t-1] > 0
        f_delta = (f_delta.transpose(0, 1) * mask.float()).transpose(0, 1)
        mask_sum += torch.sum(mask.float())
        loss -= torch.sum(f_delta.detach() * logprobs_list[t])
    loss = loss / mask_sum
    return loss, seq, arm_metric_value, seqLogprobs_total


def arsm_f_delta_fun_batch_torch(logits, pi, true_model, pre_seq, step, model, state, unfinished, fc_feats, att_feats, att_masks, opt, critic=None, type='arsm', print_pseudo=True):
    #TODO: write in torch
    batch_size, vocab_size = logits.size()
    ref_num = opt.ref_num
    index_batch = torch.arange(batch_size).cuda().long()
    A_cat = torch.min(torch.log(pi) - logits, 1)[1].long()
    if type == 'arsm':
        ref_num = vocab_size
        tic = time()
        pi_numpy = pi.cpu().numpy()
        logits_numpy = logits.data.cpu().numpy()
        pseudo_actions = pseudo_action_batch(pi_numpy, logits_numpy)
    else:
        ref_num = opt.ref_num
        random_ref = torch.topk(torch.rand((batch_size, vocab_size)), ref_num)[1]
        R_cat_set = random_ref.cuda().long() # batch * ref_num
        R_cat_set_tmp = R_cat_set.unsqueeze(1).repeat(1, vocab_size, 1)
        pseudo_actions = pseudo_action_batch(pi.cpu().numpy(), logits.data.cpu().numpy())
        pseudo_actions = torch.from_numpy(pseudo_actions).cuda()
        pseudo_actions = pseudo_actions.gather(2, R_cat_set_tmp).cpu().numpy()

    arm_metric_matrix = np.ones([batch_size, vocab_size, ref_num]) * -1
    temperature = getattr(opt, 'temperature', 1.0)
    A_cat = torch.min(torch.log(pi) - logits, 1)[1].long()
    pseudo_actions = torch.from_numpy(np.reshape(pseudo_actions, [-1, vocab_size * ref_num])).cuda().long()
    if unfinished.sum(0) != batch_size:
        pseudo_actions[(1 - unfinished), :] = A_cat[(1 - unfinished)].unsqueeze(1).repeat(1, vocab_size*ref_num)
    #print('time for pseudo action: ' + str(time() - tic))
    tic = time()
    ## concate unique pseudo actions
    arm_pseudo_action_set = []
    arm_index = []
    arm_index_2 = np.zeros(0)
    arm_pseudo_counts = []
    counts_per_sample_list = []
    arm_pseudo_index = []
    for i in range(batch_size):
        set_per_sample, index_per_sample, counts_per_sample = np.unique(pseudo_actions[i, :].cpu().numpy(), return_inverse=True, return_counts=True)
        pseudo_count = len(set_per_sample)
        arm_pseudo_index.append(pseudo_count)
        if pseudo_count > 1:
            arm_pseudo_counts.append(pseudo_count)
            arm_pseudo_action_set = np.concatenate([arm_pseudo_action_set, set_per_sample], axis=0)
            arm_index.append(index_per_sample)
            arm_index_2 = np.concatenate([arm_index_2, (np.ones(pseudo_count) * i)], axis=0)
            counts_per_sample_list.append(counts_per_sample)
    ## complete sentences
    tic= time()
    if np.sum(arm_pseudo_counts) == 0:
        if opt.arm_as_baseline == 1:
            return torch.from_numpy(np.ones([batch_size]) * -1).float().cuda()
        else:
            return torch.from_numpy(np.zeros([batch_size, vocab_size])).float().cuda()
    if opt.rl_type == 'ars_indicator':
        return torch.from_numpy((np.array(arm_pseudo_index) != 1).astype(int)).float().cuda()
    seqs_arm = pre_seq[arm_index_2, :]
    unfinished_arm = unfinished[arm_index_2]
    it = torch.from_numpy(arm_pseudo_action_set).long().cuda()
    seqs_arm[:, step-1] = it * unfinished_arm.type_as(it)
    unfinished_arm = (it > 0) * unfinished_arm
    state_h, state_c = state
    state_h_arm = state_h[:, arm_index_2, :]
    state_c_arm = state_c[:, arm_index_2, :]
    state_arm = (state_h_arm, state_c_arm)
    if critic == None:
        for t in range(step + 1, model.seq_length + 1):
            if unfinished_arm.sum() == 0:
                break
            xt = model.embed(it)
            output, state_arm = model.core(xt, state_arm)
            logprobs = F.log_softmax(model.logit(output), dim=1)
            if opt.arm_sample == 'greedy':
                it = torch.max(logprobs, 1)[1]
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu()
                else:
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
            it = it.view(-1).long()
            unfinished_arm = (it > 0) * unfinished_arm
            seqs_arm[:, t-1] = it * unfinished_arm.type_as(it)
        ## evaluate reward
        labels = torch.cat([torch.zeros(seqs_arm.size(0), 1).cuda().long(), seqs_arm], 1)
        masks = (labels > 0).float()
        #print(np.shape(arm_index_2))
        logit_reward = true_model(fc_feats[arm_index_2], None, labels, None)
        a_dim = logit_reward.size(1)
        arm_metric_value = (logit_reward.gather(2, labels[:, 1:(a_dim+1)].unsqueeze(2)).squeeze(2) * masks[:, 1:(a_dim+1)]).sum(1)
        arm_metric_value = arm_metric_value.detach().cpu().numpy()
    arm_index = np.array(arm_index)
    arm_index += np.repeat(np.expand_dims(np.concatenate([[0], np.cumsum(arm_pseudo_counts)[0:-1]]), 1), vocab_size*ref_num, 1)
    arm_index = np.reshape(arm_index, [-1])
    arm_pseudo_index = np.array(arm_pseudo_index)
    arm_metric_matrix[arm_pseudo_index > 1, :, :] = np.reshape(arm_metric_value[arm_index], [-1, vocab_size, ref_num])
    if opt.arm_as_baseline == 1:
        return torch.from_numpy(arm_metric_matrix).float().cuda().mean(1)
    #print('time for evaluating pseudo action', time()-tic)
    tic = time()
    arm_metric_matrix_cuda = torch.from_numpy(arm_metric_matrix).float().cuda()
    if type == 'arsm':
        f_delta = ((arm_metric_matrix_cuda - arm_metric_matrix_cuda.mean(1).unsqueeze(1).repeat(1, vocab_size, 1))
                   * (1.0/vocab_size - pi.unsqueeze(1).repeat(1, vocab_size, 1))).sum(2)
    else:
        f_delta = torch.zeros(batch_size, vocab_size).float().cuda()
        for i in range(ref_num):
            R_cat = R_cat_set[:, i]
            arm_metric_matrix_tmp = arm_metric_matrix_cuda[:, :, i]
            f_delta_tmp = (arm_metric_matrix_tmp.transpose(0, 1) - arm_metric_matrix_tmp.mean(1)).transpose(0, 1)
            f_delta_tmp = (f_delta_tmp.transpose(0, 1) * (1.0 - vocab_size * pi[index_batch, R_cat])).transpose(0, 1)
            f_delta += f_delta_tmp
        f_delta = f_delta / ref_num
    return f_delta

def pseudo_action_fun(logits, A_cat, R_cat, pi, temperature=1):
    #TODO: log pi.
    batch_size, vocab_size = logits.size()
    index_batch = torch.arange(batch_size).cuda().long()
    index_vocab = torch.arange(vocab_size).cuda().long()
    min_value = torch.min(torch.log(pi) - logits, 1)[0].unsqueeze(1).repeat(1, vocab_size)
    pseudo_actions = A_cat.unsqueeze(1).repeat(1, vocab_size)
    pseudo_actions += ((-logits + torch.log(pi[index_batch, R_cat]).unsqueeze(1).repeat(1, vocab_size)) < min_value).long() * \
                      (index_vocab - A_cat.unsqueeze(1))
    pseudo_actions += ((torch.log(pi) - logits[index_batch, R_cat].unsqueeze(1).repeat(1, vocab_size)) < min_value).long() * \
                      (R_cat - A_cat).unsqueeze(1).repeat(1, vocab_size)
    index_matrix = torch.zeros_like(logits).long()
    index_matrix[index_batch, A_cat] = 1
    index_matrix[R_cat == A_cat, :] = 1

    topk, indices = torch.topk(-(torch.log(pi) - logits), 2, dim=1)
    top_2_indices = indices[:, 1]
    top_2_values = -topk[:, 1].unsqueeze(1).repeat(1, vocab_size)
    candidate_i_value = -logits + torch.log(pi[index_batch, R_cat]).unsqueeze(1).repeat(1, vocab_size)
    candidate_A_value = torch.log(pi) - logits[index_batch, R_cat].unsqueeze(1).repeat(1, vocab_size)
    pseudo_actions_true = top_2_indices.unsqueeze(1).repeat(1, vocab_size)
    pseudo_actions_true += (candidate_i_value < top_2_values).long() * (candidate_i_value <= candidate_A_value).long() * \
                           (index_vocab - top_2_indices.unsqueeze(1))
    pseudo_actions_true += (candidate_A_value < top_2_values).long() * (candidate_A_value < candidate_i_value).long() * \
                           (R_cat - top_2_indices).unsqueeze(1).repeat(1, vocab_size)

    pseudo_actions = pseudo_actions + index_matrix * (pseudo_actions_true - pseudo_actions)
    return pseudo_actions

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

def reward_fun(labels, fc_feats, true_model):
    labels_pad = torch.cat([torch.zeros(labels.size(0), 1).cuda().long(), labels], 1)
    masks = (labels_pad > 0).float()
    logit_reward = true_model(fc_feats, None, labels_pad, None)
    reward = (logit_reward.gather(2, labels_pad[:, 1:].unsqueeze(2)).squeeze(2) * masks[:, 1:]).sum(1)
    return reward