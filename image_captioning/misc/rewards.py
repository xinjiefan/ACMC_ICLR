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
import math

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
        #print('Cider scores:', _)
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

def get_mct_loss(model, fc_feats, att_feats, att_masks, data, opt, loader, critic=None):
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
            logits = model.logit(output)
            probs = F.softmax(model.logit(output), dim=1)
            mct_baseline[:, t-1] = torch.from_numpy(complete_batch_fun(logprobs, data, seq, t, model, state, unfinished, loader,
                                                      opt, critic)).float().cuda()
            if opt.arm_step_sample == 'greedy':
                it = torch.max(logprobs.data, 1)[1].unsqueeze(1)
            else:
                # if temperature == 1.0:
                #     it = torch.multinomial(torch.exp(logprobs.data).cpu(), 1).cuda()
                # else:
                #     it = torch.multinomial(torch.exp(torch.div(logprobs.data, temperature)).cpu(), 1).cuda()
                pi = torch.from_numpy(np.random.uniform(0, 1, [logits.shape[0], model.vocab_size + 1])).float().cuda()
                pi = -torch.log(pi)
                it = torch.min(torch.log(pi) - logits, 1)[1].unsqueeze(1)
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

def complete_batch_fun(logits, data, pre_seq, step, model, state, unfinished, loader, opt, critic):
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
        clogits = model.logit(output)
        if opt.arm_sample == 'greedy':
            it = torch.max(logprobs, 1)[1]
        else:
            # if temperature == 1.0:
            #     prob_prev = torch.exp(logprobs.data).cpu()
            # else:
            #     prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
            # it = torch.multinomial(prob_prev, 1).cuda()
            pi = torch.from_numpy(np.random.uniform(0, 1, [clogits.shape[0], model.vocab_size + 1])).float().cuda()
            pi = -torch.log(pi)
            it = torch.min(torch.log(pi) - clogits, 1)[1].unsqueeze(1)
        it = it.view(-1).long()
        unfinished_arm = (it > 0) * unfinished_arm
        seqs_arm[:, t - 1] = it * unfinished_arm.type_as(it)
    # print('time for completion: ' + str(time() - tic))
    ## evaluate reward
    tic = time()
    seq_per_img = batch_size // len(data['gts'])
    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]
    seqs_arm = seqs_arm.data.cpu().numpy()
    if step == np.random.randint(20) and np.random.randint(20) == 1:
        sents = utils.decode_sequence(loader.get_vocab(), torch.from_numpy(seqs_arm[0:arm_pseudo_counts[0]]).cuda())
        print('imageid', data['infos'][0]['id'], '**********************At step ' + str(step))
        print('True sentence:')
        print(sents[np.argmax(counts_per_sample_list[0])])
        print('Pseudo sentences: ')
        print(sents)
        print('Pseudo action mean: ', np.mean(arm_pseudo_index), 'std: ', np.std(arm_pseudo_index), 'max: ',
              np.max(arm_pseudo_index))
    res_ = []
    gts_arm = {}
    for i in range(len(arm_pseudo_action_set)):
        res_.append({'image_id': i, 'caption': [array_to_str(seqs_arm[i])]})
        i_index = arm_index_2[i]
        gts_arm[i] = gts[i_index // seq_per_img]
    tic = time()
    _, arm_metric_value = CiderD_scorer.compute_score(gts_arm, res_)
    arm_index = np.array(arm_index)
    arm_index += np.repeat(np.expand_dims(np.concatenate([[0], np.cumsum(arm_pseudo_counts)[0:-1]]), 1),
                           mct_sample_num, 1)
    arm_index = np.reshape(arm_index, [-1])
    arm_pseudo_index = np.array(arm_pseudo_index)
    arm_metric_matrix[arm_pseudo_index > 1, :] = np.reshape(arm_metric_value[arm_index], [-1, mct_sample_num])
    return np.mean(arm_metric_matrix, 1)

def entropy(p, dim = -1):
   return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim = dim)

def get_arm_loss_daniel(model, fc_feats, att_feats, att_masks, data, opt, loader, critic=None):
    # run the true trajectory
    data_gts = data['gts']
    model.train()
    seq, seqlogits, eff_length, states = model(fc_feats, att_feats, att_masks, mode='arm_sample')
    entropy_batch = entropy(F.softmax(seqlogits, dim=-1)).mean(0)
    seq_length, batch_size, vocab_size = seqlogits.size()

    pi = torch.from_numpy(np.random.dirichlet(np.ones(vocab_size), batch_size * seq_length)) \
        .reshape(seq_length * batch_size, vocab_size).float()

    # get the ground truth for computing cider score
    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    seq_per_img = batch_size // len(data_gts)
    gts_ = {i: gts[i // seq_per_img] for i in range(batch_size)}
    seq_val = seq.cpu().data.numpy()
    res_ = [{'image_id': i, 'caption': [array_to_str(seq_val[i])]} for i in range(batch_size)]
    _, rewards_main = CiderD_scorer.compute_score(gts_, res_)
    num_states = min(seq_length + 1, len(states))

    num_ref = opt.ref_num
    ref_cat = np.random.choice(range(vocab_size), num_ref, replace=False)
    pseudo_num_length = fc_feats.new_ones(seq_length).cuda()
    pseudo_num_batch = torch.zeros((seq_length, batch_size))
    with torch.no_grad():
        # Computing the pseudo_action
        phi_c = seqlogits.reshape(seq_length * batch_size, vocab_size)
        pi_c = pi.cuda()

        R_cat = torch.from_numpy(ref_cat).long().cuda()

        t0 = time()
        pseudo_action = pseudo_action_multiple_j(phi_c, R_cat, pi_c).reshape(num_ref, seq_length, batch_size,
                                                                             vocab_size)
        t1 = time()

        pseudo_action = pseudo_action.cpu()
        seq = seq.cpu()
        pi = pi.reshape(seq_length, batch_size, vocab_size)

        torch.cuda.empty_cache()

        gars = torch.zeros((num_ref, seq_length, batch_size, vocab_size)).float()

        timer = [0, 0, 0]
        sample_times = 0
        for t in range(1, num_states):
            t2 = time()
            p_a_t = pseudo_action[:, t - 1, :, :]

            finished_b_id = torch.tensor([], dtype=torch.uint8)
            u_p_a_t = torch.tensor([]).long()
            b_id_index = torch.tensor([]).long()
            b_id_length = []
            valid_b = []

            for b in range(batch_size):

                u_p_a_t_b = torch.unique(p_a_t[:, b, :].cuda()).cpu()
                u_p_a_t_b_length = u_p_a_t_b.size(0)
                pseudo_num_batch[t-1, b] = u_p_a_t_b_length
                if u_p_a_t_b_length == 1:
                    continue
                valid_b.append(b)
                u_p_a_t = torch.cat((u_p_a_t, u_p_a_t_b))
                b_id_index = torch.cat((b_id_index, torch.ones(u_p_a_t_b_length).long() * b))
                b_id_length.append(u_p_a_t_b_length)
                # inverse_index.append(u_p_a_t_b_inverse_id)

                if t>=2 and int(seq[b, t - 2]) == 0:
                    finished_b_id = torch.cat((finished_b_id, finished_b_id.new_ones(u_p_a_t_b_length)))
                else:
                    finished_b_id = torch.cat((finished_b_id, finished_b_id.new_zeros(u_p_a_t_b_length)))

            u_p_a_t_length = u_p_a_t.size(0)
            if u_p_a_t_length == 0:
                continue
            t2_2 = time()
            # obtain the its, states and seqs according to unique pseudo_actions
            state_h, state_c = states[t]
            state_h_arm = state_h[:, b_id_index, :]
            state_c_arm = state_c[:, b_id_index, :]
            # torch.cuda.empty_cache()
            p_state_t = (state_h_arm.cuda(), state_c_arm.cuda())

            p_seq_t = seq.new_zeros((u_p_a_t_length, seq_length))
            if t == 1:
                p_seq_t[:, 0] = u_p_a_t
            else:
                p_seq_t[:, :t - 1] = seq[b_id_index, :t - 1]
                p_seq_t[:, t - 1] = u_p_a_t

            p_it_t = u_p_a_t.cuda()

            t2_5 = time()
            timer[0] += t2_2 - t2

            sampled_seq_t = model(t, p_it_t, p_state_t, p_seq_t, mode='arm_pseudo_sample')
            # torch.cuda.empty_cache()
            t3 = time()
            timer[1] += t3 - t2_5
            sample_times += b_id_index.size(0)
            pseudo_num_length[t-1] = b_id_index.size(0) * 1.0 / batch_size
            if finished_b_id.sum() > 0:
                sampled_seq_t[finished_b_id] = seq[b_id_index[finished_b_id]]

            gts_t = {i: gts[int(b_id_index[i]) // seq_per_img] for i in range(b_id_index.size(0))}

            sampled_seq_t = sampled_seq_t.data.numpy()
            res_t = [{'image_id': i, 'caption': [array_to_str(sampled_seq_t[i])]} for i in range(len(sampled_seq_t))]
            t3_2 = time()

            _, rewards_t = CiderD_scorer.compute_score(gts_t, res_t)
            t3_5 = time()
            rewards_t = torch.from_numpy(rewards_t).float().split(b_id_length)
            unique_actions = u_p_a_t.split(b_id_length)

            for i in range(len(valid_b)):
                reward = torch.zeros(vocab_size)
                reward[unique_actions[i]] = rewards_t[i]
                b = valid_b[i]
                reward_t_b = reward[p_a_t[:, b, :]]
                gars[:, t - 1, b, :] = (reward_t_b - reward_t_b.mean(dim=1).unsqueeze(1)) * (
                            1 - pi[t - 1, b, ref_cat] * vocab_size).unsqueeze(1)

            t4 = time()
            timer[2] += t3_5 - t3_2
        #print("Time for 'take unique', 'compute pseudo_action', 'compute reward': " + str(timer))
        #print("# of unique pseudo_action for all ref: ", sample_times/(num_states*batch_size))

    # compute_score(seq, gts, batch_size, seq_per_img)
    gars = gars.cuda()
    gars = gars.mean(dim=0)
    loss = -(gars[:num_states - 1] * seqlogits[:num_states - 1]).sum() / eff_length
    #print(pseudo_num_length)
    #print(rewards_main)
    return loss, sample_times/(num_states*batch_size), pseudo_num_length, pseudo_num_batch, rewards_main, entropy_batch


def pseudo_action_multiple_j(logits, R_cat, pi, temperature=1):
    num_ref = R_cat.size(0)
    maximum_num = 20
    if num_ref <= maximum_num:
        return pseudo_action_fun_cuda(logits, R_cat, pi)
    else:
        times = math.ceil(num_ref / maximum_num)
        for i in range(int(times)):
            start = i * maximum_num
            end = min(num_ref, (i + 1) * maximum_num)
            partial_ref = R_cat[start: end]

            if i == 0:
                res = pseudo_action_fun_cuda(logits, partial_ref, pi)
            else:
                res = torch.cat((res, pseudo_action_fun_cuda(logits, partial_ref, pi)))
        return res


def pseudo_action_fun_cuda(logits, R_cat, pi, temperature=1):
    # compute the top two values and their indices
    batch_size, vocab_size = logits.size()
    num_ref = R_cat.size(0)

    top2, indices = torch.topk(-(torch.log(pi) - logits), 2, dim=1)
    top2 = -top2
    min_value = top2[:, 0].unsqueeze(1).expand(num_ref, -1, -1)
    sec_min_value = top2[:, 1].unsqueeze(1).expand(num_ref, -1, -1)
    A_cat = indices[:, 0]
    sec_indices = indices[:, 1]

    pseudo_actions = A_cat.unsqueeze(1).repeat(
        num_ref, vocab_size).reshape(num_ref, -1, vocab_size)
    pseudo_actions_true_move = sec_indices.unsqueeze(1).repeat(
        num_ref, vocab_size).reshape(num_ref, -1, vocab_size)

    index_batch = torch.arange(batch_size).cuda().long()
    index_vocab = torch.arange(vocab_size).cuda().long()

    changed1 = -logits.expand(num_ref, -1, -1) + torch.log(
        pi[:, R_cat].transpose(0, 1).reshape(num_ref, -1, 1))
    changed2 = torch.log(pi).expand(num_ref, -1, -1) - \
               logits[:, R_cat].transpose(0, 1).reshape(num_ref, -1, 1)

    pseudo_actions += (changed1 < min_value).long() * (
            index_vocab - A_cat.unsqueeze(1)).unsqueeze(0)
    pseudo_actions += (changed2 < min_value).long() * (
            R_cat.unsqueeze(1).expand(num_ref, batch_size) - A_cat).reshape(
        num_ref, -1, 1)

    pseudo_actions_true_move += (changed1 < sec_min_value).long() * (
            changed1 < changed2).long() * (
                index_vocab - sec_indices.unsqueeze(1)).unsqueeze(0)
    pseudo_actions_true_move += (changed2 < sec_min_value).long() * (
            changed2 < changed1).long() * (R_cat.unsqueeze(1).expand(
        num_ref, batch_size) - sec_indices).reshape(num_ref, -1, 1)

    index_matrix = torch.zeros_like(pseudo_actions).long()
    index_matrix[:, index_batch, A_cat] = 1
    index_matrix[R_cat.unsqueeze(1).expand(num_ref, batch_size) == A_cat, :] = 1

    pseudo_actions = pseudo_actions + index_matrix * (
            pseudo_actions_true_move - pseudo_actions)

    return pseudo_actions


def get_arm_loss(model, fc_feats, att_feats, att_masks, data, opt, loader, critic=None):
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
            if opt.arm_as_baseline == 1:
                if opt.critic_model != 'att_critic_vocab' or critic == None:
                    # TODO: log space.
                    arm_baseline[:, t-1] = arsm_f_delta_fun_batch_torch(logits.data, pi, data, seq, t, model, state, unfinished, loader,
                                                     opt, critic)
                elif opt.critic_model == 'att_critic_vocab' and critic is not None:
                    pseudo_action, pi_R = arsm_f_delta_fun_batch_torch(logits.data, pi, data, seq, t, model, state,
                                                                       unfinished, loader,
                                                                       opt, critic)
                    pseudo_action_list[:, t - 1, :] = pseudo_action
                    pi_list.append(pi_R)
            else:
                if opt.critic_model != 'att_critic_vocab' or critic == None:
                    f_delta = arsm_f_delta_fun_batch_torch(logits.data, pi, data, seq, t, model, state, unfinished, loader,
                                                     opt, critic)
                    f_delta = f_delta / temperature
                    f_delta = (f_delta.transpose(0, 1) * mask).transpose(0, 1)
                    mask_sum += torch.sum(mask)
                    loss -= torch.sum(f_delta.detach() * model.logit(output))
                elif opt.critic_model == 'att_critic_vocab' and critic is not None:
                    pseudo_action, pi_R = arsm_f_delta_fun_batch_torch(logits.data, pi, data, seq, t, model, state, unfinished, loader,
                                                     opt, critic)
                    pseudo_action_list[:, t - 1, :] = pseudo_action
                    pi_list.append(pi_R)
                    logprobs = logprobs / temperature
                    logprobs_list.append(model.logit(output))
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


def get_ar_loss(model, fc_feats, att_feats, att_masks, data, opt, loader, critic=None):
    batch_size = fc_feats.size(0)
    vocab_size = opt.vocab_size + 1
    state = model.init_hidden(batch_size)
    seq = fc_feats.new_zeros(batch_size, model.seq_length, dtype=torch.long)
    unfinished = fc_feats.new_ones(batch_size, dtype=torch.uint8)
    temperature = getattr(opt, 'temperature', 1.0)
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
        if t >= 1:
            logprobs1 = model.logit(output)
            logprobs = F.log_softmax(model.logit(output), dim=1)
            probs = F.softmax(model.logit(output), dim=1)
            pi = torch.from_numpy(np.random.dirichlet(np.ones(vocab_size), batch_size)).float().cuda()
            mask = unfinished.float()
            if temperature == 1.0:
                it = torch.min(torch.log(pi) - logprobs1, 1)[1].unsqueeze(1)
            else:
                it = torch.min(torch.log(pi) - logprobs1 / temperature, 1)[1].unsqueeze(1)
            it = it.view(-1).long()
            pi_list.append(pi)
            logprobs_list.append(logprobs1)
            if t == 1:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)

            it = it * unfinished.type_as(it)
            seq[:, t-1] = it
            true_length += 1
            if unfinished.sum() == 0:
                break
    loss = fc_feats.new_zeros([])

    ## evaluate reward
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
        f_delta = torch.from_numpy(np.repeat(np.expand_dims(arm_metric_value, 1), vocab_size, 1)).float().cuda() * (1.0 - vocab_size * pi_list[t])
        if t > 0:
            mask *= seq[:, t-1] > 0
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
                if test_critic:
                    q = test_critic_sampling(probs, f, opt).detach()
                else:
                    q = importance_sampling(probs, f, opt).detach()
                q_list.append(q)
                logprobs_sample = torch.log(q)
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


def arsm_f_delta_fun_batch_torch(logits, pi, data, pre_seq, step, model, state, unfinished, loader, opt, critic=None, type='ars', print_pseudo=True):
    #TODO: write in torch
    batch_size, vocab_size = logits.size()
    ref_num = opt.ref_num
    index_batch = torch.arange(batch_size).cuda().long()
    arm_metric_matrix = np.ones([batch_size, vocab_size*ref_num]) * -1
    index_vocab = torch.arange(vocab_size).cuda()
    temperature = getattr(opt, 'temperature', 1.0)
    A_cat = torch.min(torch.log(pi) - logits, 1)[1].long()
    topk, indices = torch.topk(logits, ref_num, dim=1)
    random_ref = torch.topk(torch.rand((batch_size, vocab_size)), ref_num)[1]
    f_delta = np.zeros([batch_size, vocab_size])
    if opt.ref_cat == 'random':
        R_cat_set = random_ref.cuda().long()
    elif opt.ref_cat == 'topaction':
        R_cat_set = indices.cuda().long()
    for i in range(ref_num):
        R_cat = R_cat_set[:, i]
        pseudo_actions_tmp = pseudo_action_fun(logits, A_cat, R_cat, pi)
        if i == 0:
            pseudo_actions = pseudo_actions_tmp
        else:
            pseudo_actions = torch.cat([pseudo_actions, pseudo_actions_tmp], 1)
    if unfinished.sum(0) != batch_size:
        pseudo_actions[(1 - unfinished), :] = A_cat[(1 - unfinished)].unsqueeze(1).repeat(1, vocab_size*ref_num)
    #print('time for pseudo action: ' + str(time() - tic))
    if opt.critic_model == 'att_critic_vocab':
        return pseudo_actions, pi[index_batch, R_cat]
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
        tic = time()
        seq_per_img = batch_size // len(data['gts'])
        gts = OrderedDict()
        for i in range(len(data['gts'])):
            gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]
        seqs_arm = seqs_arm.data.cpu().numpy()

        if print_pseudo and step == np.random.randint(20) and np.random.randint(20) == 1:
            sents = utils.decode_sequence(loader.get_vocab(), torch.from_numpy(seqs_arm[0:arm_pseudo_counts[0]]).cuda())
            print('imageid', data['infos'][0]['id'], '**********************At step ' + str(step))
            print('True sentence:' )
            print(sents[np.argmax(counts_per_sample_list[0])])
            print('Pseudo sentences: ')
            print(sents)
            print('Pseudo action mean: ', np.mean(arm_pseudo_index), 'std: ', np.std(arm_pseudo_index), 'max: ', np.max(arm_pseudo_index))
        res_ = []
        gts_arm = {}
        for i in range(len(arm_pseudo_action_set)):
            res_.append({'image_id': i, 'caption': [array_to_str(seqs_arm[i])]})
            i_index = arm_index_2[i]
            gts_arm[i] = gts[i_index // seq_per_img]
        _, arm_metric_value = CiderD_scorer.compute_score(gts_arm, res_)
    else:
        if opt.critic_model == 'state_critic':
            xt = model.embed(it)
            output, state_arm = model.core(xt, state_arm)
            arm_metric_value = critic.core(state_arm).detach().cpu().numpy()
    arm_index = np.array(arm_index)
    arm_index += np.repeat(np.expand_dims(np.concatenate([[0], np.cumsum(arm_pseudo_counts)[0:-1]]), 1), vocab_size*ref_num, 1)
    arm_index = np.reshape(arm_index, [-1])
    arm_pseudo_index = np.array(arm_pseudo_index)
    arm_metric_matrix[arm_pseudo_index > 1, :] = np.reshape(arm_metric_value[arm_index], [-1, vocab_size * ref_num])
    if opt.arm_as_baseline == 1:
        return torch.from_numpy(arm_metric_matrix).float().cuda().mean(1)
    for i in range(ref_num):
        R_cat = R_cat_set[:, i]
        arm_metric_matrix_tmp = arm_metric_matrix[:, (vocab_size*i):(vocab_size*(i+1))]
        f_delta_tmp = arm_metric_matrix_tmp - np.repeat(np.expand_dims(np.mean(arm_metric_matrix_tmp, 1), 1), vocab_size, 1)
        f_delta_tmp = f_delta_tmp * np.repeat(np.expand_dims(1.0 - vocab_size * pi[index_batch, R_cat].cpu().numpy(), 1), vocab_size, 1)
        f_delta += f_delta_tmp
    return torch.from_numpy(f_delta/ref_num).float().cuda()

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

def importance_sampling(prob, f, opt):
    unnormalized_q = prob + opt.is_weight * (prob.pow(2).sum(1).unsqueeze(1).repeat(1, prob.size()[1]) + 1 - prob * 2).pow(0.5) * prob * torch.abs(f)
    q = torch.div(unnormalized_q, (unnormalized_q.sum(1).unsqueeze(1).repeat(1, prob.size()[1]) + epsilon))
    return q

def test_critic_sampling(prob, f, opt):
    unnormalized_q = torch.abs(f)
    q = torch.div(unnormalized_q, (unnormalized_q.sum(1).unsqueeze(1).repeat(1, prob.size()[1]) + epsilon))
    return q
