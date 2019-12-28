import itertools
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import copy

def entropy_fun(p):
    #p: batch * vocab
    return -(p * torch.log(p)).sum(1).mean(0)

def do_supervised_minibatch(model,
                            # Source
                            inp_grids, out_grids,
                            # Target
                            in_tgt_seq, in_tgt_seq_list, out_tgt_seq,
                            # Criterion
                            criterion):

    # Get the log probability of each token in the ground truth sequence of tokens.
    decoder_logit, _ = model(inp_grids, out_grids, in_tgt_seq, in_tgt_seq_list)

    nb_predictions = torch.numel(out_tgt_seq.data)
    # criterion is a weighted CrossEntropyLoss. The weights are used to not penalize
    # the padding prediction used to make the batch of the appropriate size.
    loss = criterion(
        decoder_logit.contiguous().view(nb_predictions, decoder_logit.size(2)),
        out_tgt_seq.view(nb_predictions)
    )

    # Do the backward pass over the loss
    loss.backward()

    # Return the value of the loss over the minibatch for monitoring
    return loss.item()

def do_syntax_weighted_minibatch(model,
                                 # Source
                                 inp_grids, out_grids,
                                 # Target
                                 in_tgt_seq, in_tgt_seq_list, out_tgt_seq,
                                 # Criterion
                                 criterion,
                                 # Beta trades off between CrossEntropyLoss
                                 # and SyntaxLoss
                                 beta):

    # Get the log probability of each token in the ground truth sequence of tokens,
    # for both the IO based model and the syntax checker.
    decoder_logit, syntax_logit = model(inp_grids, out_grids,
                                        in_tgt_seq, in_tgt_seq_list)
    # {decoder,syntax}_logit: batch_size x seq_len x vocab_size

    # The criterion is the same as in `do_supervised_minibatch`
    nb_predictions = torch.numel(out_tgt_seq.data)
    ce_loss = criterion(
        decoder_logit.contiguous().view(nb_predictions, decoder_logit.size(2)),
        out_tgt_seq.view(nb_predictions)
    )

    # A syntax loss is also computed, penalizing any masking of tokens that we
    # know are valid, given that they are in the ground truth sequence.
    syntax_loss = -syntax_logit.gather(2, out_tgt_seq.unsqueeze(2)).sum()

    loss = ce_loss + beta*syntax_loss
    # Do the backward pass over the loss
    loss.backward()

    # Return the value of the loss over the minibatch for monitoring
    return loss.item()

def do_rl_minibatch(model,
                    # Source
                    inp_grids, out_grids,
                    # Target
                    envs,
                    # Config
                    tgt_start_idx, tgt_end_idx, max_len,
                    nb_rollouts):

    # Samples `nb_rollouts` samples from the decoding model.
    rolls = model.sample_model(inp_grids, out_grids,
                               tgt_start_idx, tgt_end_idx, max_len,
                               nb_rollouts, vol=False)
    for roll, env in zip(rolls, envs):
        # Assign the rewards for each sample
        roll.assign_rewards(env, [])

    # Evaluate the performance on the minibatch
    batch_reward = sum(roll.dep_reward for roll in rolls)

    # Get all variables and all gradients from all the rolls
    variables, grad_variables = zip(*batch_rolls_reinforce(rolls))

    # For each of the sampling probability, we know their gradients.
    # See https://arxiv.org/abs/1506.05254 for what we are doing,
    # simply using the probability of the choice made, times the reward of all successors.
    autograd.backward(variables, grad_variables)

    # Return the value of the loss/reward over the minibatch for convergence
    # monitoring.
    return batch_reward

def do_rl_minibatch_two_steps(model,
                              # Source
                              inp_grids, out_grids,
                              # Target
                              envs,
                              # Config
                              tgt_start_idx, tgt_end_idx, pad_idx, max_len,
                              nb_rollouts, rl_inner_batch):
    '''
    This is an alternative to do simple expected reward maximization.
    The problem with the previous method of `do_rl_minibatch` is that it is
    memory demanding, as due to all the sampling steps / bookkeeping, the graph
    becomes large / complex. It's entirely possible that future version of pytorch
    will fix this but this has proven quite useful.

    The idea is to first sample the `nb_rollouts` samples, doing all the process with
    Volatile=True, so that no graph needs to be held.
    Once the samples have been sampled, we re-evaluate them through the
    `score_multiple_decs` functions that returns the log probabilitise of several decodings.
    This has the disadvantage that we don't make use of shared elements in the sequences,
    but that way, the decoding graph is much simpler (scoring of the whole decoded sequence
    at once vs. one timestep at a time to allow proper input feeding.)
    '''
    use_cuda = inp_grids.is_cuda
    tt = torch.cuda if use_cuda else torch
    rolls = model.sample_model(inp_grids, out_grids,
                               tgt_start_idx, tgt_end_idx, max_len,
                               nb_rollouts, vol=True)
    for roll, env in zip(rolls, envs):
        # Assign the rewards for each sample
        roll.assign_rewards(env, [])
    batch_reward = sum(roll.dep_reward for roll in rolls)

    for start_pos in range(0, len(rolls), rl_inner_batch):
        roll_ib = rolls[start_pos: start_pos + rl_inner_batch]
        to_score = []
        nb_cand_per_sp = []
        rews = []
        for roll in roll_ib:
            nb_cand = 0
            for trajectory, multiplicity, _, rew in roll.yield_final_trajectories():
                to_score.append(trajectory)
                rews.append(multiplicity*rew)
                nb_cand += 1
            nb_cand_per_sp.append(nb_cand)

        in_tgt_seqs = []
        lines = [[tgt_start_idx] + line  for line in to_score]
        lens = [len(line) for line in lines]
        ib_max_len = max(lens)
        inp_lines = [
            line[:ib_max_len-1] + [pad_idx] * (ib_max_len - len(line[:ib_max_len-1])-1) for line in lines
        ]
        out_lines = [
            line[1:] + [pad_idx] * (ib_max_len - len(line)) for line in lines
        ]
        in_tgt_seq = Variable(torch.LongTensor(inp_lines))
        out_tgt_seq = Variable(torch.LongTensor(out_lines))
        if use_cuda:
            in_tgt_seq, out_tgt_seq = in_tgt_seq.cuda(), out_tgt_seq.cuda()
        out_care_mask = (out_tgt_seq != pad_idx)

        inner_batch_in_grids = inp_grids.narrow(0, start_pos, rl_inner_batch)
        inner_batch_out_grids = out_grids.narrow(0, start_pos, rl_inner_batch)

        tkns_lpb = model.score_multiple_decs(inner_batch_in_grids,
                                             inner_batch_out_grids,
                                             in_tgt_seq, inp_lines,
                                             out_tgt_seq, nb_cand_per_sp)
        # tkns_lpb contains the log probability of each choice that was taken

        tkns_pb = tkns_lpb.exp()
        # tkns_lpb contains the probability of each choice that was taken

        # The gradient on the probability of a multinomial choice is
        # 1/p * (rewards after)
        tkns_invpb = tkns_pb.data.reciprocal()

        # The gradient on the probability will get multiplied by the reward We
        # also take advantage of the fact that we have a mask to avoid putting
        # gradients on padding
        rews_tensor = tt.FloatTensor(rews).unsqueeze(1).expand_as(tkns_invpb)
        reinforce_grad = torch.mul(-rews_tensor, out_care_mask.data.float())
        torch.mul(reinforce_grad, tkns_invpb, out=reinforce_grad)

        tkns_pb.backward(reinforce_grad)

    return batch_reward

#TODO: what is reference program, check the effect of rl_use_ref
def do_beam_rl(model,
               # source
               inp_grids, out_grids, targets,
               # Target
               envs, reward_comb_fun,
               # Config
               tgt_start_idx, tgt_end_idx, pad_idx,
               max_len, beam_size, rl_inner_batch, rl_use_ref):
    '''
    Rather than doing an actual expected reward,
    evaluate the most likely programs using a beam search (with `beam_sample`)
    If `rl_use_ref` is set, include the reference program in the search.
    Similarly to `do_rl_minibatch_two_steps`, first decode the programs as Volatile,
    then score them.
    '''
    batch_reward = 0
    use_cuda = inp_grids.is_cuda
    tt = torch.cuda if use_cuda else torch
    vol_inp_grids = Variable(inp_grids.data, volatile=True)
    vol_out_grids = Variable(out_grids.data, volatile=True)
    # Get the programs from the beam search
    decoded = model.beam_sample(vol_inp_grids, vol_out_grids,
                                tgt_start_idx, tgt_end_idx, max_len,
                                beam_size, beam_size)
    vallina_reward = 0
    # list of length batch size, each component containing beam size number of sentences
    # For each element in the batch, get the version of the log proba that can use autograd.
    for start_pos in range(0, len(decoded), rl_inner_batch):
        to_score = decoded[start_pos: start_pos + rl_inner_batch]
        scorers = envs[start_pos: start_pos + rl_inner_batch]
        # Eventually add the reference program
        if rl_use_ref:
            references = [target for target in targets[start_pos: start_pos + rl_inner_batch]]
            for ref, candidates_to_score in zip(references, to_score):
                for _, predded in candidates_to_score:
                    if ref == predded:
                        break
                else:
                    candidates_to_score.append((None, ref)) # Don't know its lpb

        # Build the inputs to be scored
        nb_cand_per_sp = [len(candidates) for candidates in to_score]
        in_tgt_seqs = []
        preds  = [pred for lp, pred in itertools.chain(*to_score)]
        lines = [[tgt_start_idx] + line  for line in preds]
        lens = [len(line) for line in lines]
        ib_max_len = max(lens)

        inp_lines = [
            line[:ib_max_len-1] + [pad_idx] * (ib_max_len - len(line[:ib_max_len-1])-1) for line in lines
        ]
        out_lines = [
            line[1:] + [pad_idx] * (ib_max_len - len(line)) for line in lines
        ]
        in_tgt_seq = Variable(torch.LongTensor(inp_lines))
        out_tgt_seq = Variable(torch.LongTensor(out_lines))
        if use_cuda:
            in_tgt_seq, out_tgt_seq = in_tgt_seq.cuda(), out_tgt_seq.cuda()
        out_care_mask = (out_tgt_seq != pad_idx)

        inner_batch_in_grids = inp_grids.narrow(0, start_pos, len(to_score))
        inner_batch_out_grids = out_grids.narrow(0, start_pos, len(to_score))

        # Get the scores for the programs we decoded.
        seq_lpb_var = model.score_multiple_decs(inner_batch_in_grids,
                                                inner_batch_out_grids,
                                                in_tgt_seq, inp_lines,
                                                out_tgt_seq, nb_cand_per_sp)
        #print('seq_lpb_var', seq_lpb_var.size())
        lpb_var = torch.mul(seq_lpb_var, out_care_mask.float()).sum(1)
        #print('lpb_var', lpb_var.size()) #512 sentences
        # Compute the reward that were obtained by each of the sampled programs
        per_sp_reward = []
        for env, all_decs in zip(scorers, to_score):
            sp_rewards = []
            for (lpb, dec) in all_decs:
                sp_rewards.append(env.step_reward(dec, True))
                #print(env.step_reward(dec, True))
            per_sp_reward.append(sp_rewards)
        # all_decs containing beam size number of sequences
        # sp_rewards containing beam size number of rewards
        # per_sp_reward: inner batch * beam size
        per_sp_lpb = []
        start = 0
        for nb_cand in nb_cand_per_sp:
            per_sp_lpb.append(lpb_var.narrow(0, start, nb_cand))
            start += nb_cand

        # Use the reward combination function to get our loss on the minibatch
        # (See `reinforce.py`, possible choices are RenormExpected and the BagExpected)
        inner_batch_reward = 0
        for pred_lpbs, pred_rewards in zip(per_sp_lpb, per_sp_reward):
            #print('1', pred_lpbs, pred_rewards)
            inner_batch_reward += reward_comb_fun(pred_lpbs, pred_rewards)
            vallina_reward += sum(pred_rewards)*1.0 / len(pred_rewards)
            #inner_batch_reward += (pred_lpbs * torch.from_numpy(
            #    reward_transform(pred_rewards, 1) - np.mean(reward_transform(pred_rewards, 1))).cuda().float()).mean()
            #inner_batch_reward += (pred_lpbs * torch.from_numpy(pred_rewards-np.mean(pred_rewards)).cuda().float()).mean()

        # We put a minus sign here because we want to maximize the reward.
        (-inner_batch_reward/len(decoded)).backward()

        batch_reward += inner_batch_reward.item()
    #print('reward', vallina_reward * 1.0 / len(decoded))

    return batch_reward


def get_sample(model, inp_grids, out_grids, tgt_start_idx, tgt_end_idx, max_len, arm_sample='greedy'):
    use_cuda = inp_grids.is_cuda
    tt = torch.cuda if use_cuda else torch
    io_embeddings = model.encoder(inp_grids, out_grids)
    batch_size, nb_ios, io_emb_size = io_embeddings.size()
    batch_state = None
    batch_grammar_state = None
    batch_inputs = inp_grids.new_ones(batch_size, dtype=torch.long) * tgt_start_idx
    batch_list_inputs = [[tgt_start_idx]]*batch_size
    batch_io_embeddings = io_embeddings
    #TODO: start and end should all be included, and end with end, what if sampled start index?

    seq = inp_grids.new_ones(batch_size, max_len + 1, dtype=torch.long) * tgt_end_idx
    seq_log = inp_grids.new_zeros(batch_size, max_len)
    loss = inp_grids.new_zeros([])
    unfinished = inp_grids.new_ones(batch_size, dtype=torch.int8)
    mask_sum = 0
    ## initialize unfinished:
    batch_inputs_unf = batch_inputs
    batch_list_inputs_unf = batch_list_inputs
    batch_state_unf = batch_state
    batch_grammar_state_unf = batch_grammar_state
    batch_io_embeddings_unf = batch_io_embeddings
    unfinished_index = torch.arange(batch_size).long().cuda()
    for t in range(max_len + 1): #TODO: reduce one iteration
        seq[unfinished_index, t] = batch_inputs_unf
        if t == max_len:
            break
        dec_outs, batch_state_unf, \
        batch_grammar_state_unf, _ = model.decoder.forward(batch_inputs_unf.unsqueeze(1),
                                                       batch_io_embeddings_unf,
                                                       batch_list_inputs_unf,
                                                       batch_state_unf,
                                                       batch_grammar_state_unf)
        dec_outs = dec_outs.squeeze(1) # batch * vocab
        vocab_size = dec_outs.size(1)
        logprobs = F.log_softmax(dec_outs, dim=1)
        unfinished = inp_grids.new_ones(dec_outs.size(0), dtype=torch.int8)
        mask = unfinished.float()
        mask_sum += torch.sum(mask)
        if arm_sample == 'greedy':
            batch_inputs = torch.max(logprobs.data, 1)[1]
        else:
            batch_inputs = torch.multinomial(torch.exp(logprobs.data).cpu(), 1).cuda().squeeze(1)
        batch_inputs = batch_inputs.long()
        seq_log[unfinished_index, t] = logprobs.gather(1, batch_inputs.unsqueeze(1)).squeeze(1)
        batch_inputs_unf = batch_inputs
        step_unfinished = batch_inputs_unf != tgt_end_idx
        batch_inputs_unf = batch_inputs_unf[step_unfinished]
        batch_state_unf = (batch_state_unf[0][:, step_unfinished, :, :],
                           batch_state_unf[1][:, step_unfinished, :, :])
        batch_io_embeddings_unf = batch_io_embeddings_unf[step_unfinished]
        batch_list_inputs_unf = list_fun(batch_inputs_unf)
        if batch_grammar_state_unf is not None:
            batch_grammar_state_unf_new = []
            for i, item in enumerate(batch_grammar_state_unf):
                if step_unfinished[i]:
                    batch_grammar_state_unf_new.append(copy.copy(item))
            batch_grammar_state_unf = batch_grammar_state_unf_new

        unfinished_index = unfinished_index[step_unfinished]
        if step_unfinished.sum() == 0:
            break
    return seq, seq_log


def get_self_critic_loss(model, inp_grids, out_grids, envs, tgt_start_idx, tgt_end_idx, max_len, arm_sample='greedy'):
    seq, seq_log = get_sample(model, inp_grids, out_grids, tgt_start_idx, tgt_end_idx, max_len, arm_sample='sample')
    greedy_seq, _ = get_sample(model, inp_grids, out_grids, tgt_start_idx, tgt_end_idx, max_len, arm_sample='greedy')
    batch_size = inp_grids.size(0)
    arm_metric_value = np.zeros(batch_size)
    arm_metric_value_greedy = np.zeros(batch_size)
    for i in range(batch_size):
        scorer = envs[i]
        arm_metric_value[i] = scorer.step_reward(pre_scoring(seq[i], tgt_end_idx), True)
        arm_metric_value_greedy[i] = scorer.step_reward(pre_scoring(greedy_seq[i], tgt_end_idx), True)
    reward = arm_metric_value - arm_metric_value_greedy

    loss = -(seq_log.sum(1) * torch.from_numpy(reward).cuda().float()).mean(0)
    loss.backward()
    return loss.item()


def get_rf_demean_loss(model, inp_grids, out_grids, envs, tgt_start_idx, tgt_end_idx, max_len, arm_sample='greedy'):
    seq, seq_log = get_sample(model, inp_grids, out_grids, tgt_start_idx, tgt_end_idx, max_len, arm_sample='sample')
    batch_size = inp_grids.size(0)
    arm_metric_value = np.zeros(batch_size)
    for i in range(batch_size):
        scorer = envs[i]
        arm_metric_value[i] = scorer.step_reward(pre_scoring(seq[i], tgt_end_idx), True)
    reward = arm_metric_value# - np.mean(arm_metric_value)

    loss = -(seq_log.sum(1) * torch.from_numpy(reward).cuda().float()).mean(0)
    loss.backward()
    return loss.item()


def get_mct_loss_2(model, inp_grids, out_grids, envs, tgt_start_idx, tgt_end_idx, max_len, arm_sample='greedy',
                 decay_factor=1, logits_factor=0):
    use_cuda = inp_grids.is_cuda
    tt = torch.cuda if use_cuda else torch
    io_embeddings = model.encoder(inp_grids, out_grids)
    batch_size, nb_ios, io_emb_size = io_embeddings.size()
    batch_state = None
    batch_grammar_state = None
    batch_inputs = inp_grids.new_ones(batch_size, dtype=torch.long) * tgt_start_idx
    batch_list_inputs = [[tgt_start_idx]] * batch_size
    batch_io_embeddings = io_embeddings
    # TODO: start and end should all be included, and end with end, what if sampled start index?

    seq = inp_grids.new_ones(batch_size, max_len + 1, dtype=torch.long) * tgt_end_idx
    seq_log = inp_grids.new_zeros(batch_size, max_len)
    word_baseline = inp_grids.new_zeros(batch_size, max_len)
    loss = inp_grids.new_zeros([])
    unfinished = inp_grids.new_ones(batch_size, dtype=torch.int8)
    mask_sum = 0
    ## initialize unfinished:
    batch_inputs_unf = batch_inputs
    batch_list_inputs_unf = batch_list_inputs
    batch_state_unf = batch_state
    batch_grammar_state_unf = batch_grammar_state
    batch_io_embeddings_unf = batch_io_embeddings
    unfinished_index = torch.arange(batch_size).long().cuda()
    for t in range(max_len + 1):  # TODO: reduce one iteration
        seq[unfinished_index, t] = batch_inputs_unf
        if t == max_len:
            break
        dec_outs, batch_state_unf, \
        batch_grammar_state_unf, _ = model.decoder.forward(batch_inputs_unf.unsqueeze(1),
                                                           batch_io_embeddings_unf,
                                                           batch_list_inputs_unf,
                                                           batch_state_unf,
                                                           batch_grammar_state_unf)
        dec_outs = dec_outs.squeeze(1)  # batch * vocab
        vocab_size = dec_outs.size(1)
        logprobs = F.log_softmax(dec_outs, dim=1)
        unfinished = inp_grids.new_ones(dec_outs.size(0), dtype=torch.int8)
        word_baseline[unfinished_index, t] = complete_batch_fun(logprobs.data, copy.copy(envs), seq,
                                                           t, model, copy.copy(batch_state_unf),
                                                           copy.copy(batch_grammar_state_unf),
                                                           unfinished,
                                                           batch_io_embeddings_unf,
                                                           tgt_end_idx, max_len, arm_sample,
                                                           seq_log[unfinished_index], logits_factor)
        mask = unfinished.float()
        mask_sum += torch.sum(mask)
        if arm_sample == 'greedy':
            batch_inputs = torch.max(logprobs.data, 1)[1]
        else:
            batch_inputs = torch.multinomial(torch.exp(logprobs.data).cpu(), 1).cuda().squeeze(1)
        batch_inputs = batch_inputs.long()
        seq_log[unfinished_index, t] = logprobs.gather(1, batch_inputs.unsqueeze(1)).squeeze(1)
        batch_inputs_unf = batch_inputs
        step_unfinished = batch_inputs_unf != tgt_end_idx
        batch_inputs_unf = batch_inputs_unf[step_unfinished]
        batch_state_unf = (batch_state_unf[0][:, step_unfinished, :, :],
                           batch_state_unf[1][:, step_unfinished, :, :])
        batch_io_embeddings_unf = batch_io_embeddings_unf[step_unfinished]
        batch_list_inputs_unf = list_fun(batch_inputs_unf)
        if batch_grammar_state_unf is not None:
            batch_grammar_state_unf_new = []
            for i, item in enumerate(batch_grammar_state_unf):
                if step_unfinished[i]:
                    batch_grammar_state_unf_new.append(copy.copy(item))
            batch_grammar_state_unf = batch_grammar_state_unf_new

        unfinished_index = unfinished_index[step_unfinished]
        if step_unfinished.sum() == 0:
            break

    batch_size = inp_grids.size(0)
    arm_metric_value = np.zeros(batch_size)
    for i in range(batch_size):
        scorer = envs[i]
        arm_metric_value[i] = scorer.step_reward(pre_scoring(seq[i], tgt_end_idx), True)
    reward = torch.from_numpy(arm_metric_value).cuda().float().unsqueeze(1).repeat(1, max_len) - word_baseline

    loss = -(seq_log * reward).sum(1).mean(0)
    loss.backward()
    return loss.item()


def get_mct_loss(model, inp_grids, out_grids, envs, tgt_start_idx, tgt_end_idx, max_len, arm_sample='greedy',
                 decay_factor=1, logits_factor=0):
    use_cuda = inp_grids.is_cuda
    tt = torch.cuda if use_cuda else torch
    io_embeddings = model.encoder(inp_grids, out_grids)
    batch_size, nb_ios, io_emb_size = io_embeddings.size()
    batch_state = None
    batch_grammar_state = None
    batch_inputs = inp_grids.new_ones(batch_size, dtype=torch.long) * tgt_start_idx
    batch_list_inputs = [[tgt_start_idx]] * batch_size
    batch_io_embeddings = io_embeddings
    # TODO: start and end should all be included, and end with end, what if sampled start index?

    seq = inp_grids.new_ones(batch_size, max_len + 1, dtype=torch.long) * tgt_end_idx
    seq_log = inp_grids.new_zeros(batch_size, max_len)
    word_baseline = inp_grids.new_zeros(batch_size, max_len)
    loss = inp_grids.new_zeros([])
    unfinished = inp_grids.new_ones(batch_size, dtype=torch.int8)
    mask_sum = 0
    ## initialize unfinished:
    batch_inputs_unf = batch_inputs
    batch_list_inputs_unf = batch_list_inputs
    batch_state_unf = batch_state
    batch_grammar_state_unf = batch_grammar_state
    batch_io_embeddings_unf = batch_io_embeddings
    unfinished_index = torch.arange(batch_size).long().cuda()
    for t in range(max_len + 1):  # TODO: reduce one iteration
        seq[unfinished_index, t] = batch_inputs_unf
        if t == max_len:
            break
        dec_outs, batch_state_unf, \
        batch_grammar_state_unf, _ = model.decoder.forward(batch_inputs_unf.unsqueeze(1),
                                                           batch_io_embeddings_unf,
                                                           batch_list_inputs_unf,
                                                           batch_state_unf,
                                                           batch_grammar_state_unf)
        dec_outs = dec_outs.squeeze(1)  # batch * vocab
        vocab_size = dec_outs.size(1)
        logprobs = F.log_softmax(dec_outs, dim=1)
        unfinished = inp_grids.new_ones(dec_outs.size(0), dtype=torch.int8)
        word_baseline[unfinished_index, t] = complete_batch_fun(logprobs.data, copy.copy(envs), seq,
                                                           t, model, copy.copy(batch_state_unf),
                                                           copy.copy(batch_grammar_state_unf),
                                                           unfinished,
                                                           batch_io_embeddings_unf,
                                                           tgt_end_idx, max_len, arm_sample,
                                                           seq_log[unfinished_index], logits_factor)
        mask = unfinished.float()
        mask_sum += torch.sum(mask)
        if arm_sample == 'greedy':
            batch_inputs = torch.max(logprobs.data, 1)[1]
        else:
            batch_inputs = torch.multinomial(torch.exp(logprobs.data).cpu(), 1).cuda().squeeze(1)
        batch_inputs = batch_inputs.long()
        seq_log[unfinished_index, t] = logprobs.gather(1, batch_inputs.unsqueeze(1)).squeeze(1)
        batch_inputs_unf = batch_inputs
        step_unfinished = batch_inputs_unf != tgt_end_idx
        batch_inputs_unf = batch_inputs_unf[step_unfinished]
        batch_state_unf = (batch_state_unf[0][:, step_unfinished, :, :],
                           batch_state_unf[1][:, step_unfinished, :, :])
        batch_io_embeddings_unf = batch_io_embeddings_unf[step_unfinished]
        batch_list_inputs_unf = list_fun(batch_inputs_unf)
        if batch_grammar_state_unf is not None:
            batch_grammar_state_unf_new = []
            for i, item in enumerate(batch_grammar_state_unf):
                if step_unfinished[i]:
                    batch_grammar_state_unf_new.append(copy.copy(item))
            batch_grammar_state_unf = batch_grammar_state_unf_new

        unfinished_index = unfinished_index[step_unfinished]
        if step_unfinished.sum() == 0:
            break

    batch_size = inp_grids.size(0)
    arm_metric_value = np.zeros(batch_size)
    for i in range(batch_size):
        scorer = envs[i]
        arm_metric_value[i] = scorer.step_reward(pre_scoring(seq[i], tgt_end_idx), True)

    reward = torch.cat([word_baseline[:, 1:], torch.from_numpy(arm_metric_value).cuda().float().unsqueeze(1)], 1)

    greedy_seq, _ = get_sample(model, inp_grids, out_grids, tgt_start_idx, tgt_end_idx, max_len, arm_sample='greedy')
    batch_size = inp_grids.size(0)
    arm_metric_value_greedy = np.zeros(batch_size)
    for i in range(batch_size):
        scorer = envs[i]
        arm_metric_value_greedy[i] = scorer.step_reward(pre_scoring(greedy_seq[i], tgt_end_idx), True)

    reward = reward - torch.from_numpy(arm_metric_value_greedy).cuda().float().unsqueeze(1).repeat(1, max_len)
    loss = -(seq_log * reward).sum(1).mean(0)
    loss.backward()
    return loss.item()


def complete_batch_fun(logits, envs, pre_seq, step, model, batch_state, batch_grammar_state,
                       unfinished, batch_io_embeddings, tgt_end_idx, max_len, arm_sample, seq_log, logits_factor):
    mct_sample_num = 5
    batch_size, vocab = logits.size()
    rewards = np.zeros([batch_size, mct_sample_num])
    arm_metric_matrix = np.ones([batch_size, mct_sample_num]) * -1
    pseudo_actions = torch.multinomial(torch.exp(logits.data).cpu(), mct_sample_num, replacement=True).cuda()
    arm_pseudo_action_set = []
    arm_index = []
    arm_index_2 = np.zeros(0)
    arm_pseudo_counts = []
    counts_per_sample_list = []
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
    # print('pseudo numbers', np.mean(arm_pseudo_index))

    if np.sum(arm_pseudo_index) == batch_size:
        return torch.zeros(batch_size).float().cuda()
    seqs_arm = pre_seq[arm_index_2, :]
    #seq_log_arm = seq_log[arm_index_2]
    batch_io_embeddings_arm = batch_io_embeddings[arm_index_2]
    batch_inputs_arm = torch.from_numpy(arm_pseudo_action_set).long().cuda()
    #seq_log_arm += logits[arm_index_2, :].gather(1, batch_inputs_arm.unsqueeze(1)).squeeze(1)
    unfinished_arm = (batch_inputs_arm != tgt_end_idx)
    batch_state_arm = (batch_state[0][:, arm_index_2, :, :], batch_state[1][:, arm_index_2, :, :])
    batch_grammar_state_arm = []
    if batch_grammar_state is not None:
        for i in range(len(arm_index_2)):
            batch_grammar_state_arm.append(copy.copy(batch_grammar_state[int(arm_index_2[i])]))

    ## initial unfinished:
    batch_inputs_arm_unf = batch_inputs_arm[unfinished_arm]
    batch_list_inputs_arm_unf = list_fun(batch_inputs_arm_unf)
    batch_state_arm_unf = (batch_state_arm[0][:, unfinished_arm, :, :], batch_state_arm[1][:, unfinished_arm, :, :])
    if batch_grammar_state_arm is not None:
        batch_grammar_state_arm_unf = []
        for i, item in enumerate(batch_grammar_state_arm):
            if unfinished_arm[i]:
                batch_grammar_state_arm_unf.append(copy.copy(item))
    batch_io_embeddings_arm_unf = batch_io_embeddings_arm[unfinished_arm]
    unfinished_index = torch.arange(len(arm_index_2)).long().cuda()[unfinished_arm]

    for t in range(step + 1, max_len + 1):
        seqs_arm[unfinished_index, t] = batch_inputs_arm_unf
        if t == max_len:
            break
        # TODO: only run unfinished
        dec_outs, batch_state_arm_unf, batch_grammar_state_arm_unf, _ = \
            model.decoder.forward(batch_inputs_arm_unf.unsqueeze(1), batch_io_embeddings_arm_unf,
                                  batch_list_inputs_arm_unf, batch_state_arm_unf,
                                  batch_grammar_state_arm_unf)
        # batch_grammar_state_arm = [copy.copy(item) for item in batch_grammar_state_arm]
        dec_outs = dec_outs.squeeze(1)
        logprobs = F.log_softmax(dec_outs, dim=1)
        if arm_sample == 'greedy':
            batch_inputs_arm = torch.max(logprobs.data, 1)[1]
        else:
            batch_inputs_arm = torch.multinomial(torch.exp(logprobs.data).cpu(), 1).cuda().squeeze(1)
        batch_inputs_arm = batch_inputs_arm.long()
        #seq_log_arm[unfinished_index] = seq_log_arm[unfinished_index] + \
        #                                logprobs.data.gather(1, batch_inputs_arm.unsqueeze(1)).squeeze(1)
        batch_inputs_arm_unf = batch_inputs_arm

        step_unfinished = batch_inputs_arm_unf != tgt_end_idx
        batch_inputs_arm_unf = batch_inputs_arm_unf[step_unfinished]
        batch_state_arm_unf = (batch_state_arm_unf[0][:, step_unfinished, :, :],
                               batch_state_arm_unf[1][:, step_unfinished, :, :])
        batch_io_embeddings_arm_unf = batch_io_embeddings_arm_unf[step_unfinished]
        batch_list_inputs_arm_unf = list_fun(batch_inputs_arm_unf)
        if batch_grammar_state_arm_unf is not None:
            batch_grammar_state_arm_unf_new = []
            for i, item in enumerate(batch_grammar_state_arm_unf):
                if step_unfinished[i]:
                    batch_grammar_state_arm_unf_new.append(copy.copy(item))
        batch_grammar_state_arm_unf = batch_grammar_state_arm_unf_new

        unfinished_index = unfinished_index[step_unfinished]
        if step_unfinished.sum() == 0:
            break
    # print(seqs_arm)

    ## evaluate reward
    arm_metric_value = np.zeros(seqs_arm.size(0))
    for i in range(seqs_arm.size(0)):
        scorer = envs[int(arm_index_2[i])]
        arm_metric_value[i] = scorer.step_reward(pre_scoring(seqs_arm[i], tgt_end_idx), True)
    # print('pseudo reward', np.mean(arm_metric_value))
    # arm_metric_value += logits_factor * seq_log_arm.data.cpu().numpy()
    arm_index = np.array(arm_index)
    arm_index += np.repeat(np.expand_dims(np.concatenate([[0], np.cumsum(arm_pseudo_counts)[0:-1]]), 1),
                           mct_sample_num, 1)
    arm_index = np.reshape(arm_index, [-1])
    arm_pseudo_index = np.array(arm_pseudo_index)
    arm_metric_matrix[arm_pseudo_index > 1, :] = np.reshape(arm_metric_value[arm_index], [-1, mct_sample_num])
    return torch.from_numpy(np.mean(arm_metric_matrix, 1)).float().cuda()

def get_arm_loss(model, inp_grids, out_grids, envs, tgt_start_idx, tgt_end_idx, max_len, arm_sample='greedy', decay_factor=1, logits_factor=0):
    use_cuda = inp_grids.is_cuda
    tt = torch.cuda if use_cuda else torch
    io_embeddings = model.encoder(inp_grids, out_grids)
    batch_size, nb_ios, io_emb_size = io_embeddings.size()
    batch_state = None
    batch_grammar_state = None
    batch_inputs = inp_grids.new_ones(batch_size, dtype=torch.long) * tgt_start_idx
    batch_list_inputs = [[tgt_start_idx]]*batch_size
    batch_io_embeddings = io_embeddings
    #TODO: start and end should all be included, and end with end, what if sampled start index?

    seq = inp_grids.new_ones(batch_size, max_len + 1, dtype=torch.long) * tgt_end_idx
    seq_log = inp_grids.new_zeros(batch_size)
    loss = inp_grids.new_zeros([])
    unfinished = inp_grids.new_ones(batch_size, dtype=torch.int8)
    mask_sum = 0
    entropy_reg = 0.0001
    ## initialize unfinished:
    batch_inputs_unf = batch_inputs
    batch_list_inputs_unf = batch_list_inputs
    batch_state_unf = batch_state
    batch_grammar_state_unf = batch_grammar_state
    batch_io_embeddings_unf = batch_io_embeddings
    unfinished_index = torch.arange(batch_size).long().cuda()
    backward_flag = 0
    pseudo_num_list = []
    entropy = 0
    for t in range(max_len + 1): #TODO: reduce one iteration
        seq[unfinished_index, t] = batch_inputs_unf
        if t == max_len:
            break
        dec_outs, batch_state_unf, \
        batch_grammar_state_unf, _ = model.decoder.forward(batch_inputs_unf.unsqueeze(1),
                                                       batch_io_embeddings_unf,
                                                       batch_list_inputs_unf,
                                                       batch_state_unf,
                                                       batch_grammar_state_unf)
        dec_outs = dec_outs.squeeze(1) # batch * vocab
        vocab_size = dec_outs.size(1)
        logprobs = F.log_softmax(dec_outs, dim=1)
        entropy = entropy + entropy_fun(F.softmax(logprobs, 1))
        pi = torch.from_numpy(np.random.dirichlet(np.ones(vocab_size), dec_outs.size(0))).float().cuda()
        unfinished = inp_grids.new_ones(dec_outs.size(0), dtype=torch.int8)
        mask = unfinished.float()
        f_delta, yes_pseudo, pseudo_num = arsm_f_delta_fun_batch_torch(logprobs.data, pi, copy.copy(envs), seq,
                                                           t, model, copy.copy(batch_state_unf),
                                                           copy.copy(batch_grammar_state_unf),
                                                           unfinished,
                                                           batch_io_embeddings_unf,
                                                           tgt_end_idx, max_len, arm_sample,
                                                           seq_log[unfinished_index], logits_factor)
        pseudo_num_list.append(pseudo_num)
        if yes_pseudo:
            tmp = f_delta.detach() * dec_outs * (dec_outs != -float('inf')).type_as(dec_outs)
            #print(f_delta[0,:])
            #print(dec_outs[0,:])
            #print(tmp[1 - torch.isnan(tmp)])
            loss -= tmp[1 - torch.isnan(tmp)].sum()
            backward_flag = 1
        if arm_sample == 'greedy':
            batch_inputs = torch.max(logprobs.data, 1)[1]
        else:
            batch_inputs = torch.multinomial(torch.exp(logprobs.data).cpu(), 1).cuda().squeeze(1)
        batch_inputs = batch_inputs.long()
        seq_log[unfinished_index] = seq_log[unfinished_index] + logprobs.data.gather(1, batch_inputs.unsqueeze(1)).squeeze(1)
        batch_inputs_unf = batch_inputs
        step_unfinished = batch_inputs_unf != tgt_end_idx
        batch_inputs_unf = batch_inputs_unf[step_unfinished]
        batch_state_unf = (batch_state_unf[0][:, step_unfinished, :, :],
                           batch_state_unf[1][:, step_unfinished, :, :])
        batch_io_embeddings_unf = batch_io_embeddings_unf[step_unfinished]
        batch_list_inputs_unf = list_fun(batch_inputs_unf)
        if batch_grammar_state_unf is not None:
            batch_grammar_state_unf_new = []
            for i, item in enumerate(batch_grammar_state_unf):
                if step_unfinished[i]:
                    batch_grammar_state_unf_new.append(copy.copy(item))
            batch_grammar_state_unf = batch_grammar_state_unf_new

        unfinished_index = unfinished_index[step_unfinished]
        if step_unfinished.sum() == 0:
            break
    #print('main', seq)
    # arm_metric_value = np.zeros(batch_size)
    # for i in range(batch_size):
    #     scorer = envs[i]
    #     arm_metric_value[i] = scorer.step_reward(pre_scoring(seq[i], tgt_end_idx), True)
    # print('main', np.mean(arm_metric_value))
    loss = loss / batch_size
    loss = loss - entropy_reg * entropy
    if backward_flag:
        loss.backward()
    return loss.item(), np.mean(pseudo_num_list)


def arsm_f_delta_fun_batch_torch(logits, pi, envs, pre_seq, step, model, batch_state, batch_grammar_state,
                                 unfinished, batch_io_embeddings, tgt_end_idx, max_len, arm_sample, seq_log, logits_factor):
    batch_size, vocab_size = logits.size()
    arm_metric_matrix = np.ones([batch_size, vocab_size, vocab_size]) * -1
    A_cat = torch.min(torch.log(pi) - logits, 1)[1].long()
    pseudo_actions = pseudo_action_batch(pi.cpu().numpy(), logits.data.cpu().numpy())
    pseudo_actions = torch.from_numpy(np.reshape(pseudo_actions, [-1, vocab_size * vocab_size])).cuda().long()
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
    #print('pseudo numbers', np.mean(arm_pseudo_index))

    if np.sum(arm_pseudo_index) == batch_size:
        return torch.zeros(batch_size, vocab_size).float().cuda(), False, 0
    seqs_arm = pre_seq[arm_index_2, :]
    seq_log_arm = seq_log[arm_index_2]
    batch_io_embeddings_arm = batch_io_embeddings[arm_index_2]
    batch_inputs_arm = torch.from_numpy(arm_pseudo_action_set).long().cuda()
    seq_log_arm += logits[arm_index_2, :].gather(1, batch_inputs_arm.unsqueeze(1)).squeeze(1)
    unfinished_arm = (batch_inputs_arm != tgt_end_idx)
    batch_state_arm = (batch_state[0][:, arm_index_2, :, :], batch_state[1][:, arm_index_2, :, :])
    batch_grammar_state_arm = []
    if batch_grammar_state is not None:
        for i in range(len(arm_index_2)):
            batch_grammar_state_arm.append(copy.copy(batch_grammar_state[int(arm_index_2[i])]))

    ## initial unfinished:
    batch_inputs_arm_unf = batch_inputs_arm[unfinished_arm]
    batch_list_inputs_arm_unf = list_fun(batch_inputs_arm_unf)
    batch_state_arm_unf = (batch_state_arm[0][:, unfinished_arm, :, :], batch_state_arm[1][:, unfinished_arm, :, :])
    if batch_grammar_state_arm is not None:
        batch_grammar_state_arm_unf = []
        for i, item in enumerate(batch_grammar_state_arm):
            if unfinished_arm[i]:
                batch_grammar_state_arm_unf.append(copy.copy(item))
    batch_io_embeddings_arm_unf = batch_io_embeddings_arm[unfinished_arm]
    unfinished_index = torch.arange(len(arm_index_2)).long().cuda()[unfinished_arm]

    for t in range(step + 1, max_len + 1):
        seqs_arm[unfinished_index, t] = batch_inputs_arm_unf
        if t == max_len:
            break
        #TODO: only run unfinished
        dec_outs, batch_state_arm_unf, batch_grammar_state_arm_unf, _ = \
            model.decoder.forward(batch_inputs_arm_unf.unsqueeze(1), batch_io_embeddings_arm_unf,
                          batch_list_inputs_arm_unf, batch_state_arm_unf,
                          batch_grammar_state_arm_unf)
        #batch_grammar_state_arm = [copy.copy(item) for item in batch_grammar_state_arm]
        dec_outs = dec_outs.squeeze(1)
        logprobs = F.log_softmax(dec_outs, dim=1)
        if arm_sample == 'greedy':
            batch_inputs_arm = torch.max(logprobs.data, 1)[1]
        else:
            batch_inputs_arm = torch.multinomial(torch.exp(logprobs.data).cpu(), 1).cuda().squeeze(1)
        batch_inputs_arm = batch_inputs_arm.long()
        seq_log_arm[unfinished_index] = seq_log_arm[unfinished_index] + \
                                        logprobs.data.gather(1, batch_inputs_arm.unsqueeze(1)).squeeze(1)
        batch_inputs_arm_unf = batch_inputs_arm

        step_unfinished = batch_inputs_arm_unf != tgt_end_idx
        batch_inputs_arm_unf = batch_inputs_arm_unf[step_unfinished]
        batch_state_arm_unf = (batch_state_arm_unf[0][:, step_unfinished, :, :],
                               batch_state_arm_unf[1][:, step_unfinished, :, :])
        batch_io_embeddings_arm_unf = batch_io_embeddings_arm_unf[step_unfinished]
        batch_list_inputs_arm_unf = list_fun(batch_inputs_arm_unf)
        if batch_grammar_state_arm_unf is not None:
            batch_grammar_state_arm_unf_new = []
            for i, item in enumerate(batch_grammar_state_arm_unf):
                if step_unfinished[i]:
                    batch_grammar_state_arm_unf_new.append(copy.copy(item))
        batch_grammar_state_arm_unf = batch_grammar_state_arm_unf_new

        unfinished_index = unfinished_index[step_unfinished]
        if step_unfinished.sum() == 0:
            break
    #print(seqs_arm)

    ## evaluate reward
    arm_metric_value = np.zeros(seqs_arm.size(0))
    for i in range(seqs_arm.size(0)):
        scorer = envs[int(arm_index_2[i])]
        arm_metric_value[i] = scorer.step_reward(pre_scoring(seqs_arm[i], tgt_end_idx), True)
    #print('pseudo reward', np.mean(arm_metric_value))
    # arm_metric_value += logits_factor * seq_log_arm.data.cpu().numpy()
    arm_index = np.array(arm_index)
    arm_index += np.repeat(np.expand_dims(np.concatenate([[0], np.cumsum(arm_pseudo_counts)[0:-1]]), 1),
                           vocab_size * vocab_size, 1)
    arm_index = np.reshape(arm_index, [-1])
    arm_pseudo_index = np.array(arm_pseudo_index)
    arm_metric_matrix[arm_pseudo_index > 1, :, :] = np.reshape(arm_metric_value[arm_index],
                                                               [-1, vocab_size, vocab_size])
    arm_metric_matrix_cuda = torch.from_numpy(arm_metric_matrix).float().cuda()
    f_delta = ((arm_metric_matrix_cuda - arm_metric_matrix_cuda.mean(1).unsqueeze(1).repeat(1, vocab_size, 1))
               * (1.0 / vocab_size - pi.unsqueeze(1).repeat(1, vocab_size, 1))).sum(2)
    return f_delta, True, np.mean(arm_pseudo_index)


def get_ar_loss(model, inp_grids, out_grids, envs, tgt_start_idx, tgt_end_idx, max_len, arm_sample='greedy', decay_factor=1, logits_factor=0):
    use_cuda = inp_grids.is_cuda
    tt = torch.cuda if use_cuda else torch
    io_embeddings = model.encoder(inp_grids, out_grids)
    batch_size, nb_ios, io_emb_size = io_embeddings.size()
    batch_state = None
    batch_grammar_state = None
    batch_inputs = inp_grids.new_ones(batch_size, dtype=torch.long) * tgt_start_idx
    batch_list_inputs = [[tgt_start_idx]] * batch_size
    batch_io_embeddings = io_embeddings
    # TODO: start and end should all be included, and end with end, what if sampled start index?

    seq = inp_grids.new_ones(batch_size, max_len + 1, dtype=torch.long) * tgt_end_idx
    seq_log = inp_grids.new_zeros(batch_size, max_len)
    loss = inp_grids.new_zeros([])
    unfinished = inp_grids.new_ones(batch_size, dtype=torch.int8)
    mask_sum = 0
    ## initialize unfinished:
    batch_inputs_unf = batch_inputs
    batch_list_inputs_unf = batch_list_inputs
    batch_state_unf = batch_state
    batch_grammar_state_unf = batch_grammar_state
    batch_io_embeddings_unf = batch_io_embeddings
    unfinished_index = torch.arange(batch_size).long().cuda()
    pi_list = []
    logprobs_list = []
    unfinished_index_list = []
    for t in range(max_len + 1):  # TODO: reduce one iteration
        seq[unfinished_index, t] = batch_inputs_unf
        if t == max_len:
            break
        dec_outs, batch_state_unf, \
        batch_grammar_state_unf, _ = model.decoder.forward(batch_inputs_unf.unsqueeze(1),
                                                           batch_io_embeddings_unf,
                                                           batch_list_inputs_unf,
                                                           batch_state_unf,
                                                           batch_grammar_state_unf)
        dec_outs = dec_outs.squeeze(1)  # batch * vocab
        vocab_size = dec_outs.size(1)
        logprobs = F.log_softmax(dec_outs, dim=1)
        unfinished = inp_grids.new_ones(dec_outs.size(0), dtype=torch.int8)
        mask = unfinished.float()
        mask_sum += torch.sum(mask)
        pi = torch.from_numpy(np.random.dirichlet(np.ones(vocab_size), dec_outs.size(0))).float().cuda()
        pi_list.append(pi)
        unfinished_index_list.append(unfinished_index)
        logprobs_list.append(dec_outs)
        batch_inputs = torch.min(torch.log(pi) - logprobs, 1)[1]
        batch_inputs = batch_inputs.long()
        seq_log[unfinished_index, t] = logprobs.gather(1, batch_inputs.unsqueeze(1)).squeeze(1)
        batch_inputs_unf = batch_inputs
        step_unfinished = batch_inputs_unf != tgt_end_idx
        batch_inputs_unf = batch_inputs_unf[step_unfinished]
        batch_state_unf = (batch_state_unf[0][:, step_unfinished, :, :],
                           batch_state_unf[1][:, step_unfinished, :, :])
        batch_io_embeddings_unf = batch_io_embeddings_unf[step_unfinished]
        batch_list_inputs_unf = list_fun(batch_inputs_unf)
        batch_grammar_state_unf_new = []
        for i, item in enumerate(batch_grammar_state_unf):
            if step_unfinished[i]:
                batch_grammar_state_unf_new.append(copy.copy(item))
        batch_grammar_state_unf = batch_grammar_state_unf_new

        unfinished_index = unfinished_index[step_unfinished]
        if step_unfinished.sum() == 0:
            break
    batch_size = inp_grids.size(0)
    arm_metric_value = np.zeros(batch_size)
    for i in range(batch_size):
        scorer = envs[i]
        arm_metric_value[i] = scorer.step_reward(pre_scoring(seq[i], tgt_end_idx), True)
    reward = arm_metric_value  # - np.mean(arm_metric_value)
    for i in range(len(pi_list)):
        f_delta = torch.from_numpy(np.repeat(np.expand_dims(
            arm_metric_value, 1), vocab_size, 1)).float().cuda()[unfinished_index_list[i], :] * (
                    1.0 - vocab_size * pi_list[i])
        mask = seq[unfinished_index_list[i], i] != tgt_end_idx
        print(mask)
        f_delta = (f_delta.transpose(0, 1) * mask.float()).transpose(0, 1)
        #loss -= torch.sum(f_delta.detach() * logprobs_list[i])
        tmp = f_delta.detach() * logprobs_list[i] * (logprobs_list[i] != -float('inf')).type_as(logprobs_list[i])
        loss -= tmp[1 - torch.isnan(tmp)].sum()

    loss = loss / batch_size
    loss.backward()

    return loss.item()

def get_adaptive_arm_loss(model, inp_grids, out_grids, envs, tgt_start_idx, tgt_end_idx, max_len, arm_sample='greedy', decay_factor=1, logits_factor=0):
    use_cuda = inp_grids.is_cuda
    tt = torch.cuda if use_cuda else torch
    io_embeddings = model.encoder(inp_grids, out_grids)
    batch_size, nb_ios, io_emb_size = io_embeddings.size()
    batch_state = None
    batch_grammar_state = None
    batch_inputs = inp_grids.new_ones(batch_size, dtype=torch.long) * tgt_start_idx
    batch_list_inputs = [[tgt_start_idx]]*batch_size
    batch_io_embeddings = io_embeddings
    #TODO: start and end should all be included, and end with end, what if sampled start index?

    seq = inp_grids.new_ones(batch_size, max_len + 1, dtype=torch.long) * tgt_end_idx
    seq_log = inp_grids.new_zeros(batch_size)
    loss = inp_grids.new_zeros([])
    unfinished = inp_grids.new_ones(batch_size, dtype=torch.int8)
    mask_sum = 0

    ## initialize unfinished:
    batch_inputs_unf = batch_inputs
    batch_list_inputs_unf = batch_list_inputs
    batch_state_unf = batch_state
    batch_grammar_state_unf = batch_grammar_state
    batch_io_embeddings_unf = batch_io_embeddings
    unfinished_index = torch.arange(batch_size).long().cuda()
    for t in range(max_len + 1): #TODO: reduce one iteration
        seq[unfinished_index, t] = batch_inputs_unf
        if t == max_len:
            break
        dec_outs, batch_state_unf, \
        batch_grammar_state_unf, _ = model.decoder.forward(batch_inputs_unf.unsqueeze(1),
                                                       batch_io_embeddings_unf,
                                                       batch_list_inputs_unf,
                                                       batch_state_unf,
                                                       batch_grammar_state_unf)
        dec_outs = dec_outs.squeeze(1) # batch * vocab
        vocab_size = dec_outs.size(1)
        logprobs = F.log_softmax(dec_outs, dim=1)
        unfinished = inp_grids.new_ones(dec_outs.size(0), dtype=torch.int8)
        mask = unfinished.float()

        loss_tmp = adaptive_logits_step(dec_outs, envs, seq, t, model, copy.copy(batch_state_unf),
                                               copy.copy(batch_grammar_state_unf),
                                               batch_io_embeddings_unf,
                                               tgt_end_idx, max_len, arm_sample, seq_log[unfinished_index])
        loss += loss_tmp
        if arm_sample == 'greedy':
            batch_inputs = torch.max(logprobs.data, 1)[1]
        else:
            batch_inputs = torch.multinomial(torch.exp(logprobs.data).cpu(), 1).cuda().squeeze(1)
        batch_inputs = batch_inputs.long()
        seq_log[unfinished_index] = seq_log[unfinished_index] + logprobs.data.gather(1, batch_inputs.unsqueeze(1)).squeeze(1)
        batch_inputs_unf = batch_inputs
        step_unfinished = batch_inputs_unf != tgt_end_idx
        batch_inputs_unf = batch_inputs_unf[step_unfinished]
        batch_state_unf = (batch_state_unf[0][:, step_unfinished, :, :],
                           batch_state_unf[1][:, step_unfinished, :, :])
        batch_io_embeddings_unf = batch_io_embeddings_unf[step_unfinished]
        batch_list_inputs_unf = list_fun(batch_inputs_unf)
        batch_grammar_state_unf_new = []
        for i, item in enumerate(batch_grammar_state_unf):
            if step_unfinished[i]:
                batch_grammar_state_unf_new.append(copy.copy(item))
        batch_grammar_state_unf = batch_grammar_state_unf_new

        unfinished_index = unfinished_index[step_unfinished]
        if step_unfinished.sum() == 0:
            break
    #print('main', seq)
    # arm_metric_value = np.zeros(batch_size)
    # for i in range(batch_size):
    #     scorer = envs[i]
    #     arm_metric_value[i] = scorer.step_reward(pre_scoring(seq[i], tgt_end_idx), True)
    # print('main', np.mean(arm_metric_value))
    loss = loss / batch_size
    loss.backward()
    return loss.item()


def adaptive_arsm_step(logits, envs, pre_seq, step, model, batch_state, batch_grammar_state,
                       batch_io_embeddings, tgt_end_idx, max_len, arm_sample, seq_log):
    batch_size, vocab_size = logits.size()
    pseudo_actions_list, pi_list, vocab_list = pseudo_action_batch_uninf(logits.data.cpu().numpy())
    arm_pseudo_action_set = []
    arm_index = []
    arm_index_2 = np.zeros(0)
    arm_pseudo_counts = []
    counts_per_sample_list = []
    arm_pseudo_index = []

    for i in range(batch_size): #TODO: what if vocab is zero?
        set_per_sample, index_per_sample, counts_per_sample = np.unique(pseudo_actions_list[i].cpu().numpy(), return_inverse=True, return_counts=True)
        pseudo_count = len(set_per_sample)
        arm_pseudo_index.append(pseudo_count)
        if pseudo_count > 1:
            arm_pseudo_counts.append(pseudo_count)
            arm_pseudo_action_set = np.concatenate([arm_pseudo_action_set, set_per_sample], axis=0)
            arm_index.append(index_per_sample)
            arm_index_2 = np.concatenate([arm_index_2, (np.ones(pseudo_count) * i)], axis=0)
            counts_per_sample_list.append(counts_per_sample)
    #print('pseudo numbers', np.mean(arm_pseudo_index))
    #print(pseudo_actions_list, pi_list, vocab_list, arm_pseudo_action_set, arm_index_2)

    if np.sum(arm_pseudo_counts) == 0:
        return 0
    seqs_arm = pre_seq[arm_index_2, :]
    seq_log_arm = seq_log[arm_index_2]
    batch_io_embeddings_arm = batch_io_embeddings[arm_index_2]
    batch_inputs_arm = torch.from_numpy(arm_pseudo_action_set).long().cuda()
    seq_log_arm += logits[arm_index_2, :].gather(1, batch_inputs_arm.unsqueeze(1)).squeeze(1)
    unfinished_arm = (batch_inputs_arm != tgt_end_idx)
    batch_state_arm = (batch_state[0][:, arm_index_2, :, :], batch_state[1][:, arm_index_2, :, :])
    batch_grammar_state_arm = []
    for i in range(len(arm_index_2)):
        batch_grammar_state_arm.append(copy.copy(batch_grammar_state[int(arm_index_2[i])]))

    ## initial unfinished:
    batch_inputs_arm_unf = batch_inputs_arm[unfinished_arm]
    batch_list_inputs_arm_unf = list_fun(batch_inputs_arm_unf)
    batch_state_arm_unf = (batch_state_arm[0][:, unfinished_arm, :, :], batch_state_arm[1][:, unfinished_arm, :, :])
    batch_grammar_state_arm_unf = []
    for i, item in enumerate(batch_grammar_state_arm):
        if unfinished_arm[i]:
            batch_grammar_state_arm_unf.append(copy.copy(item))
    batch_io_embeddings_arm_unf = batch_io_embeddings_arm[unfinished_arm]
    unfinished_index = torch.arange(len(arm_index_2)).long().cuda()[unfinished_arm]

    for t in range(step + 1, max_len + 1):
        seqs_arm[unfinished_index, t] = batch_inputs_arm_unf
        if t == max_len:
            break
        #TODO: only run unfinished
        dec_outs, batch_state_arm_unf, batch_grammar_state_arm_unf, _ = \
            model.decoder.forward(batch_inputs_arm_unf.unsqueeze(1), batch_io_embeddings_arm_unf,
                          batch_list_inputs_arm_unf, batch_state_arm_unf,
                          batch_grammar_state_arm_unf)
        #batch_grammar_state_arm = [copy.copy(item) for item in batch_grammar_state_arm]
        dec_outs = dec_outs.squeeze(1)
        logprobs = F.log_softmax(dec_outs, dim=1)
        if arm_sample == 'greedy':
            batch_inputs_arm = torch.max(logprobs.data, 1)[1]
        else:
            batch_inputs_arm = torch.multinomial(torch.exp(logprobs.data).cpu(), 1).cuda().squeeze(1)
        batch_inputs_arm = batch_inputs_arm.long()
        seq_log_arm[unfinished_index] = seq_log_arm[unfinished_index] + \
                                        logprobs.data.gather(1, batch_inputs_arm.unsqueeze(1)).squeeze(1)
        batch_inputs_arm_unf = batch_inputs_arm

        step_unfinished = batch_inputs_arm_unf != tgt_end_idx
        batch_inputs_arm_unf = batch_inputs_arm_unf[step_unfinished]
        batch_state_arm_unf = (batch_state_arm_unf[0][:, step_unfinished, :, :],
                               batch_state_arm_unf[1][:, step_unfinished, :, :])
        batch_io_embeddings_arm_unf = batch_io_embeddings_arm_unf[step_unfinished]
        batch_list_inputs_arm_unf = list_fun(batch_inputs_arm_unf)
        batch_grammar_state_arm_unf_new = []
        for i, item in enumerate(batch_grammar_state_arm_unf):
            if step_unfinished[i]:
                batch_grammar_state_arm_unf_new.append(copy.copy(item))
        batch_grammar_state_arm_unf = batch_grammar_state_arm_unf_new

        unfinished_index = unfinished_index[step_unfinished]
        if step_unfinished.sum() == 0:
            break
    #print(seqs_arm)

    ## evaluate reward
    arm_metric_value = np.zeros(seqs_arm.size(0))
    for i in range(seqs_arm.size(0)):
        scorer = envs[int(arm_index_2[i])]
        arm_metric_value[i] = scorer.step_reward(pre_scoring(seqs_arm[i], tgt_end_idx), True)
    #print('pseudo reward', np.mean(arm_metric_value))
    # arm_metric_value += logits_factor * seq_log_arm.data.cpu().numpy()
    loss = 0
    start = 0
    start_index = 0
    for i in range(batch_size):
        if arm_pseudo_index[i] > 1:
            vocab = vocab_list[i]
            reward = arm_metric_value[start:(start+arm_pseudo_index[i])]
            arm_metric_matrix = np.reshape(reward[arm_index[start_index]], [vocab, vocab])
            arm_metric_matrix_cuda = torch.from_numpy(arm_metric_matrix).float().cuda()
            f_delta = arm_metric_matrix_cuda - arm_metric_matrix_cuda.mean(0).unsqueeze(0).repeat(vocab, 1)
            f_delta = (f_delta * (1.0 / vocab - pi_list[i].repeat(vocab, 1))).sum(1)

            # arm_metric_matrix = np.reshape(reward[arm_index[start_index]], [vocab])
            # arm_metric_matrix_cuda = torch.from_numpy(arm_metric_matrix).float().cuda()
            # f_delta = arm_metric_matrix_cuda - arm_metric_matrix_cuda.mean(0).unsqueeze(0).repeat(vocab)
            # f_delta = (f_delta * (1 - pi_list[i][0]))


            index = logits[i, :] != -float('inf')
            loss -= (f_delta * logits[i, index]).sum()
            start = start + arm_pseudo_index[i]
            start_index = start_index + 1
    return loss




def adaptive_logits_step(logits, envs, pre_seq, step, model, batch_state, batch_grammar_state,
                       batch_io_embeddings, tgt_end_idx, max_len, arm_sample, seq_log):
    batch_size, vocab_size = logits.size()
    pseudo_actions_list, pi_list, vocab_list = pseudo_action_batch_uninf(logits.data.cpu().numpy())
    arm_pseudo_action_set = []
    arm_index = []
    arm_index_2 = np.zeros(0)
    arm_pseudo_counts = []
    counts_per_sample_list = []
    arm_pseudo_index = []

    for i in range(batch_size): #TODO: what if vocab is zero?
        set_per_sample, index_per_sample, counts_per_sample = np.unique(pseudo_actions_list[i].cpu().numpy(), return_inverse=True, return_counts=True)
        pseudo_count = len(set_per_sample)
        arm_pseudo_index.append(pseudo_count)
        if pseudo_count > 1:
            arm_pseudo_counts.append(pseudo_count)
            arm_pseudo_action_set = np.concatenate([arm_pseudo_action_set, set_per_sample], axis=0)
            arm_index.append(index_per_sample)
            arm_index_2 = np.concatenate([arm_index_2, (np.ones(pseudo_count) * i)], axis=0)
            counts_per_sample_list.append(counts_per_sample)
    #print('pseudo numbers', np.mean(arm_pseudo_index))
    #print(pseudo_actions_list, pi_list, vocab_list, arm_pseudo_action_set, arm_index_2)

    if np.sum(arm_pseudo_counts) == 0:
        return 0
    seqs_arm = pre_seq[arm_index_2, :]
    seq_log_arm = seq_log[arm_index_2]
    batch_io_embeddings_arm = batch_io_embeddings[arm_index_2]
    batch_inputs_arm = torch.from_numpy(arm_pseudo_action_set).long().cuda()
    seq_log_arm += logits[arm_index_2, :].gather(1, batch_inputs_arm.unsqueeze(1)).squeeze(1)
    unfinished_arm = (batch_inputs_arm != tgt_end_idx)
    batch_state_arm = (batch_state[0][:, arm_index_2, :, :], batch_state[1][:, arm_index_2, :, :])
    batch_grammar_state_arm = []
    for i in range(len(arm_index_2)):
        batch_grammar_state_arm.append(copy.copy(batch_grammar_state[int(arm_index_2[i])]))

    ## initial unfinished:
    batch_inputs_arm_unf = batch_inputs_arm[unfinished_arm]
    batch_list_inputs_arm_unf = list_fun(batch_inputs_arm_unf)
    batch_state_arm_unf = (batch_state_arm[0][:, unfinished_arm, :, :], batch_state_arm[1][:, unfinished_arm, :, :])
    batch_grammar_state_arm_unf = []
    for i, item in enumerate(batch_grammar_state_arm):
        if unfinished_arm[i]:
            batch_grammar_state_arm_unf.append(copy.copy(item))
    batch_io_embeddings_arm_unf = batch_io_embeddings_arm[unfinished_arm]
    unfinished_index = torch.arange(len(arm_index_2)).long().cuda()[unfinished_arm]

    for t in range(step + 1, max_len + 1):
        seqs_arm[unfinished_index, t] = batch_inputs_arm_unf
        if t == max_len:
            break
        #TODO: only run unfinished
        dec_outs, batch_state_arm_unf, batch_grammar_state_arm_unf, _ = \
            model.decoder.forward(batch_inputs_arm_unf.unsqueeze(1), batch_io_embeddings_arm_unf,
                          batch_list_inputs_arm_unf, batch_state_arm_unf,
                          batch_grammar_state_arm_unf)
        #batch_grammar_state_arm = [copy.copy(item) for item in batch_grammar_state_arm]
        dec_outs = dec_outs.squeeze(1)
        logprobs = F.log_softmax(dec_outs, dim=1)
        if arm_sample == 'greedy':
            batch_inputs_arm = torch.max(logprobs.data, 1)[1]
        else:
            batch_inputs_arm = torch.multinomial(torch.exp(logprobs.data).cpu(), 1).cuda().squeeze(1)
        batch_inputs_arm = batch_inputs_arm.long()
        seq_log_arm[unfinished_index] = seq_log_arm[unfinished_index] + \
                                        logprobs.data.gather(1, batch_inputs_arm.unsqueeze(1)).squeeze(1)
        batch_inputs_arm_unf = batch_inputs_arm

        step_unfinished = batch_inputs_arm_unf != tgt_end_idx
        batch_inputs_arm_unf = batch_inputs_arm_unf[step_unfinished]
        batch_state_arm_unf = (batch_state_arm_unf[0][:, step_unfinished, :, :],
                               batch_state_arm_unf[1][:, step_unfinished, :, :])
        batch_io_embeddings_arm_unf = batch_io_embeddings_arm_unf[step_unfinished]
        batch_list_inputs_arm_unf = list_fun(batch_inputs_arm_unf)
        batch_grammar_state_arm_unf_new = []
        for i, item in enumerate(batch_grammar_state_arm_unf):
            if step_unfinished[i]:
                batch_grammar_state_arm_unf_new.append(copy.copy(item))
        batch_grammar_state_arm_unf = batch_grammar_state_arm_unf_new

        unfinished_index = unfinished_index[step_unfinished]
        if step_unfinished.sum() == 0:
            break
    #print(seqs_arm)

    ## evaluate reward
    arm_metric_value = np.zeros(seqs_arm.size(0))
    for i in range(seqs_arm.size(0)):
        scorer = envs[int(arm_index_2[i])]
        arm_metric_value[i] = scorer.step_reward(pre_scoring(seqs_arm[i], tgt_end_idx), True)
    #print('pseudo reward', np.mean(arm_metric_value))
    # arm_metric_value += logits_factor * seq_log_arm.data.cpu().numpy()
    loss = 0
    start = 0
    start_index = 0

    for i in range(batch_size):
        if arm_pseudo_index[i] > 1:
            vocab = vocab_list[i]
            reward = arm_metric_value[start:(start+arm_pseudo_index[i])]
            action = arm_pseudo_action_set[start:(start+arm_pseudo_index[i])]
            loss -= (F.softmax(logits[i, action], 0) * torch.from_numpy(reward).cuda().float()).mean()
            start = start + arm_pseudo_index[i]
            start_index = start_index + 1
    return loss



def batch_rolls_reinforce(rolls):
    for roll in rolls:
        for var, grad in roll.yield_var_and_grad():
            if grad is None:
                assert var.requires_grad is False
            else:
                yield var, grad

def pre_scoring(seq, end_idx):
    result = []
    for i in seq[1:]:
        result.append(i.item())
        if i == end_idx:
            break
    return result



def list_fun(tensor):
    list_tensor = tensor.cpu().numpy().tolist()
    result = []
    for item in list_tensor:
        result.append([item])
    return result


def pseudo_action_batch_uninf(phi_batch):
    batch, vocab_0 = np.shape(phi_batch)
    pi_list = []
    pseudo_actions_list = []
    vocab_list = []
    for i in range(batch):
        phi = phi_batch[i, :]
        inf_index = phi != -float('inf')
        phi_effect = phi[inf_index] #TODO: check
        if np.shape(phi_effect)[0] == 0:
            pseudo_actions_list.append([])
            pi_list.append([])
            vocab_list.append(0)
        else:
            vocab = np.shape(phi_effect)[0]
            pi = np.random.dirichlet(np.ones(vocab))
            pseudo_actions = np.reshape(pseudo_action_swap_matrix(pi, phi_effect), [vocab*vocab])
            # pseudo_actions = np.reshape(pseudo_action_swap_matrix(pi, phi_effect)[0,:], [vocab])

            pseudo_actions = np.arange(vocab_0)[inf_index][pseudo_actions]
            pseudo_actions_list.append(torch.from_numpy(pseudo_actions).cuda().long())
            pi_list.append(torch.from_numpy(pi).cuda().float())
            vocab_list.append(vocab)
    return pseudo_actions_list, pi_list, vocab_list


def pseudo_action_batch(pi_batch, phi_batch):
    batch, vocab = np.shape(pi_batch)
    result = np.zeros(shape=[batch, vocab, vocab])
    for i in range(batch):
        result[i, :, :] = pseudo_action_swap_matrix(pi_batch[i, :], phi_batch[i, :])

    # result = np.zeros(shape=[batch, vocab])
    # for i in range(batch):
    #     result[i, :, :] = pseudo_action_swap_matrix(pi_batch[i, :], phi_batch[i, :])[0, :]
    return result

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

def reward_transform(reward_list, multiplier):
    result = []
    for i in reward_list:
        if i > 0:
            result.append(multiplier * i)
        else:
            result.append(i)
    return np.array(result)
