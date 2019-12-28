from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
import eval_utils_binary
import misc.utils as utils
from misc.rewards import get_self_critical_reward, get_reward, get_arm_loss, get_mct_loss, get_ar_loss, get_rf_loss
from models.FCModel_binary import init_scorer
#from misc.rewards import init_scorer
from models.CriticModel import CriticModel
from models.AttCriticModel import AttCriticModel, critic_loss_fun, target_critic_loss_fun, target_critic_loss_fun_mask
try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)


def train(opt):
    # opt.use_att = utils.if_use_att(opt.caption_model)
    opt.use_att = True
    if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5

    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    print(opt.checkpoint_path)
    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')) as f:
                histories = cPickle.load(f)

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    critic_loss_history = histories.get('critic_loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})
    variance_history = histories.get('variance_history', {})
    time_history = histories.get('time_history', {})
    pseudo_num_history = histories.get('pseudo_num_history', {})
    pseudo_num_depth_history = histories.get('pseudo_num_depth_history', {})
    pseudo_num_length_history = histories.get('pseudo_num_length_history', {})
    pseudo_num_batch_history = histories.get('pseudo_num_batch_history', {})
    reward_batch_history = histories.get('reward_batch_history', {})
    entropy_batch_history = histories.get('entropy_batch_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    model = models.setup(opt).cuda()
    dp_model = model




    ######################### Actor-critic Training #####################################################################

    update_lr_flag = True
    # Assure in training mode
    dp_model.train()
    #TODO: change this to a flag
    crit = utils.LanguageModelCriterion_binary()
    rl_crit = utils.RewardCriterion_binary()

    optimizer = utils.build_optimizer(model.parameters(), opt)
    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    first_order = 0
    second_order = 0
    while True:
        if update_lr_flag:
                # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            utils.set_lr(optimizer, opt.current_lr)
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                sc_flag = False

            update_lr_flag = False

        # Load data from train split (0)
        data = loader.get_batch('train')
        if data['bounds']['it_pos_now'] > 10000:
            loader.reset_iterator('train')
            continue
        dp_model.train()

        torch.cuda.synchronize()
        start = time.time()
        gen_result = None
        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp
        optimizer.zero_grad()
        if not sc_flag:
            loss = crit(dp_model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:], dp_model.depth,
                        dp_model.vocab2code, dp_model.phi_list, dp_model.cluster_size)
        else:
            if opt.rl_type == 'sc':
                gen_result, sample_logprobs = dp_model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
                reward = get_self_critical_reward(dp_model, fc_feats, att_feats, att_masks, data, gen_result, opt)
                loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda(), dp_model.depth)
            elif opt.rl_type == 'reinforce':
                gen_result, sample_logprobs = dp_model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
                reward = get_reward(data, gen_result, opt)
                loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda(), dp_model.depth)
            elif opt.rl_type == 'arm':
                loss, pseudo_num, pseudo_num_depth, pseudo_num_length, pseudo_num_batch, reward_batch, entropy_batch = dp_model.get_arm_loss_binary_fast(fc_feats, att_feats, att_masks, opt, data, loader)
                #print(loss)
                reward = np.zeros([2,2])
            elif opt.rl_type == 'rf4':
                loss,_,_,_ = get_rf_loss(dp_model, fc_feats, att_feats, att_masks, data, opt, loader)
                # print(loss)
                reward = np.zeros([2, 2])
            elif opt.rl_type == 'ar':
                loss = get_ar_loss(dp_model, fc_feats, att_feats, att_masks, data, opt, loader)
                reward = np.zeros([2,2])
            elif opt.rl_type =='mct_baseline':
                opt.rf_demean = 0
                gen_result, sample_logprobs, probs, mct_baseline = get_mct_loss(dp_model, fc_feats, att_feats, att_masks, data,
                                                                         opt, loader)
                reward = get_reward(data, gen_result, opt)
                reward_cuda = torch.from_numpy(reward).float().cuda()
                mct_baseline[mct_baseline < 0] = reward_cuda[mct_baseline < 0]
                if opt.arm_step_sample == 'greedy':
                    sample_logprobs = sample_logprobs * probs
                loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda() - mct_baseline)
            elif opt.rl_type == 'arsm_baseline':
                opt.arm_as_baseline = 1
                opt.rf_demean = 0
                gen_result, sample_logprobs, probs, arm_baseline = get_arm_loss(dp_model, fc_feats, att_feats, att_masks, data, opt, loader)
                reward = get_reward(data, gen_result, opt)
                reward_cuda = torch.from_numpy(reward).float().cuda()
                arm_baseline[arm_baseline < 0] = reward_cuda[arm_baseline < 0]
                if opt.arm_step_sample == 'greedy' and False:
                    sample_logprobs = sample_logprobs * probs
                loss = rl_crit(sample_logprobs, gen_result.data, reward_cuda - arm_baseline)
            elif opt.rl_type == 'ars_indicator':
                opt.arm_as_baseline = 1
                opt.rf_demean = 0
                gen_result, sample_logprobs, probs, arm_baseline = get_arm_loss(dp_model, fc_feats, att_feats, att_masks, data, opt, loader)
                reward = get_self_critical_reward(dp_model, fc_feats, att_feats, att_masks, data, gen_result, opt)
                reward_cuda = torch.from_numpy(reward).float().cuda()
                loss = rl_crit(sample_logprobs, gen_result.data, reward_cuda * arm_baseline)
        if opt.mle_weights != 0:
            loss += opt.mle_weights * crit(dp_model(fc_feats, att_feats, labels, att_masks), labels[:, 1:], masks[:, 1:])
        #TODO make sure all sampling replaced by greedy for critic
        #### update the actor
        loss.backward()
        # with open(os.path.join(opt.checkpoint_path, 'embeddings.pkl'), 'wb') as f:
        #     cPickle.dump(list(dp_model.embed.parameters())[0].data.cpu().numpy(), f)
        ## compute variance
        gradient = torch.zeros([0]).cuda()
        for i in model.parameters():
            gradient = torch.cat((gradient, i.grad.view(-1)), 0)
        first_order = 0.999 * first_order + 0.001 * gradient
        second_order = 0.999 * second_order + 0.001 * gradient.pow(2)
        # print(torch.max(torch.abs(gradient)))
        variance = torch.mean(torch.abs(second_order - first_order.pow(2))).item()
        if opt.rl_type != 'arsm' or not sc_flag:
            utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        # ### update the critic

        train_loss = loss.item()
        torch.cuda.synchronize()
        end = time.time()
        if (iteration % opt.losses_log_every == 0):
            if not sc_flag:
                print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, end - start))
                print(opt.checkpoint_path)
            else:
                print("iter {} (epoch {}), avg_reward = {:.3f}, variance = {:g}, time/batch = {:.3f}, pseudo num = {:.3f}, " \
                      .format(iteration, epoch, np.mean(reward[:, 0]), variance, end - start, pseudo_num))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
            add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
            if sc_flag:
                add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward), iteration)
                add_summary_value(tb_summary_writer, 'variance', variance, iteration)

            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward)
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob
            variance_history[iteration] = variance
            time_history[iteration] = end - start
            pseudo_num_history[iteration] = pseudo_num.item()
            pseudo_num_length_history[iteration] = pseudo_num_length.data.cpu().numpy()
            pseudo_num_depth_history[iteration] = pseudo_num_depth.data.cpu().numpy()
            pseudo_num_batch_history[iteration] = pseudo_num_batch.data.cpu().numpy()
            reward_batch_history[iteration] = reward_batch
            entropy_batch_history[iteration] = entropy_batch.data.cpu().numpy()


        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils_binary.eval_split(dp_model, crit, loader, eval_kwargs)
            print('1')
            # Write validation result into summary
            add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
            if lang_stats is not None:
                for k,v in lang_stats.items():
                    add_summary_value(tb_summary_writer, k, v, iteration)
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss
            print('2')
            best_flag = False
            if True: # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                if not os.path.isdir(opt.checkpoint_path):
                    os.mkdir(opt.checkpoint_path)
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print('3')
                checkpoint_path = os.path.join(opt.checkpoint_path, opt.critic_model + '_model.pth')
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)
                print('4')
                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['critic_loss_history'] = critic_loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                histories['variance_history'] = variance_history
                histories['time'] = time_history
                histories['pseudo_num_history'] = pseudo_num_history
                histories['pseudo_num_length_history'] = pseudo_num_length_history
                histories['pseudo_num_depth_history'] = pseudo_num_depth_history
                histories['pseudo_num_batch_history'] = pseudo_num_batch_history
                histories['reward_batch_history'] = reward_batch_history
                histories['entropy_batch_history'] = entropy_batch_history
                # histories['variance'] = 0
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

opt = opts.parse_opt()
train(opt)

