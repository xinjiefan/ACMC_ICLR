import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticModel(nn.Module):
    # def __init__(self, opt):
    #     super(CriticModel, self).__init__()
    #     self.rnn_size = opt.rnn_size
    #     self.input_encoding_size = opt.input_encoding_size
    #     self.latent_dim = 50
    #     self.h1 = nn.Linear(self.rnn_size * 2 + self.input_encoding_size, self.latent_dim)
    #     self.h2 = nn.Linear(self.latent_dim, 1)
    #     self.dropout = nn.Dropout(opt.drop_prob_lm)
    #
    # def core(self, state, img_embedding):
    #     result = self.h1(torch.cat([state[0][-1], state[1][-1], img_embedding.detach()], 1))
    #     result = self.dropout(F.sigmoid(result))
    #     result = F.sigmoid(self.h2(result))
    #     return result.squeeze(1)

    def __init__(self, opt):
        super(CriticModel, self).__init__()
        self.rnn_size = opt.rnn_size
        self.input_encoding_size = opt.input_encoding_size
        self.latent_dim = 100
        self.h1 = nn.Linear(self.rnn_size * 2, self.latent_dim)
        self.h2 = nn.Linear(self.latent_dim, self.latent_dim)
        self.h3 = nn.Linear(self.latent_dim, self.latent_dim)
        self.h4 = nn.Linear(self.latent_dim, self.latent_dim)
        self.h5 = nn.Linear(self.latent_dim, 1)
        self.dropout = nn.Dropout(opt.drop_prob_lm)

    def core(self, state):
        result = self.h1(torch.cat([state[0][-1], state[1][-1]], 1))
        result = F.relu(result)
        result = self.dropout(result)
        result = self.h2(result)
        result = F.relu(result)
        result = self.dropout(result)
        result = self.h3(result)
        result = F.relu(result)
        result = self.dropout(result)
        result = self.h4(result)
        result = F.relu(result)
        result = self.dropout(result)
        result = self.h5(result)
        return result.squeeze(1)


    def forward(self, Actor, fc_feats, att_feats, opt, att_masks=None):
        batch_size = fc_feats.size(0)
        state = Actor.init_hidden(batch_size)
        seq = fc_feats.new_zeros(batch_size, Actor.seq_length, dtype=torch.long)
        value = fc_feats.new_zeros(batch_size, Actor.seq_length + 1)
        temperature = opt.get('temperature', 1.0)
        seqLogprobs = fc_feats.new_zeros(batch_size, Actor.seq_length)
        for t in range(Actor.seq_length + 2):
            if t == 0:
                img_embedding = Actor.img_embed(fc_feats)
                xt = img_embedding
            else:
                if t == 1:
                    it = fc_feats.data.new(batch_size).long().zero_()
                xt = Actor.embed(it)

            output, state = Actor.core(xt, state)
            if t >= 1:
                value[:, t-1] = self.core(state)
            if t == Actor.seq_length + 1:
                break
            if t >= 1:
                logprobs = F.log_softmax(Actor.logit(output), dim=1)
                if opt.arm_sample == 'greedy':
                    sampleLogprobs, it = torch.max(logprobs, 1)
                else:
                    if temperature == 1.0:
                        prob_prev = torch.exp(logprobs.data).cpu()
                    else:
                        prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                    it = torch.multinomial(prob_prev, 1).cuda()
                    sampleLogprobs = logprobs.gather(1, it)
                it = it.view(-1).long()
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                it = it * unfinished.type_as(it)
                seq[:,t-1] = it
                seqLogprobs[:,t-1] = sampleLogprobs.view(-1)
                if unfinished.sum() == 0:
                    break
        return value, seq, seqLogprobs

