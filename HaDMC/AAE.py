# ！/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2023/7/10 22:00
# @Author  : doujizhe
# @File    : AAE
# @Software: PyCharm
import os
from torch import float32
import torch.nn as nn
import torch.nn.functional as F
from utils import pairwise_distances
import torch.optim as optim
import torch

from torch.autograd import Variable
from torch.nn.parameter import Parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# q(z|x)
class Encoder(nn.Module):
    def __init__(self, X_dim, N, z_dim):
        super(Encoder, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3_gauss = nn.Linear(N, z_dim)

    def forward(self, x, y, s):
        cat = torch.cat([x, y], dim=1)
        x1 = self.lin1(cat)
        x1 = F.relu(x1)
        x1 = self.lin2(x1)
        x1 = F.relu(x1)
        z_gauss = self.lin3_gauss(x1)
        return F.tanh(z_gauss)


# p(x|z)
class Decoder(nn.Module):
    def __init__(self, X_dim, N, z_dim):
        super(Decoder, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, X_dim)
        self.lin4 = nn.Linear(N, N)
        self.lin5 = nn.Linear(N, 1)

    def forward(self, x):
        cat = torch.cat([x], dim=1)
        x0 = self.lin1(cat)
        x0 = F.relu(x0)
        shared_layer = self.lin2(x0)
        shared_layer = F.relu(shared_layer)
        x1 = self.lin3(shared_layer)
        x2 = self.lin4(shared_layer)
        x2 = F.relu(x2)
        x2 = self.lin5(x2)
        x1 = F.tanh(x1)
        x2 = F.tanh(x2)
        return x1, x2


# D()
class Discriminator(nn.Module):
    def __init__(self, S, N, z_dim):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)
        self.s1 = nn.Linear(S, 64)

    def forward(self, x, a, s):
        s = self.s1(s)
        s = F.relu(s)
        cat = torch.cat([x, a, s], dim=1)
        x0 = self.lin1(cat)
        x0 = F.relu(x0)
        x0 = self.lin2(x0)
        x0 = F.relu(x0)
        return F.sigmoid(self.lin3(x0))


class Action_representation:
    def __init__(self,
                 state_dim,
                 action_dim,
                 parameter_action_dim,
                 reduced_action_dim=2,
                 reduce_parameter_action_dim=2,
                 embed_lr=1e-4,
                 ):
        super(Action_representation, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parameter_action_dim = parameter_action_dim
        self.reduced_action_dim = reduced_action_dim
        self.reduce_parameter_action_dim = reduce_parameter_action_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = self.reduce_parameter_action_dim
        self.embed_lr = embed_lr
        # embedding table
        init_tensor = torch.rand(self.action_dim,
                                 self.reduced_action_dim) * 2 - 1
        self.embeddings = torch.nn.Parameter(init_tensor.type(float32), requires_grad=True)
        self.embeddings = Parameter(self.embeddings.to(self.device))

        self.hidden_size = 256
        self.encoder = Encoder(self.reduced_action_dim + self.parameter_action_dim, self.hidden_size,
                               self.latent_dim).to(self.device)
        # decoder
        self.decoder = Decoder(self.reduced_action_dim, self.hidden_size,
                               self.latent_dim).to(self.device)
        # discriminator
        self.discriminator = Discriminator(self.state_dim, self.hidden_size,
                                           64 + self.reduced_action_dim + self.latent_dim).to(self.device)
        self.optimizer_dis = optim.Adam([{'params': self.discriminator.parameters()}], lr=1e-3)
        self.optimizer_ae = optim.Adam([{'params': self.encoder.parameters()},
                                        {'params': self.decoder.parameters()}, {'params': self.embeddings}],
                                       lr=1e-3)

        self.ae_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def discrete_embedding(self, ):
        emb = self.embeddings
        return emb

    def unsupervised_loss(self, s1, a1, a2, batch_size, embed_lr):
        a1 = self.get_embedding(a1).to(self.device)
        s1 = s1.to(self.device)
        a2 = a2.to(self.device)
        encoder_decoder_loss = self.aae(a1, a2, s1, batch_size)
        return encoder_decoder_loss

    def aae(self, a1, a2, s, batch_size):
        z_red_dims = self.latent_dim
        # encoder
        z_real = Variable(torch.randn(batch_size, z_red_dims)).clamp(-1, 1).to(self.device)
        z_sample = self.encoder(a1, a2, s)
        # decoder 解码z, 生成x'
        X_sample, Y_sample = self.decoder(z_sample)
        X_sample.to(self.device)
        Y_sample.to(self.device)
        d_fake = self.discriminator(z_sample, a1, s).to(self.device)
        d_real = self.discriminator(z_real, a1, s).to(self.device)
        encoder_decoder_loss1 = self.ae_loss(X_sample, a1).to(self.device)
        encoder_decoder_loss2 = self.ae_loss(Y_sample, a2).to(self.device)
        # encoder_decoder_loss3 = self.ae_loss(Y_sample, a1).to(self.device)
        encoder_decoder_loss = ((encoder_decoder_loss1 + encoder_decoder_loss2) / 2).to(self.device)
        generator_loss = self.bce_loss(d_fake, target=torch.ones_like(d_fake))
        # encoder： d_fake值越接近1越好
        discriminator_loss = self.bce_loss(d_fake, target=torch.zeros_like(d_fake)) + \
                             self.bce_loss(d_real, target=torch.ones_like(d_real))
        # 优化autoencoder
        self.optimizer_dis.zero_grad()
        tot_loss = (discriminator_loss + generator_loss).to(self.device)
        tot_loss.backward(retain_graph=True)
        self.optimizer_dis.step()
        self.optimizer_ae.zero_grad()
        encoder_decoder_loss.backward(retain_graph=False)
        self.optimizer_ae.step()
        return encoder_decoder_loss

    def select_parameter_action(self, z):
        with torch.no_grad():
            z = torch.FloatTensor(z).reshape(-1, self.reduce_parameter_action_dim).to(self.device)
            # action_c, state = self.decoder(state, z, action)
            discrete_action, parameter_action = self.decoder(z)
        return parameter_action.cpu().data.numpy().flatten()

    def select_delta_state(self, state, z, action):
        with torch.no_grad():
            action_c, state = self.decode(state, z, action)
        return state.cpu().data.numpy()

    def get_embedding(self, action):
        # Get the corresponding target embedding
        action_emb = self.embeddings[action]
        action_emb = torch.tanh(action_emb)
        return action_emb

    def get_match_scores(self, action):
        embeddings = self.embeddings
        action = action.to(self.device)
        # compute similarity probability based on L2 norm
        similarity = - pairwise_distances(action, torch.tanh(embeddings))  # Negate euclidean to convert diff into similarity score
        return similarity

    def select_discrete_action(self, action):
        similarity = self.get_match_scores(action)
        val, pos = torch.max(similarity, dim=1)
        if len(pos) == 1:
            return pos.cpu().item()
        else:
            return pos.cpu().numpy()

    def save(self, directory):

        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.decoder, '%s/decoder.pt' % directory)
        torch.save(self.embeddings, '%s/embeddings.pt' % directory)

    def load(self, directory, type=1):
        if type == 0:
            self.decoder = torch.load('%s/decoder.pt' % directory)
            self.decoder = self.decoder.to(self.device)
        self.embeddings = torch.load('%s/embeddings.pt' % directory)
        self.embeddings = Parameter(self.embeddings.to(self.device))
