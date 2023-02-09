import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from replay_buffer import ReplayBuffer
from model import CriticNetwork, ActorNetwork
from noise import OUActionNoise


class Agent(object):
    def __init__(self, alpha_c, alpha_a, input_dims, tau, env, gamma=0.99, n_actions=2,
                        max_size=1000000, layer1_size=400, layer2_size=300, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.critic = CriticNetwork(alpha_c, input_dims, layer1_size, layer2_size, n_actions=n_actions, model_dir='models')
        self.actor = ActorNetwork(alpha_a, input_dims, layer1_size, layer2_size, n_actions=n_actions, model_dir='models')

        self.target_critic = CriticNetwork(alpha_c, input_dims, layer1_size, layer2_size, n_actions=n_actions, model_dir='models')
        self.target_actor = ActorNetwork(alpha_a, input_dims, layer1_size, layer2_size, n_actions=n_actions, model_dir='models')

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        observation = np.array([observation], dtype=np.float32)
        observation = torch.from_numpy(observation).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + torch.Tensor(self.noise()).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)
        action = torch.tensor(action, dtype=torch.float).to(self.actor.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(new_state)
        target_critic_value = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*target_critic_value[j]*done[j])

        target = torch.tensor(target).to(self.actor.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        self.critic.eval()

        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        critic_params = dict(self.critic.named_parameters())
        target_critic_params = dict(self.target_critic.named_parameters())

        actor_params = dict(self.actor.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())

        for name in critic_params:
            critic_params[name] = tau*critic_params[name].clone() + \
                                    (1-tau)*target_critic_params[name].clone()

        self.target_critic.load_state_dict(critic_params, strict=False)

        for name in actor_params:
            actor_params[name] = tau*actor_params[name].clone() + \
                                    (1-tau)*target_actor_params[name].clone()

        self.target_actor.load_state_dict(actor_params, strict=False)

    def save_models(self, episode_no):
        print('... saving models ...')
        self.actor.save_checkpoint('actor_%d' % episode_no)
        self.critic.save_checkpoint('critic_%d' % episode_no)
        self.target_actor.save_checkpoint('target_actor_%d' % episode_no)
        self.target_critic.save_checkpoint('target_critic_%d' % episode_no)

    def load_models(self, episode_no):
        print('... loading models ...')
        self.actor.load_checkpoint('actor_%d' % episode_no)
        self.critic.load_checkpoint('critic_%d' % episode_no)
        self.target_actor.load_checkpoint('target_actor_%d' % episode_no)
        self.target_critic.load_checkpoint('target_critic_%d' % episode_no)

        