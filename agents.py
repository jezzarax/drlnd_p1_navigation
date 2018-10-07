from collections import namedtuple

import numpy as np
import random
import torch
import torch.nn.functional as F
from abc import abstractmethod
from torch import optim

AgentConfig = namedtuple("AgentConfig", [
    "agent_name",
    "state_size",
    "action_size",
    "lr",
    "hidden_neurons",
    "update_every",
    "batch_size",
    "buffer_size",
    "gamma",
    "tau"
])


class DummyAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, agent_config, network_builder, replay_buffer, device, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.config = agent_config
        self.name = agent_config.agent_name
        self.seed = seed
        random.seed(seed)

        # Q-Network
        self.qnetwork_local = network_builder()
        self.qnetwork_target = network_builder()
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.config.lr)

        # Replay memory
        self.memory = replay_buffer
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.debug_counter = 0
        self.device = device

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.config.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.config.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.config.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.config.action_size))

    @abstractmethod
    def learn(self, experiences, gamma):
        pass

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class DDQNAgent(DummyAgent):
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        Q_optimal_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        Q_crossed_values = self.qnetwork_target(next_states).detach().gather(1, Q_optimal_actions)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_crossed_values * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.smooth_l1_loss(Q_expected, Q_targets)

        self.debug_counter += 1
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.config.tau)

class DQNAgent(DummyAgent):
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.smooth_l1_loss(Q_expected, Q_targets)

        self.debug_counter += 1
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.config.tau)

class DQNPAgent(DummyAgent):
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, sample_ixs, weights = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        prio_update = (Q_targets - Q_expected.detach()).squeeze()
        super().replay_buffer.update_probs(sample_ixs, prio_update)

        # Compute loss
        loss = F.smooth_l1_loss(Q_expected*weights, Q_targets*weights)

        self.debug_counter += 1
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.config.tau)