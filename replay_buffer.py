from collections import namedtuple, deque

import numpy as np
import random
import torch
import math


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, device, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        return self.extract_memory_sample(random.sample(self.memory, k=self.batch_size))

    def extract_memory_sample(self, experiences):
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)
        return states, actions, rewards, next_states, dones



    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, action_size, buffer_size, batch_size, alpha, beta_start, beta_step_change, device, seed):
        self.priorities = deque(maxlen=buffer_size)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_step_change = beta_step_change
        self.max_prio = math.pow(1.0, self.alpha)
        self.prio_sum = 0
        super().__init__(action_size, buffer_size, batch_size, device, seed)

    def sample_experience(self):
        probs = np.array(self.priorities) / self.prio_sum
        sampled_experiences_ixs = np.random.choice(len(self.priorities), self.batch_size, p = probs)
        sampled_experiences = list([self.memory[i] for i in sampled_experiences_ixs])
        self.beta = min(1.0, self.beta + self.beta_step_change)
        max_weight = math.pow(probs.min() * len(self.priorities), -self.beta)
        weights = np.power((len(self.priorities)* probs[sampled_experiences_ixs]), (-self.beta)) / max_weight
        weights = torch.tensor(weights, device=self.device, dtype=torch.float)
        return (sampled_experiences, sampled_experiences_ixs, weights)

    def sample(self):
        sampled_experiences, sampled_experiences_ixs, weights = self.sample_experience()
        states, actions, rewards, next_states, dones = super().extract_memory_sample(sampled_experiences)
        return states, actions, rewards, next_states, dones, sampled_experiences_ixs, weights

    def add(self, state, action, reward, next_state, done):
        super().add(state, action, reward, next_state, done)
        new_prio = math.pow(1.0, self.alpha) if len(self.priorities) == 0 else self.max_prio
        self.prio_sum = self.prio_sum + new_prio - (self.priorities[0] if len(self.priorities) == self.priorities.maxlen else 0)
        self.priorities.append(new_prio)

    def update_probs(self, ixs, prios):
        for idx, prio in zip(ixs, prios.tolist()):
            new_prio = (abs(prio) + 1e-5) ** self.alpha
            self.prio_sum = self.prio_sum - self.priorities[idx] + new_prio
            self.priorities[idx] = new_prio
            if self.max_prio < new_prio:
                self.max_prio = new_prio

