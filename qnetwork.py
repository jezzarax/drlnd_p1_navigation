import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_layer_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layer_size (int): Number of neurons in each hidden layer
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DuelQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_layer_size, val_layer_size, adv_layer_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layer_size (int): Number of neurons in each hidden layer
            val_layer_size (int): Number of neurons in each value layer
            adv_layer_size (int): Number of neurons in each advantage layer
            seed (int): Random seed
        """
        super(DuelQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)

        self.value_layer1 = nn.Linear(hidden_layer_size, val_layer_size)
        self.value_layer2 = nn.Linear(val_layer_size, 1)
        self.advantage_layer1 = nn.Linear(hidden_layer_size, adv_layer_size)
        self.advantage_layer2 = nn.Linear(adv_layer_size, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))

        value = F.leaky_relu(self.value_layer1(x))
        value = self.value_layer2(value)

        advantage = F.leaky_relu(self.advantage_layer1(x))
        advantage = self.advantage_layer2(advantage)
        
        result = value + advantage - advantage.mean()
        return result