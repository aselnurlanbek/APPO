import collections
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(CURRENT_PATH, "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

CSV_DIR = os.path.join(CURRENT_PATH, "csv")
if not os.path.exists(CSV_DIR):
    os.mkdir(CSV_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, n_features: int = 105, n_actions: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, n_actions)

        # ln_e(x) = 1.0 --> x = e^1.0 = 2.71
        log_std_param = nn.Parameter(torch.full((n_actions,), -0.5))
        self.register_parameter("log_std", log_std_param)
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu_v = F.tanh(self.mu(x))

        std_v = self.log_std.exp()
        std_v = torch.clamp(std_v, min=0.1, max=2.0)  # Clamping for numerical stability

        return mu_v, std_v

    def get_action(self, x: torch.Tensor, exploration: bool = True) -> np.ndarray:
        mu_v, std_v = self.forward(x)

        if exploration:
            dist = Normal(loc=mu_v, scale=std_v)
            action = dist.sample()
        else:
            action = mu_v

        # Clip the actions to ensure they are within the desired range
        action = torch.clamp(action, min=-1.0, max=1.0).detach().cpu().numpy()
        return action


class Critic(nn.Module):
    """
    Value network V(s_t) = E[G_t | s_t] to use as a baseline in the reinforce
    update. This a Neural Net with 1 hidden layer
    """

    def __init__(self, n_features: int = 105):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


Transition = collections.namedtuple(
    typename="Transition", field_names=["observation", "action", "next_observation", "reward", "done"]
)


class Buffer:
    def __init__(self):
        self.buffer = collections.deque()

    def size(self) -> int:
        return len(self.buffer)

    def append(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def pop(self) -> Transition:
        return self.buffer.pop()

    def clear(self) -> None:
        self.buffer.clear()

    def get(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sample
        observations, actions, next_observations, rewards, dones = zip(*self.buffer)

        # Convert to ndarray for speed up cuda
        observations = np.array(observations)
        next_observations = np.array(next_observations)
        # observations.shape, next_observations.shape: (32, 4), (32, 4)

        actions = np.array(actions)
        actions = np.expand_dims(actions, axis=-1) if actions.ndim == 1 else actions

        rewards = np.array(rewards)
        rewards = np.expand_dims(rewards, axis=-1) if rewards.ndim == 1 else rewards
        dones = np.array(dones, dtype=bool)
        # actions.shape, rewards.shape, dones.shape: (32, 1) (32, 1) (32,)

        # Convert to tensor
        observations = torch.tensor(observations, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.float32, device=DEVICE)
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool, device=DEVICE)

        return observations, actions, next_observations, rewards, dones
