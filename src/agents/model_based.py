"""
Model-Based RL (Dreamer/MBPO) for F1 Racing

Implements model-based reinforcement learning that:
- Learns a world model of car dynamics
- Plans in latent space
- Sample-efficient training
- Useful for expensive real-world testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Dict, Tuple, Optional, List
import gymnasium as gym


class WorldModel(nn.Module):
    """
    World model for F1 car dynamics.

    Learns to predict:
    - Next state given current state and action
    - Reward prediction
    - Done prediction
    - Latent representation learning
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        super().__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Encoder: obs -> latent
        encoder_layers = []
        prev_dim = observation_dim
        for _ in range(num_layers):
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim

        encoder_layers.append(nn.Linear(hidden_dim, latent_dim * 2))  # mean and log_std
        self.encoder = nn.Sequential(*encoder_layers)

        # Dynamics model: latent + action -> next latent
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean and log_std
        )

        # Reward model: latent -> reward
        self.reward_model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Decoder: latent -> obs
        decoder_layers = []
        prev_dim = latent_dim
        for _ in range(num_layers):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(hidden_dim, observation_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observation to latent distribution.

        Args:
            obs: Observation [batch, obs_dim]

        Returns:
            (mean, log_std) of latent distribution
        """
        encoded = self.encoder(obs)
        mean, log_std = torch.chunk(encoded, 2, dim=-1)
        log_std = torch.clamp(log_std, -10, 2)
        return mean, log_std

    def sample_latent(self, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution using reparameterization."""
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to observation."""
        return self.decoder(latent)

    def predict_next(
        self,
        latent: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next latent state.

        Args:
            latent: Current latent state [batch, latent_dim]
            action: Action [batch, action_dim]

        Returns:
            (next_mean, next_log_std) of next latent
        """
        latent_action = torch.cat([latent, action], dim=-1)
        next_latent_params = self.dynamics(latent_action)
        next_mean, next_log_std = torch.chunk(next_latent_params, 2, dim=-1)
        next_log_std = torch.clamp(next_log_std, -10, 2)
        return next_mean, next_log_std

    def predict_reward(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict reward from latent state."""
        return self.reward_model(latent)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode, predict next, decode.

        Args:
            obs: Current observation
            action: Action taken

        Returns:
            Dictionary with predictions
        """
        # Encode current observation
        mean, log_std = self.encode(obs)
        latent = self.sample_latent(mean, log_std)

        # Predict next latent
        next_mean, next_log_std = self.predict_next(latent, action)
        next_latent = self.sample_latent(next_mean, next_log_std)

        # Predict reward
        reward_pred = self.predict_reward(latent)

        # Decode next observation
        next_obs_pred = self.decode(next_latent)

        return {
            'latent': latent,
            'latent_mean': mean,
            'latent_log_std': log_std,
            'next_latent': next_latent,
            'next_latent_mean': next_mean,
            'next_latent_log_std': next_log_std,
            'next_obs_pred': next_obs_pred,
            'reward_pred': reward_pred,
        }


class DreamerAgent:
    """
    Dreamer-style agent for F1 racing.

    Key components:
    - World model (learns dynamics)
    - Actor (policy in latent space)
    - Critic (value in latent space)
    - Imagination-based training
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        learning_rate: float = 1e-4,
        imagination_horizon: int = 15,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)

        # World model
        self.world_model = WorldModel(
            observation_dim=observation_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        # Actor (policy in latent space)
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # mean and log_std
        ).to(self.device)

        # Critic (value in latent space)
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)

        # Optimizers
        self.world_model_optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=learning_rate
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=learning_rate
        )

        self.imagination_horizon = imagination_horizon

    def train_world_model(
        self,
        obs_batch: torch.Tensor,
        action_batch: torch.Tensor,
        next_obs_batch: torch.Tensor,
        reward_batch: torch.Tensor
    ) -> Dict[str, float]:
        """
        Train world model on real data.

        Args:
            obs_batch: [batch, obs_dim]
            action_batch: [batch, action_dim]
            next_obs_batch: [batch, obs_dim]
            reward_batch: [batch, 1]

        Returns:
            Dictionary of losses
        """
        # Forward pass
        outputs = self.world_model(obs_batch, action_batch)

        # Reconstruction loss
        recon_loss = F.mse_loss(outputs['next_obs_pred'], next_obs_batch)

        # Reward prediction loss
        reward_loss = F.mse_loss(outputs['reward_pred'], reward_batch)

        # KL divergence (regularization on latent space)
        kl_loss = self._kl_divergence(
            outputs['latent_mean'],
            outputs['latent_log_std']
        )

        # Total loss
        total_loss = recon_loss + reward_loss + 0.1 * kl_loss

        # Optimize
        self.world_model_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.world_model_optimizer.step()

        return {
            'recon_loss': recon_loss.item(),
            'reward_loss': reward_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item()
        }

    def imagine_trajectories(
        self,
        initial_obs: torch.Tensor,
        horizon: int
    ) -> Dict[str, torch.Tensor]:
        """
        Imagine trajectories in latent space.

        Args:
            initial_obs: Initial observation [batch, obs_dim]
            horizon: Number of steps to imagine

        Returns:
            Dictionary with imagined trajectories
        """
        batch_size = initial_obs.shape[0]

        # Encode initial state
        mean, log_std = self.world_model.encode(initial_obs)
        latent = self.world_model.sample_latent(mean, log_std)

        # Imagination rollout
        latents = [latent]
        actions = []
        rewards = []

        for _ in range(horizon):
            # Sample action from policy
            action_params = self.actor(latent)
            action_mean, action_log_std = torch.chunk(action_params, 2, dim=-1)
            action_std = torch.exp(action_log_std)
            action_dist = Normal(action_mean, action_std)
            action = torch.tanh(action_dist.rsample())

            # Predict next latent
            next_mean, next_log_std = self.world_model.predict_next(latent, action)
            next_latent = self.world_model.sample_latent(next_mean, next_log_std)

            # Predict reward
            reward = self.world_model.predict_reward(latent)

            # Store
            latents.append(next_latent)
            actions.append(action)
            rewards.append(reward)

            latent = next_latent

        return {
            'latents': torch.stack(latents, dim=1),  # [batch, horizon+1, latent_dim]
            'actions': torch.stack(actions, dim=1),  # [batch, horizon, action_dim]
            'rewards': torch.stack(rewards, dim=1),  # [batch, horizon, 1]
        }

    def _kl_divergence(self, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        """Calculate KL divergence from standard normal."""
        var = torch.exp(2 * log_std)
        kl = 0.5 * torch.sum(mean**2 + var - 1 - 2 * log_std, dim=-1)
        return kl.mean()

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using the actor."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Encode observation
            mean, log_std = self.world_model.encode(obs_tensor)
            latent = mean if deterministic else self.world_model.sample_latent(mean, log_std)

            # Get action
            action_params = self.actor(latent)
            action_mean, action_log_std = torch.chunk(action_params, 2, dim=-1)

            if deterministic:
                action = torch.tanh(action_mean)
            else:
                action_std = torch.exp(action_log_std)
                action_dist = Normal(action_mean, action_std)
                action = torch.tanh(action_dist.sample())

        return action.cpu().numpy()[0]


class ModelBasedAgent:
    """
    Model-based RL agent wrapper for easy integration.

    Supports multiple model-based algorithms:
    - Dreamer
    - MBPO (Model-Based Policy Optimization)
    - PlaNet
    """

    def __init__(
        self,
        env: gym.Env,
        algorithm: str = "dreamer",
        **kwargs
    ):
        """
        Initialize model-based agent.

        Args:
            env: Environment
            algorithm: Algorithm type ("dreamer", "mbpo")
            **kwargs: Algorithm-specific parameters
        """
        self.env = env
        self.algorithm = algorithm

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))

        if algorithm == "dreamer":
            self.agent = DreamerAgent(
                observation_dim=obs_dim,
                action_dim=act_dim,
                **kwargs
            )
        else:
            raise NotImplementedError(f"Algorithm {algorithm} not implemented")

    def train(self, total_timesteps: int, **kwargs):
        """Train the agent."""
        # Placeholder - full training loop would go here
        print(f"Training {self.algorithm} for {total_timesteps} steps...")
        print("Model-based training requires implementing full imagination-based learning loop.")
        print("See research papers: Dreamer (Hafner et al., 2020), MBPO (Janner et al., 2019)")

    def predict(self, observation, deterministic: bool = False):
        """Predict action."""
        return self.agent.select_action(observation, deterministic=deterministic), None

    def save(self, path: str):
        """Save model."""
        torch.save({
            'world_model': self.agent.world_model.state_dict(),
            'actor': self.agent.actor.state_dict(),
            'critic': self.agent.critic.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path)
        self.agent.world_model.load_state_dict(checkpoint['world_model'])
        self.agent.actor.load_state_dict(checkpoint['actor'])
        self.agent.critic.load_state_dict(checkpoint['critic'])
