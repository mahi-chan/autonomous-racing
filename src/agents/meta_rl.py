"""
Meta-RL for Track Adaptation

Implements meta-learning (MAML/Reptile) for:
- Quick adaptation to new circuits
- Few-shot learning for track variations
- Transfer from simulation to reality
- Setup optimization across conditions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import gymnasium as gym
from copy import deepcopy


class AdaptiveRacingPolicy(nn.Module):
    """
    Policy network that can quickly adapt to new tracks.

    Features:
    - Fast weight adaptation
    - Task/track embedding
    - Few-shot learning capability
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        task_embedding_dim: int = 32
    ):
        super().__init__()

        self.task_embedding_dim = task_embedding_dim

        # Task encoder (learns track-specific features)
        self.task_encoder = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, task_embedding_dim)
        )

        # Main policy network (task-conditioned)
        layers = []
        prev_dim = observation_dim + task_embedding_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim

        self.feature_net = nn.Sequential(*layers)

        # Action head
        self.action_mean = nn.Linear(prev_dim, action_dim)
        self.action_log_std = nn.Linear(prev_dim, action_dim)

    def encode_task(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Encode task/track from observations.

        Args:
            observations: Batch of observations from current task [batch, obs_dim]

        Returns:
            Task embedding [task_embedding_dim]
        """
        # Average over batch to get task representation
        task_features = self.task_encoder(observations)
        task_embedding = task_features.mean(dim=0)
        return task_embedding

    def forward(
        self,
        observation: torch.Tensor,
        task_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with task conditioning.

        Args:
            observation: Current observation [batch, obs_dim]
            task_embedding: Task embedding [task_embedding_dim]

        Returns:
            (action_mean, action_log_std)
        """
        # Expand task embedding to batch
        batch_size = observation.shape[0]
        task_emb_expanded = task_embedding.unsqueeze(0).expand(batch_size, -1)

        # Concatenate observation and task embedding
        obs_task = torch.cat([observation, task_emb_expanded], dim=-1)

        # Forward through network
        features = self.feature_net(obs_task)

        action_mean = self.action_mean(features)
        action_log_std = self.action_log_std(features)
        action_log_std = torch.clamp(action_log_std, -20, 2)

        return action_mean, action_log_std

    def sample_action(
        self,
        observation: torch.Tensor,
        task_embedding: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        action_mean, action_log_std = self.forward(observation, task_embedding)

        if deterministic:
            return torch.tanh(action_mean), None

        action_std = torch.exp(action_log_std)
        action_dist = torch.distributions.Normal(action_mean, action_std)
        action_sample = action_dist.rsample()
        action = torch.tanh(action_sample)

        # Log probability with tanh correction
        log_prob = action_dist.log_prob(action_sample)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class MAML:
    """
    Model-Agnostic Meta-Learning for racing.

    Learns to quickly adapt to new tracks with few laps of experience.
    """

    def __init__(
        self,
        policy: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize MAML.

        Args:
            policy: Adaptive policy network
            inner_lr: Learning rate for inner loop (task adaptation)
            outer_lr: Learning rate for outer loop (meta-learning)
            num_inner_steps: Number of gradient steps for adaptation
            device: Device to use
        """
        self.policy = policy.to(device)
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.device = torch.device(device)

        # Meta-optimizer
        self.meta_optimizer = optim.Adam(self.policy.parameters(), lr=outer_lr)

    def inner_loop(
        self,
        task_data: Dict[str, torch.Tensor],
        task_embedding: torch.Tensor
    ) -> nn.Module:
        """
        Adapt policy to a specific task (track).

        Args:
            task_data: Dictionary with 'observations', 'actions', 'rewards'
            task_embedding: Embedding for this task

        Returns:
            Adapted policy
        """
        # Clone policy for adaptation
        adapted_policy = deepcopy(self.policy)

        # Create optimizer for adapted policy
        inner_optimizer = optim.SGD(adapted_policy.parameters(), lr=self.inner_lr)

        # Adaptation steps
        for _ in range(self.num_inner_steps):
            # Forward pass
            action_mean, action_log_std = adapted_policy(
                task_data['observations'],
                task_embedding
            )

            # Compute loss (e.g., behavioral cloning from expert or policy gradient)
            # For simplicity, using behavioral cloning here
            loss = F.mse_loss(torch.tanh(action_mean), task_data['actions'])

            # Update
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

        return adapted_policy

    def meta_train_step(
        self,
        task_batch: List[Dict[str, torch.Tensor]]
    ) -> float:
        """
        Perform one meta-training step.

        Args:
            task_batch: List of task datasets (different tracks/conditions)

        Returns:
            Meta loss
        """
        meta_loss = 0.0

        for task_data in task_batch:
            # Split into support and query sets
            support_data, query_data = self._split_support_query(task_data)

            # Encode task from support set
            task_embedding = self.policy.encode_task(support_data['observations'])

            # Inner loop adaptation on support set
            adapted_policy = self.inner_loop(support_data, task_embedding)

            # Evaluate on query set
            with torch.no_grad():
                query_embedding = adapted_policy.encode_task(query_data['observations'])

            action_mean, _ = adapted_policy(query_data['observations'], query_embedding)
            task_loss = F.mse_loss(torch.tanh(action_mean), query_data['actions'])

            meta_loss += task_loss

        # Average over tasks
        meta_loss = meta_loss / len(task_batch)

        # Meta-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.meta_optimizer.step()

        return meta_loss.item()

    def _split_support_query(
        self,
        task_data: Dict[str, torch.Tensor],
        support_ratio: float = 0.5
    ) -> Tuple[Dict, Dict]:
        """Split task data into support and query sets."""
        n_samples = task_data['observations'].shape[0]
        n_support = int(n_samples * support_ratio)

        indices = torch.randperm(n_samples)
        support_idx = indices[:n_support]
        query_idx = indices[n_support:]

        support_data = {
            'observations': task_data['observations'][support_idx],
            'actions': task_data['actions'][support_idx],
            'rewards': task_data['rewards'][support_idx]
        }

        query_data = {
            'observations': task_data['observations'][query_idx],
            'actions': task_data['actions'][query_idx],
            'rewards': task_data['rewards'][query_idx]
        }

        return support_data, query_data

    def adapt_to_new_track(
        self,
        track_data: Dict[str, np.ndarray],
        num_adaptation_steps: Optional[int] = None
    ) -> nn.Module:
        """
        Quickly adapt to a new track with few samples.

        Args:
            track_data: Small dataset from new track
            num_adaptation_steps: Override default adaptation steps

        Returns:
            Adapted policy
        """
        # Convert to tensors
        task_data = {
            k: torch.FloatTensor(v).to(self.device)
            for k, v in track_data.items()
        }

        # Get task embedding
        task_embedding = self.policy.encode_task(task_data['observations'])

        # Adapt
        if num_adaptation_steps:
            original_steps = self.num_inner_steps
            self.num_inner_steps = num_adaptation_steps

        adapted_policy = self.inner_loop(task_data, task_embedding)

        if num_adaptation_steps:
            self.num_inner_steps = original_steps

        return adapted_policy


class MetaRLAgent:
    """
    Meta-RL agent wrapper for F1 racing.

    Provides easy interface for:
    - Meta-training across multiple tracks
    - Quick adaptation to new tracks
    - Transfer learning
    """

    def __init__(
        self,
        env: gym.Env,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5,
        task_embedding_dim: int = 32,
        **kwargs
    ):
        """
        Initialize Meta-RL agent.

        Args:
            env: Environment
            inner_lr: Inner loop learning rate
            outer_lr: Outer loop (meta) learning rate
            num_inner_steps: Adaptation steps
            task_embedding_dim: Task embedding dimension
        """
        self.env = env

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))

        # Create policy
        self.policy = AdaptiveRacingPolicy(
            observation_dim=obs_dim,
            action_dim=act_dim,
            task_embedding_dim=task_embedding_dim
        )

        # Create MAML trainer
        self.maml = MAML(
            policy=self.policy,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            num_inner_steps=num_inner_steps
        )

        # Track current task embedding
        self.current_task_embedding = None

    def meta_train(
        self,
        track_datasets: Dict[str, Dict[str, np.ndarray]],
        num_iterations: int = 1000,
        batch_size: int = 4
    ):
        """
        Meta-train across multiple tracks.

        Args:
            track_datasets: Dict mapping track names to datasets
            num_iterations: Number of meta-training iterations
            batch_size: Number of tasks per meta-update
        """
        track_names = list(track_datasets.keys())

        for iteration in range(num_iterations):
            # Sample batch of tracks
            sampled_tracks = np.random.choice(track_names, size=batch_size, replace=False)

            # Prepare task batch
            task_batch = []
            for track_name in sampled_tracks:
                task_data = {
                    k: torch.FloatTensor(v).to(self.maml.device)
                    for k, v in track_datasets[track_name].items()
                }
                task_batch.append(task_data)

            # Meta-update
            meta_loss = self.maml.meta_train_step(task_batch)

            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Meta Loss: {meta_loss:.4f}")

    def adapt_to_track(
        self,
        track_data: Dict[str, np.ndarray],
        num_steps: Optional[int] = None
    ):
        """
        Adapt to a new track with few samples.

        Args:
            track_data: Small dataset from new track
            num_steps: Number of adaptation steps
        """
        adapted_policy = self.maml.adapt_to_new_track(track_data, num_steps)

        # Update current policy
        self.policy = adapted_policy

        # Update task embedding
        obs_tensor = torch.FloatTensor(track_data['observations']).to(self.maml.device)
        self.current_task_embedding = self.policy.encode_task(obs_tensor)

    def predict(self, observation: np.ndarray, deterministic: bool = False):
        """Predict action for current task."""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.maml.device)

        if self.current_task_embedding is None:
            # Use zero embedding if no task adaptation yet
            self.current_task_embedding = torch.zeros(
                self.policy.task_embedding_dim,
                device=self.maml.device
            )

        with torch.no_grad():
            action, _ = self.policy.sample_action(
                obs_tensor,
                self.current_task_embedding,
                deterministic=deterministic
            )

        return action.cpu().numpy()[0], None

    def save(self, path: str):
        """Save meta-learned policy."""
        torch.save({
            'policy': self.policy.state_dict(),
            'maml_optimizer': self.maml.meta_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load meta-learned policy."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.maml.meta_optimizer.load_state_dict(checkpoint['maml_optimizer'])


# Utility functions
def collect_track_dataset(
    env: gym.Env,
    num_episodes: int = 10
) -> Dict[str, np.ndarray]:
    """
    Collect dataset from a track for meta-learning.

    Args:
        env: Environment
        num_episodes: Number of episodes to collect

    Returns:
        Dataset dict with observations, actions, rewards
    """
    observations = []
    actions = []
    rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            # Random or expert policy
            action = env.action_space.sample()

            observations.append(obs)
            actions.append(action)

            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)

            done = terminated or truncated

    return {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards)
    }
