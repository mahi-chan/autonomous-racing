"""
Domain Randomization for Sim-to-Real Transfer

Adds realistic noise and variability to physics parameters to make
the learned policy more robust when transferring to real-world scenarios.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class RandomizationConfig:
    """Configuration for domain randomization."""

    # Grip randomization
    grip_multiplier_range: tuple = (0.92, 1.08)  # ±8%
    tire_deg_rate_range: tuple = (0.85, 1.15)  # ±15%

    # Aerodynamics
    drag_coefficient_range: tuple = (0.97, 1.03)  # ±3%
    downforce_coefficient_range: tuple = (0.95, 1.05)  # ±5%

    # Power unit
    power_multiplier_range: tuple = (0.98, 1.02)  # ±2%
    fuel_consumption_range: tuple = (0.95, 1.05)  # ±5%

    # Track conditions
    track_temp_offset_range: tuple = (-5.0, 5.0)  # ±5°C
    wind_speed_range: tuple = (0.0, 5.0)  # 0-5 m/s

    # Sensor noise
    position_noise_std: float = 0.02  # meters
    velocity_noise_std: float = 0.1  # m/s
    heading_noise_std: float = 0.01  # radians

    # Actuation noise
    throttle_lag: tuple = (0.01, 0.03)  # 10-30ms delay
    brake_lag: tuple = (0.01, 0.03)
    steering_lag: tuple = (0.02, 0.04)


class DomainRandomizer:
    """
    Applies domain randomization to physics parameters.

    Usage:
        randomizer = DomainRandomizer()
        randomized_params = randomizer.randomize()
        # Apply randomized_params to environment
    """

    def __init__(self, config: RandomizationConfig = None):
        self.config = config or RandomizationConfig()
        self.current_params = {}

    def randomize(self) -> Dict[str, float]:
        """
        Generate randomized physics parameters.

        Returns:
            Dictionary with randomized multipliers and offsets
        """
        params = {
            # Grip and tire
            'grip_multiplier': np.random.uniform(*self.config.grip_multiplier_range),
            'tire_deg_rate': np.random.uniform(*self.config.tire_deg_rate_range),

            # Aerodynamics
            'drag_multiplier': np.random.uniform(*self.config.drag_coefficient_range),
            'downforce_multiplier': np.random.uniform(*self.config.downforce_coefficient_range),

            # Power
            'power_multiplier': np.random.uniform(*self.config.power_multiplier_range),
            'fuel_consumption_multiplier': np.random.uniform(*self.config.fuel_consumption_range),

            # Environment
            'track_temp_offset': np.random.uniform(*self.config.track_temp_offset_range),
            'wind_speed': np.random.uniform(*self.config.wind_speed_range),
            'wind_direction': np.random.uniform(0, 2 * np.pi),

            # Delays
            'throttle_lag': np.random.uniform(*self.config.throttle_lag),
            'brake_lag': np.random.uniform(*self.config.brake_lag),
            'steering_lag': np.random.uniform(*self.config.steering_lag),
        }

        self.current_params = params
        return params

    def add_sensor_noise(self, state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Add realistic sensor noise to state measurements.

        Args:
            state: Clean state dictionary

        Returns:
            Noisy state dictionary
        """
        noisy_state = state.copy()

        # Position noise
        if 'position' in noisy_state:
            pos_noise = np.random.normal(0, self.config.position_noise_std, 2)
            noisy_state['position'] = noisy_state['position'] + pos_noise

        # Velocity noise
        if 'velocity' in noisy_state:
            vel_noise = np.random.normal(0, self.config.velocity_noise_std, 2)
            noisy_state['velocity'] = noisy_state['velocity'] + vel_noise

        # Heading noise
        if 'heading' in noisy_state:
            heading_noise = np.random.normal(0, self.config.heading_noise_std)
            noisy_state['heading'] = noisy_state['heading'] + heading_noise

        return noisy_state

    def add_actuation_delay(
        self,
        actions: np.ndarray,
        prev_actions: np.ndarray,
        action_buffer: list
    ) -> np.ndarray:
        """
        Simulate actuation delays.

        Args:
            actions: Current commanded actions
            prev_actions: Previous actions
            action_buffer: Buffer of recent actions

        Returns:
            Delayed actions
        """
        # Simple exponential smoothing for delay
        lags = np.array([
            self.current_params.get('throttle_lag', 0.02),
            self.current_params.get('brake_lag', 0.02),
            self.current_params.get('steering_lag', 0.03),
            0.0,  # gear (discrete, no lag)
            0.02,  # ers_mode
            0.0,  # drs (discrete, no lag)
        ])

        # Exponential moving average
        alpha = 1.0 - lags / 0.02  # Assuming 20ms timestep
        alpha = np.clip(alpha, 0.0, 1.0)

        delayed_actions = alpha * actions + (1 - alpha) * prev_actions

        return delayed_actions

    def apply_to_car_config(self, car_config):
        """
        Apply randomization to car configuration.

        Args:
            car_config: F1CarConfig object

        Returns:
            Modified car_config
        """
        if not self.current_params:
            self.randomize()

        # Modify aero
        car_config.drag_coefficient *= self.current_params['drag_multiplier']
        car_config.downforce_coefficient *= self.current_params['downforce_multiplier']

        # Modify power
        car_config.max_power_ice *= self.current_params['power_multiplier']

        # Modify fuel consumption
        car_config.fuel_consumption_rate *= self.current_params['fuel_consumption_multiplier']

        return car_config


def create_randomized_env(base_env, enable_randomization: bool = True):
    """
    Wrap environment with domain randomization.

    Args:
        base_env: Base F1RacingEnv
        enable_randomization: Enable randomization

    Returns:
        Wrapped environment
    """
    if not enable_randomization:
        return base_env

    class RandomizedEnv:
        """Environment wrapper with domain randomization."""

        def __init__(self, env):
            self.env = env
            self.randomizer = DomainRandomizer()
            self.prev_actions = None
            self.action_buffer = []

        def reset(self, **kwargs):
            # Randomize physics parameters
            randomized_params = self.randomizer.randomize()

            # Apply to environment
            self.env.conditions.grip_multiplier_base = randomized_params['grip_multiplier']
            self.env.car.config = self.randomizer.apply_to_car_config(
                self.env.car.config
            )

            # Reset action buffer
            self.prev_actions = np.zeros(6)
            self.action_buffer = []

            obs, info = self.env.reset(**kwargs)
            return obs, info

        def step(self, action):
            # Apply actuation delay
            if self.prev_actions is not None:
                delayed_action = self.randomizer.add_actuation_delay(
                    action, self.prev_actions, self.action_buffer
                )
            else:
                delayed_action = action

            self.prev_actions = action.copy()

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(delayed_action)

            # Add sensor noise to observation
            # Note: Observation is a flat array, so we'll add small noise
            obs_noisy = obs + np.random.normal(
                0, 0.01, size=obs.shape
            )

            return obs_noisy, reward, terminated, truncated, info

        def __getattr__(self, name):
            return getattr(self.env, name)

    return RandomizedEnv(base_env)


# Preset randomization levels
RANDOMIZATION_PRESETS = {
    'none': RandomizationConfig(
        grip_multiplier_range=(1.0, 1.0),
        drag_coefficient_range=(1.0, 1.0),
        downforce_coefficient_range=(1.0, 1.0),
        power_multiplier_range=(1.0, 1.0),
    ),
    'light': RandomizationConfig(
        grip_multiplier_range=(0.98, 1.02),
        drag_coefficient_range=(0.99, 1.01),
        downforce_coefficient_range=(0.98, 1.02),
        power_multiplier_range=(0.99, 1.01),
    ),
    'moderate': RandomizationConfig(),  # Uses defaults
    'heavy': RandomizationConfig(
        grip_multiplier_range=(0.85, 1.15),
        tire_deg_rate_range=(0.75, 1.25),
        drag_coefficient_range=(0.93, 1.07),
        downforce_coefficient_range=(0.90, 1.10),
        power_multiplier_range=(0.95, 1.05),
    ),
}
