"""
F1 Racing Gymnasium Environment

A production-grade RL environment for F1 racing that includes:
- Realistic car physics
- Dynamic tire degradation
- Fuel consumption
- Weather conditions
- Track evolution
- Telemetry logging
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Any
import yaml

from src.physics.f1_car import F1Car, F1CarConfig
from src.physics.tire_model import TireModel, TireCompound
from src.physics.aerodynamics import AerodynamicsModel, AeroConfig
from src.physics.power_unit import PowerUnit, PowerUnitConfig
from src.tracks.circuit import Circuit
from src.tracks import get_circuit
from src.envs.dynamic_conditions import DynamicConditions, WeatherCondition


class F1RacingEnv(gym.Env):
    """
    Gymnasium environment for F1 racing.

    Observation space:
        - Car state: position, velocity, heading, etc.
        - Tire state: temps, wear, grip
        - Track state: distance, segment type, racing line deviation
        - Dynamic conditions: weather, track temp, etc.
        - Temporal info: lap time, sector times, fuel remaining

    Action space:
        - throttle: [0, 1]
        - brake: [0, 1]
        - steering: [-1, 1]
        - gear: discrete [0-8]
        - ers_mode: [-1, 1]
        - drs: {0, 1} (if available)

    Reward:
        - Lap time improvement
        - Racing line adherence
        - Tire management
        - Fuel efficiency
        - Safety (avoiding crashes)
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(
        self,
        circuit_name: str = 'silverstone',
        car_config: Optional[F1CarConfig] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        max_episode_time: float = 200.0,  # seconds
        enable_dynamic_conditions: bool = True,
        tire_compound: TireCompound = TireCompound.C3,
        render_mode: Optional[str] = None,
        # v1.2.0 Advanced Features
        use_advanced_tire_model: bool = False,
        use_advanced_aero: bool = False,
        domain_randomization: Optional[str] = None,  # 'none', 'light', 'moderate', 'heavy'
    ):
        super().__init__()

        self.circuit_name = circuit_name
        self.circuit: Circuit = get_circuit(circuit_name)
        self.render_mode = render_mode

        # Store advanced features flags
        self.use_advanced_tire_model = use_advanced_tire_model
        self.use_advanced_aero = use_advanced_aero

        # Initialize car systems
        self.car = F1Car(car_config or F1CarConfig())

        # Choose tire model (standard or advanced)
        if use_advanced_tire_model:
            try:
                from src.physics.tire_model_advanced import AdvancedTireModel
                # Convert TireCompound enum to string for advanced model
                compound_map = {
                    TireCompound.C1: "C1",
                    TireCompound.C2: "C2",
                    TireCompound.C3: "C3",
                    TireCompound.C4: "C4",
                    TireCompound.C5: "C5",
                    TireCompound.INTERMEDIATE: "INTER",
                    TireCompound.WET: "WET"
                }
                compound_str = compound_map.get(tire_compound, "C3")
                self.tire_model = AdvancedTireModel(compound=compound_str)
                print(f"✓ Using Advanced Tire Model (Pacejka MF 6.2) - Compound: {compound_str}")
            except ImportError:
                print("⚠ Advanced tire model not available, using standard model")
                self.tire_model = TireModel(tire_compound)
        else:
            self.tire_model = TireModel(tire_compound)

        # Choose aero model (standard or advanced)
        if use_advanced_aero:
            try:
                from src.physics.aerodynamics_advanced import AdvancedAeroModel
                self.aero_model = AdvancedAeroModel()
                print("✓ Using Advanced Aero Model (CFD-based with ground effect)")
            except ImportError:
                print("⚠ Advanced aero model not available, using standard model")
                self.aero_model = AerodynamicsModel()
        else:
            self.aero_model = AerodynamicsModel()

        self.power_unit = PowerUnit()

        # Apply domain randomization wrapper if specified
        self.domain_randomization = domain_randomization
        if domain_randomization and domain_randomization != 'none':
            try:
                from src.envs.domain_randomization import DomainRandomizer
                self.randomizer = DomainRandomizer(level=domain_randomization)
                print(f"✓ Domain Randomization enabled: {domain_randomization}")
            except ImportError:
                print("⚠ Domain randomization not available")
                self.randomizer = None
        else:
            self.randomizer = None

        # Dynamic conditions
        self.enable_dynamic_conditions = enable_dynamic_conditions
        if enable_dynamic_conditions:
            self.conditions = DynamicConditions(
                circuit=self.circuit,
                weather=WeatherCondition.DRY
            )
        else:
            self.conditions = None

        # Simulation parameters
        self.dt = 0.02  # 20ms timestep (50 Hz)
        self.max_episode_time = max_episode_time
        self.max_steps = int(max_episode_time / self.dt)

        # Reward weights
        self.reward_weights = reward_weights or {
            'progress': 1.0,
            'racing_line': 0.5,
            'speed': 0.3,
            'smoothness': 0.2,
            'tire_management': 0.3,
            'fuel_efficiency': 0.2,
            'crash_penalty': -10.0,
            'off_track_penalty': -1.0,
        }

        # State tracking
        self.step_count = 0
        self.total_distance = 0.0
        self.lap_count = 0
        self.lap_start_time = 0.0
        self.best_lap_time = np.inf
        self.sector_times = []
        self.episode_telemetry = []

        # Previous state for reward calculation
        self.prev_distance = 0.0
        self.prev_racing_line_error = 0.0

        # Define observation and action spaces
        self._setup_spaces()

    def _setup_spaces(self):
        """Define observation and action spaces."""

        # === ACTION SPACE ===
        # Use Box space (6-dimensional) for compatibility with standard RL algorithms
        # [throttle, brake, steering, gear_normalized, ers_mode, drs_normalized]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, 0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # === OBSERVATION SPACE ===
        # We'll use a flattened observation for easier use with standard RL algorithms
        # Total: ~50-60 dimensional observation space

        obs_low = []
        obs_high = []

        # Car dynamics (10)
        obs_low.extend([0.0, -100.0, -100.0, -np.pi, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        obs_high.extend([100.0, 100.0, 100.0, np.pi, 10.0, 350.0, 4.0, 110.0, 1.0, 8.0])
        # speed, vx, vy, heading, angular_vel, kmh, fuel, max_fuel, ers_soc, gear

        # Tire state (12: 4 temps, 4 wear, 4 grip)
        obs_low.extend([20.0] * 4 + [0.0] * 4 + [0.3] * 4)
        obs_high.extend([150.0] * 4 + [1.0] * 4 + [2.0] * 4)

        # Track position (5)
        obs_low.extend([0.0, -20.0, -np.pi, 0.0, 0.0])
        obs_high.extend([10000.0, 20.0, np.pi, 350.0, 1.0])
        # distance_on_track, deviation_from_racing_line, track_heading, segment_optimal_speed, segment_type

        # Aero/Power (4)
        obs_low.extend([0.0, 0.0, 0.0, 0.0])
        obs_high.extend([5000.0, 20000.0, 1.0, 1.0])
        # drag, downforce, drs_available, drs_active

        # Dynamic conditions (6)
        obs_low.extend([5.0, 10.0, 0.9, 0.0, 0.0, 0.0])
        obs_high.extend([45.0, 60.0, 1.2, 1.0, 1.0, 1.0])
        # air_temp, track_temp, grip_multiplier, weather_type, track_wetness, time_of_day_normalized

        # Temporal (5)
        obs_low.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        obs_high.extend([300.0, 100.0, 1.0, 1.0, 1.0])
        # current_lap_time, best_lap_time_delta, lap_progress, episode_progress, next_segment_type

        self.observation_space = spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32),
            dtype=np.float32
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.

        Args:
            seed: Random seed
            options: Additional options (e.g., starting position, weather)

        Returns:
            (observation, info)
        """
        super().reset(seed=seed)

        options = options or {}

        # Reset all systems
        self.car.reset_state()
        self.tire_model.reset()
        self.power_unit.reset()

        # Set initial position on track
        start_distance = options.get('start_distance', 0.0)
        self.total_distance = start_distance

        # Get starting position
        start_pos = self.circuit.get_racing_line_at_distance(start_distance)
        start_heading = self.circuit.get_track_direction_at_distance(start_distance)

        self.car.position = np.array(start_pos)
        self.car.heading = start_heading
        self.car.velocity = np.array([options.get('start_speed', 50.0), 0.0])  # m/s

        # Reset tracking variables
        self.step_count = 0
        self.lap_count = 0
        self.lap_start_time = 0.0
        self.prev_distance = start_distance
        self.episode_telemetry = []

        # Reset dynamic conditions
        if self.conditions:
            weather = options.get('weather', WeatherCondition.DRY)
            self.conditions.reset(weather=weather)

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: Action array [throttle, brake, steering, gear_norm, ers_mode, drs_norm]

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1

        # Convert array action to components
        # action: [throttle, brake, steering, gear_normalized, ers_mode, drs_normalized]
        throttle = np.clip(float(action[0]), 0.0, 1.0)
        brake = np.clip(float(action[1]), 0.0, 1.0)
        steering = np.clip(float(action[2]), -1.0, 1.0)
        gear_normalized = np.clip(float(action[3]), 0.0, 1.0)
        gear = int(gear_normalized * 8)  # Scale to 0-8
        ers_mode = np.clip(float(action[4]), -1.0, 1.0)
        drs_normalized = np.clip(float(action[5]), 0.0, 1.0)
        drs_request = bool(drs_normalized > 0.5)

        # Check if DRS is available
        segment = self.circuit.get_segment_at_distance(self.total_distance)
        drs_active = drs_request and segment.drs_zone

        # Get current speed
        speed = np.linalg.norm(self.car.velocity)

        # === UPDATE AERODYNAMICS ===
        aero_forces = self.aero_model.calculate_forces(
            velocity=speed,
            drs_active=drs_active
        )

        # === UPDATE POWER UNIT ===
        power_output = self.power_unit.calculate_power(
            throttle=throttle,
            rpm=self.car.engine_rpm,
            ers_mode=ers_mode,
            brake_pressure=brake,
            speed=speed,
            dt=self.dt
        )

        # === UPDATE TIRES ===
        # Calculate accelerations (simplified)
        lat_accel = abs(steering) * speed * 2.0  # Simplified lateral accel
        long_accel = (throttle - brake) * 10.0  # Simplified longitudinal accel

        # Get conditions
        if self.conditions:
            track_temp = self.conditions.track_temperature
            air_temp = self.conditions.air_temperature
        else:
            track_temp = 30.0
            air_temp = 20.0

        self.tire_model.update(
            speed=speed,
            lateral_accel=lat_accel,
            longitudinal_accel=long_accel,
            steering_angle=steering,
            brake_pressure=brake,
            downforce=aero_forces['downforce'],
            track_temp=track_temp,
            ambient_temp=air_temp,
            dt=self.dt
        )

        # === UPDATE CAR PHYSICS ===
        # Prepare controls for car physics
        car_controls = {
            'throttle': throttle,
            'brake': brake,
            'steering': steering,
            'gear': gear,
            'ers_mode': ers_mode,
            'drs_active': drs_active,
        }

        # Step car physics (this updates car state)
        self.car.step(car_controls, dt=self.dt)

        # === UPDATE POSITION ON TRACK ===
        # Find closest point on racing line
        _, deviation, distance_on_track = self.circuit.find_nearest_racing_line_point(
            tuple(self.car.position)
        )

        self.total_distance = distance_on_track

        # === CHECK TERMINATION CONDITIONS ===
        terminated = False
        truncated = False

        # Off track
        if self.circuit.is_off_track(tuple(self.car.position)):
            terminated = True

        # Critical tire wear
        if self.tire_model.is_critical_wear():
            terminated = True

        # Out of fuel
        if self.car.fuel_load <= 0.0:
            terminated = True

        # Max episode time
        if self.step_count >= self.max_steps:
            truncated = True

        # Lap complete
        if self.total_distance < self.prev_distance - 100:  # Wrapped around
            self.lap_count += 1
            lap_time = self.step_count * self.dt - self.lap_start_time
            if lap_time < self.best_lap_time:
                self.best_lap_time = lap_time
            self.lap_start_time = self.step_count * self.dt
            self.power_unit.on_lap_complete()

        # === CALCULATE REWARD ===
        reward = self._calculate_reward(deviation, terminated)

        # === UPDATE DYNAMIC CONDITIONS ===
        if self.conditions:
            self.conditions.step(self.dt)

        # === COLLECT TELEMETRY ===
        self._log_telemetry()

        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()

        self.prev_distance = distance_on_track
        self.prev_racing_line_error = deviation

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector.

        Returns:
            Flattened observation array
        """
        speed = np.linalg.norm(self.car.velocity)
        speed_kmh = speed * 3.6

        # Find track info
        _, deviation, distance_on_track = self.circuit.find_nearest_racing_line_point(
            tuple(self.car.position)
        )
        segment = self.circuit.get_segment_at_distance(distance_on_track)
        track_heading = self.circuit.get_track_direction_at_distance(distance_on_track)

        # Tire state
        tire_state = self.tire_model.get_state()

        # Conditions
        if self.conditions:
            air_temp = self.conditions.air_temperature
            track_temp = self.conditions.track_temperature
            grip_mult = self.conditions.get_grip_multiplier()
            weather = float(self.conditions.weather.value)
            track_wetness = self.conditions.track_wetness
            time_of_day = self.conditions.time_of_day / 24.0
        else:
            air_temp, track_temp, grip_mult = 20.0, 30.0, 1.0
            weather, track_wetness, time_of_day = 0.0, 0.0, 0.5

        # Aero
        aero_forces = self.aero_model.calculate_forces(speed)

        # Temporal
        current_lap_time = (self.step_count * self.dt - self.lap_start_time)
        best_delta = current_lap_time - self.best_lap_time if self.best_lap_time < np.inf else 0.0
        lap_progress = distance_on_track / self.circuit.length
        episode_progress = self.step_count / self.max_steps

        # Next segment type (for prediction)
        next_segment = self.circuit.get_segment_at_distance(distance_on_track + 100.0)
        next_seg_type = hash(next_segment.type.value) % 10 / 10.0  # Normalize to [0, 1]

        # Construct observation
        obs = np.array([
            # Car dynamics (10)
            speed,
            self.car.velocity[0],
            self.car.velocity[1],
            self.car.heading,
            self.car.angular_velocity,
            speed_kmh,
            self.car.fuel_load,
            self.car.config.max_fuel_load,
            self.car.ers_energy / self.car.config.ers_max_energy,
            float(self.car.current_gear),

            # Tire state (12)
            *tire_state['temperatures'],
            *tire_state['wear'],
            *tire_state['grip_coefficients'],

            # Track position (5)
            distance_on_track,
            deviation,
            track_heading,
            segment.optimal_speed / 3.6 if segment.optimal_speed > 0 else 100.0,
            hash(segment.type.value) % 10 / 10.0,

            # Aero/Power (4)
            aero_forces['drag'],
            aero_forces['downforce'],
            float(segment.drs_zone),
            float(self.aero_model.drs_active),

            # Dynamic conditions (6)
            air_temp,
            track_temp,
            grip_mult,
            weather,
            track_wetness,
            time_of_day,

            # Temporal (5)
            current_lap_time,
            best_delta,
            lap_progress,
            episode_progress,
            next_seg_type,
        ], dtype=np.float32)

        return obs

    def _calculate_reward(self, racing_line_deviation: float, terminated: bool) -> float:
        """
        Calculate reward for current step.

        Reward components:
        - Progress: reward forward progress
        - Racing line: penalize deviation
        - Speed: reward higher speeds (when appropriate)
        - Smoothness: reward smooth control inputs
        - Tire/fuel management: reward efficiency
        - Crash: large penalty for crashes
        """
        if terminated:
            # Large penalty for crashes/failures
            return self.reward_weights['crash_penalty']

        reward = 0.0

        # Progress reward (forward movement)
        progress = (self.total_distance - self.prev_distance)
        if progress > 0:  # Normal forward progress
            reward += self.reward_weights['progress'] * progress / 100.0
        elif progress < -100:  # Lap completed
            reward += self.reward_weights['progress'] * 10.0  # Bonus for lap complete

        # Racing line adherence
        racing_line_error = racing_line_deviation / (self.circuit.track_width / 2.0)
        racing_line_reward = np.exp(-racing_line_error ** 2)  # Gaussian reward
        reward += self.reward_weights['racing_line'] * racing_line_reward

        # Speed reward (segment-appropriate)
        segment = self.circuit.get_segment_at_distance(self.total_distance)
        current_speed_kmh = np.linalg.norm(self.car.velocity) * 3.6

        if segment.optimal_speed > 0:
            speed_ratio = min(current_speed_kmh / segment.optimal_speed, 1.5)
            speed_reward = speed_ratio if speed_ratio <= 1.1 else (1.1 - (speed_ratio - 1.1))
        else:
            speed_reward = 0.5

        reward += self.reward_weights['speed'] * speed_reward

        # Tire management (penalize excessive wear rate)
        tire_wear_rate = np.mean(self.tire_model.wear) / max(self.tire_model.distance_traveled, 0.1)
        tire_reward = 1.0 - min(tire_wear_rate * 100, 1.0)
        reward += self.reward_weights['tire_management'] * tire_reward

        # Fuel efficiency
        fuel_used_ratio = 1.0 - (self.car.fuel_load / self.car.config.max_fuel_load)
        distance_ratio = self.total_distance / self.circuit.length
        if distance_ratio > 0.01:
            fuel_efficiency = 1.0 - abs(fuel_used_ratio - distance_ratio)
        else:
            fuel_efficiency = 1.0
        reward += self.reward_weights['fuel_efficiency'] * fuel_efficiency

        return reward

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the current state."""
        speed_kmh = np.linalg.norm(self.car.velocity) * 3.6

        info = {
            'step': self.step_count,
            'lap': self.lap_count,
            'distance': self.total_distance,
            'speed_kmh': speed_kmh,
            'lap_time': self.step_count * self.dt - self.lap_start_time,
            'best_lap_time': self.best_lap_time if self.best_lap_time < np.inf else None,
            'fuel_remaining': self.car.fuel_load,
            'ers_soc': self.car.ers_energy / self.car.config.ers_max_energy,
            'tire_wear_avg': np.mean(self.tire_model.wear) * 100,
            'tire_temp_avg': np.mean(self.tire_model.temperatures),
        }

        return info

    def _log_telemetry(self):
        """Log telemetry data for analysis."""
        telemetry = {
            'step': self.step_count,
            'time': self.step_count * self.dt,
            **self.car.get_telemetry(),
            **self.tire_model.get_state(),
            **self.power_unit.get_telemetry(),
        }

        self.episode_telemetry.append(telemetry)

    def get_episode_telemetry(self) -> list:
        """Get telemetry data for the episode."""
        return self.episode_telemetry

    def render(self):
        """Render the environment (optional)."""
        if self.render_mode is None:
            return

        # Basic text rendering for now
        if self.render_mode == "human":
            speed_kmh = np.linalg.norm(self.car.velocity) * 3.6
            print(f"Step: {self.step_count:5d} | "
                  f"Distance: {self.total_distance:7.1f}m | "
                  f"Speed: {speed_kmh:5.1f} km/h | "
                  f"Fuel: {self.car.fuel_load:4.1f} kg | "
                  f"Tire wear: {np.mean(self.tire_model.wear)*100:4.1f}%")

    def close(self):
        """Clean up resources."""
        pass

    @classmethod
    def from_config(cls, config_path: str) -> 'F1RacingEnv':
        """
        Create environment from YAML config file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            F1RacingEnv instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract parameters
        circuit_name = config.get('circuit', 'silverstone')
        car_config_dict = config.get('car', {})
        reward_weights = config.get('reward_weights', None)
        tire_compound_str = config.get('tire_compound', 'C3')

        # Create car config
        car_config = F1CarConfig(**car_config_dict) if car_config_dict else None

        # Parse tire compound
        tire_compound = TireCompound[tire_compound_str]

        return cls(
            circuit_name=circuit_name,
            car_config=car_config,
            reward_weights=reward_weights,
            tire_compound=tire_compound,
        )
