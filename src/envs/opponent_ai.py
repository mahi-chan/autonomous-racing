"""
Opponent AI System for Multi-Agent Racing

Provides AI-controlled opponents with various skill levels and behaviors.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class OpponentProfile:
    """AI opponent configuration."""

    name: str = "AI Driver"
    skill_level: float = 0.8  # 0=beginner, 1=perfect
    aggression: float = 0.5  # 0=defensive, 1=aggressive
    consistency: float = 0.9  # 0=erratic, 1=consistent
    tire_management: float = 0.7  # 0=poor, 1=excellent
    reaction_time: float = 0.15  # seconds

    # Racing line deviation
    racing_line_std: float = 0.5  # meters

    # Overtaking behavior
    overtake_probability: float = 0.7
    defend_probability: float = 0.8


class OpponentAI:
    """
    AI-controlled opponent driver.

    Provides realistic racing behavior including:
    - Racing line following with skill-dependent accuracy
    - Defensive and offensive maneuvers
    - Tire and fuel management
    - Mistakes and variability
    """

    def __init__(self, profile: Optional[OpponentProfile] = None):
        self.profile = profile or OpponentProfile()

        # State
        self.position = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        self.heading = 0.0

        # Racing state
        self.is_defending = False
        self.is_attacking = False
        self.target_car_ahead = None
        self.target_car_behind = None

        # Internal control smoothing
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        self.prev_steering = 0.0

        # Mistake system
        self.mistake_probability = (1.0 - self.profile.skill_level) * 0.05
        self.in_mistake = False
        self.mistake_duration = 0

    def get_action(
        self,
        state: Dict,
        circuit,
        opponent_positions: List[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Get AI action based on current state.

        Args:
            state: Current car state
            circuit: Circuit object
            opponent_positions: List of other car positions

        Returns:
            Action array [throttle, brake, steering, gear, ers_mode, drs]
        """
        # Extract state
        position = state.get('position', self.position)
        velocity = state.get('velocity', self.velocity)
        heading = state.get('heading', self.heading)
        speed = np.linalg.norm(velocity)

        # Find track info
        _, deviation, distance_on_track = circuit.find_nearest_racing_line_point(
            tuple(position)
        )
        segment = circuit.get_segment_at_distance(distance_on_track)
        track_heading = circuit.get_track_direction_at_distance(distance_on_track)

        # Check for opponents
        self._update_opponent_awareness(position, opponent_positions)

        # Decide racing mode
        if self.is_defending and np.random.random() < self.profile.defend_probability:
            return self._defensive_action(state, circuit)
        elif self.is_attacking and np.random.random() < self.profile.overtake_probability:
            return self._attacking_action(state, circuit)
        else:
            return self._racing_action(state, circuit)

    def _racing_action(self, state: Dict, circuit) -> np.ndarray:
        """Standard racing line following."""
        position = state.get('position', self.position)
        velocity = state.get('velocity', self.velocity)
        speed = np.linalg.norm(velocity) * 3.6  # km/h

        _, deviation, distance_on_track = circuit.find_nearest_racing_line_point(
            tuple(position)
        )
        segment = circuit.get_segment_at_distance(distance_on_track)
        track_heading = circuit.get_track_direction_at_distance(distance_on_track)
        heading = state.get('heading', 0.0)

        # Target speed (skill-dependent)
        if segment.optimal_speed > 0:
            target_speed = segment.optimal_speed * (0.95 + 0.05 * self.profile.skill_level)
        else:
            target_speed = 200.0  # Default

        # Add consistency noise
        target_speed *= np.random.normal(1.0, (1.0 - self.profile.consistency) * 0.05)

        # Steering to follow racing line
        heading_error = self._normalize_angle(track_heading - heading)

        # Skill-based steering accuracy
        steering_gain = 2.0 * self.profile.skill_level
        racing_line_correction = -deviation * 0.1
        steering = np.clip(
            heading_error * steering_gain + racing_line_correction,
            -1.0, 1.0
        )

        # Add skill-dependent noise
        steering += np.random.normal(0, self.profile.racing_line_std * 0.1)

        # Speed control
        speed_error = target_speed - speed

        if speed_error > 5.0:
            # Accelerate
            throttle = np.clip(speed_error / 50.0, 0.0, 1.0)
            brake = 0.0
        elif speed_error < -10.0:
            # Brake
            throttle = 0.0
            brake = np.clip(-speed_error / 100.0, 0.0, 1.0)
        else:
            # Maintain
            throttle = 0.5
            brake = 0.0

        # Smooth controls
        throttle = self._smooth_control(throttle, self.prev_throttle, 0.3)
        brake = self._smooth_control(brake, self.prev_brake, 0.5)
        steering = self._smooth_control(steering, self.prev_steering, 0.2)

        self.prev_throttle = throttle
        self.prev_brake = brake
        self.prev_steering = steering

        # Gear selection (simplified - based on speed)
        gear_normalized = np.clip(speed / 300.0, 0.0, 1.0)

        # ERS deployment (skill-dependent)
        if segment.drs_zone and self.profile.skill_level > 0.6:
            ers_mode = 1.0  # Deploy on straights
        else:
            ers_mode = 0.0

        # DRS
        drs = 1.0 if segment.drs_zone else 0.0

        # Random mistakes
        if np.random.random() < self.mistake_probability:
            self.in_mistake = True
            self.mistake_duration = np.random.randint(5, 20)

        if self.in_mistake:
            steering += np.random.uniform(-0.3, 0.3)
            throttle *= 0.7
            self.mistake_duration -= 1
            if self.mistake_duration <= 0:
                self.in_mistake = False

        action = np.array([
            throttle,
            brake,
            steering,
            gear_normalized,
            ers_mode,
            drs
        ], dtype=np.float32)

        return np.clip(action, [0, 0, -1, 0, -1, 0], [1, 1, 1, 1, 1, 1])

    def _defensive_action(self, state: Dict, circuit) -> np.ndarray:
        """Defensive driving to protect position."""
        # Base action
        action = self._racing_action(state, circuit)

        # Modify steering to block racing line
        if self.target_car_behind is not None:
            action[2] *= 0.7  # Reduce steering aggression
            # Move slightly off racing line to block
            action[2] += np.random.uniform(-0.2, 0.2)

        return action

    def _attacking_action(self, state: Dict, circuit) -> np.ndarray:
        """Aggressive overtaking maneuver."""
        # Base action
        action = self._racing_action(state, circuit)

        # More aggressive acceleration
        action[0] = min(1.0, action[0] * 1.1)

        # Wider line for overtake
        action[2] += np.random.uniform(-0.3, 0.3)

        # Deploy ERS
        action[4] = 1.0

        return action

    def _update_opponent_awareness(
        self,
        position: np.ndarray,
        opponent_positions: Optional[List[Tuple[float, float]]]
    ):
        """Update awareness of nearby opponents."""
        if opponent_positions is None or len(opponent_positions) == 0:
            self.is_defending = False
            self.is_attacking = False
            return

        # Find closest opponent ahead and behind
        distances = [
            np.linalg.norm(np.array(opp_pos) - position)
            for opp_pos in opponent_positions
        ]

        min_dist = min(distances) if distances else float('inf')

        # Within 20 meters - racing mode
        if min_dist < 20.0:
            self.is_defending = True
            self.is_attacking = True
        else:
            self.is_defending = False
            self.is_attacking = False

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-π, π]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    @staticmethod
    def _smooth_control(current: float, previous: float, alpha: float) -> float:
        """Smooth control input."""
        return alpha * current + (1 - alpha) * previous


# Preset opponent profiles
OPPONENT_PROFILES = {
    'beginner': OpponentProfile(
        name="Rookie",
        skill_level=0.5,
        consistency=0.7,
        racing_line_std=1.5,
        reaction_time=0.3,
    ),
    'intermediate': OpponentProfile(
        name="Pro",
        skill_level=0.75,
        consistency=0.85,
        racing_line_std=0.8,
        reaction_time=0.18,
    ),
    'expert': OpponentProfile(
        name="Champion",
        skill_level=0.92,
        consistency=0.95,
        racing_line_std=0.3,
        reaction_time=0.12,
    ),
    'alien': OpponentProfile(
        name="SimRacer",
        skill_level=0.98,
        consistency=0.98,
        racing_line_std=0.1,
        reaction_time=0.08,
    ),
}


def create_opponent_grid(num_opponents: int = 19, skill_distribution: str = 'mixed'):
    """
    Create a grid of AI opponents.

    Args:
        num_opponents: Number of opponents (max 19 for F1)
        skill_distribution: 'mixed', 'similar', 'range'

    Returns:
        List of OpponentAI instances
    """
    opponents = []

    if skill_distribution == 'mixed':
        # Realistic F1 grid distribution
        skills = np.random.beta(5, 2, num_opponents)  # Skewed towards higher skill
    elif skill_distribution == 'similar':
        # All similar skill
        base_skill = 0.85
        skills = np.random.normal(base_skill, 0.05, num_opponents)
    else:  # 'range'
        # Evenly distributed
        skills = np.linspace(0.6, 0.95, num_opponents)

    skills = np.clip(skills, 0.5, 0.98)

    for i, skill in enumerate(skills):
        profile = OpponentProfile(
            name=f"AI_{i+1}",
            skill_level=skill,
            consistency=np.random.uniform(0.85, 0.98),
            aggression=np.random.uniform(0.3, 0.8),
            racing_line_std=np.random.uniform(0.3, 1.0) * (1.0 - skill),
        )
        opponents.append(OpponentAI(profile))

    return opponents
