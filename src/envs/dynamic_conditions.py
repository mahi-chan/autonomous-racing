"""
Dynamic Environmental Conditions

Simulates changing conditions during a race:
- Weather (dry, wet, changing)
- Track temperature
- Track evolution (rubber buildup)
- Time of day
- Tire degradation factors
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional

from src.tracks.circuit import Circuit


class WeatherCondition(Enum):
    """Weather conditions."""
    DRY = 0
    LIGHT_RAIN = 1
    HEAVY_RAIN = 2
    CHANGING = 3  # Unpredictable


@dataclass
class ConditionsConfig:
    """Configuration for dynamic conditions."""

    # Weather
    initial_weather: WeatherCondition = WeatherCondition.DRY
    weather_change_probability: float = 0.0001  # Per step
    rain_intensity_variation: float = 0.1

    # Temperature
    initial_air_temp: float = 20.0  # °C
    initial_track_temp: float = 30.0  # °C
    temp_variation_range: float = 5.0  # ±°C
    temp_change_rate: float = 0.01  # °C per second

    # Track evolution
    rubber_buildup_rate: float = 0.0001  # Per car passing
    max_rubber_buildup: float = 0.3  # Max 30% grip improvement
    cleaning_effect: bool = True  # Rain cleans the track

    # Time of day
    time_of_day: float = 14.0  # 2 PM
    time_progression: bool = False  # Real-time progression


class DynamicConditions:
    """
    Manages dynamic environmental conditions.

    Updates weather, track conditions, temperature, etc. over time.
    """

    def __init__(
        self,
        circuit: Circuit,
        weather: WeatherCondition = WeatherCondition.DRY,
        config: Optional[ConditionsConfig] = None
    ):
        self.circuit = circuit
        self.config = config or ConditionsConfig()

        # Initialize conditions
        self.weather = weather
        self.air_temperature = self.config.initial_air_temp
        self.track_temperature = self.config.initial_track_temp
        self.time_of_day = self.config.time_of_day  # 24-hour format

        # Track-specific
        self.track_wetness = 0.0 if weather == WeatherCondition.DRY else 0.5
        self.rubber_buildup = 0.0  # 0=clean, 1=maximum rubber

        # Rain tracking
        self.rain_intensity = 0.0  # 0=dry, 1=maximum rain
        if weather == WeatherCondition.LIGHT_RAIN:
            self.rain_intensity = 0.3
        elif weather == WeatherCondition.HEAVY_RAIN:
            self.rain_intensity = 0.8

        # Session time
        self.session_time = 0.0  # seconds

    def reset(
        self,
        weather: Optional[WeatherCondition] = None,
        air_temp: Optional[float] = None,
        track_temp: Optional[float] = None
    ):
        """Reset conditions to initial state."""
        if weather is not None:
            self.weather = weather

        self.air_temperature = air_temp or self.config.initial_air_temp
        self.track_temperature = track_temp or self.config.initial_track_temp
        self.time_of_day = self.config.time_of_day

        self.track_wetness = 0.0 if self.weather == WeatherCondition.DRY else 0.5
        self.rubber_buildup = 0.0
        self.session_time = 0.0

        # Reset rain intensity
        if self.weather == WeatherCondition.LIGHT_RAIN:
            self.rain_intensity = 0.3
        elif self.weather == WeatherCondition.HEAVY_RAIN:
            self.rain_intensity = 0.8
        else:
            self.rain_intensity = 0.0

    def step(self, dt: float):
        """
        Update conditions for one timestep.

        Args:
            dt: Timestep in seconds
        """
        self.session_time += dt

        # Update weather
        self._update_weather(dt)

        # Update temperatures
        self._update_temperatures(dt)

        # Update track conditions
        self._update_track_evolution(dt)

        # Update time of day
        if self.config.time_progression:
            self.time_of_day += dt / 3600.0  # Convert to hours
            if self.time_of_day >= 24.0:
                self.time_of_day -= 24.0

    def _update_weather(self, dt: float):
        """Update weather conditions."""
        if self.weather == WeatherCondition.CHANGING:
            # Random weather changes
            if np.random.random() < self.config.weather_change_probability * dt:
                # Transition weather
                if self.rain_intensity < 0.1:
                    # Start raining
                    self.rain_intensity = 0.3
                elif self.rain_intensity > 0.7:
                    # Reduce rain
                    self.rain_intensity = 0.4
                else:
                    # Random change
                    self.rain_intensity += np.random.uniform(-0.2, 0.2)

            # Gradual rain intensity variation
            variation = np.random.normal(0, self.config.rain_intensity_variation * dt)
            self.rain_intensity += variation
            self.rain_intensity = np.clip(self.rain_intensity, 0.0, 1.0)

        # Update track wetness based on rain
        if self.rain_intensity > 0.1:
            # Track getting wet
            wetness_increase = self.rain_intensity * 0.1 * dt
            self.track_wetness = min(1.0, self.track_wetness + wetness_increase)

            # Update weather enum
            if self.rain_intensity < 0.5:
                self.weather = WeatherCondition.LIGHT_RAIN
            else:
                self.weather = WeatherCondition.HEAVY_RAIN
        else:
            # Track drying
            drying_rate = 0.05 * dt  # Slow drying
            self.track_wetness = max(0.0, self.track_wetness - drying_rate)

            if self.track_wetness < 0.1:
                self.weather = WeatherCondition.DRY

    def _update_temperatures(self, dt: float):
        """Update air and track temperatures."""
        # Track temperature follows air temperature with some lag
        # Also affected by sun/clouds (simplified)

        # Natural variation
        air_temp_change = np.random.normal(0, self.config.temp_change_rate * dt)
        self.air_temperature += air_temp_change
        self.air_temperature = np.clip(
            self.air_temperature,
            self.config.initial_air_temp - self.config.temp_variation_range,
            self.config.initial_air_temp + self.config.temp_variation_range
        )

        # Track temperature tracks air temperature
        temp_diff = self.air_temperature - self.track_temperature
        track_temp_change = temp_diff * 0.01 * dt  # Slow convergence

        # Track is typically hotter than air (sun heating)
        if self.weather == WeatherCondition.DRY:
            track_temp_offset = 10.0  # Track is 10°C hotter when dry
        else:
            track_temp_offset = 2.0  # Much cooler in rain

        target_track_temp = self.air_temperature + track_temp_offset
        self.track_temperature += (target_track_temp - self.track_temperature) * 0.01 * dt

        # Rain cools track quickly
        if self.rain_intensity > 0.1:
            cooling = self.rain_intensity * 0.5 * dt
            self.track_temperature -= cooling

        self.track_temperature = np.clip(self.track_temperature, 10.0, 60.0)

    def _update_track_evolution(self, dt: float):
        """Update track evolution (rubber buildup)."""
        if self.weather == WeatherCondition.DRY:
            # Rubber builds up during dry running
            self.rubber_buildup += self.config.rubber_buildup_rate * dt
            self.rubber_buildup = min(self.rubber_buildup, self.config.max_rubber_buildup)
        else:
            # Rain cleans the track
            if self.config.cleaning_effect:
                cleaning_rate = self.rain_intensity * 0.01 * dt
                self.rubber_buildup = max(0.0, self.rubber_buildup - cleaning_rate)

    def get_grip_multiplier(self) -> float:
        """
        Calculate overall grip multiplier based on conditions.

        Returns:
            Grip multiplier (0.5 to 1.3)
        """
        grip = 1.0

        # Weather effect
        if self.weather == WeatherCondition.DRY:
            grip *= 1.0
        elif self.weather == WeatherCondition.LIGHT_RAIN:
            grip *= 0.85
        elif self.weather == WeatherCondition.HEAVY_RAIN:
            grip *= 0.70

        # Track wetness effect (more detailed than weather alone)
        if self.track_wetness > 0.1:
            wetness_penalty = self.track_wetness * 0.3
            grip *= (1.0 - wetness_penalty)

        # Rubber buildup (improves grip when dry)
        if self.weather == WeatherCondition.DRY:
            grip *= (1.0 + self.rubber_buildup)

        # Track temperature effect
        # Optimal: 30-40°C
        if 30.0 <= self.track_temperature <= 40.0:
            temp_mult = 1.0
        elif self.track_temperature < 20.0:
            # Cold track
            temp_mult = 0.9 + (self.track_temperature - 20.0) * 0.005
        elif self.track_temperature > 50.0:
            # Very hot track
            temp_mult = 1.0 - (self.track_temperature - 50.0) * 0.01
        else:
            temp_mult = 1.0

        temp_mult = np.clip(temp_mult, 0.85, 1.05)
        grip *= temp_mult

        return np.clip(grip, 0.5, 1.3)

    def get_optimal_tire_compound(self) -> str:
        """
        Recommend optimal tire compound for current conditions.

        Returns:
            Tire compound name
        """
        if self.weather in [WeatherCondition.LIGHT_RAIN, WeatherCondition.HEAVY_RAIN]:
            if self.rain_intensity > 0.6:
                return "WET"
            else:
                return "INTERMEDIATE"

        # Dry conditions - depends on temperature
        if self.track_temperature < 25.0:
            return "C4"  # Soft for cool conditions
        elif self.track_temperature > 40.0:
            return "C2"  # Hard for hot conditions
        else:
            return "C3"  # Medium

    def get_state(self) -> Dict:
        """Get current conditions state."""
        return {
            'weather': self.weather,
            'air_temperature': self.air_temperature,
            'track_temperature': self.track_temperature,
            'track_wetness': self.track_wetness,
            'rubber_buildup': self.rubber_buildup,
            'rain_intensity': self.rain_intensity,
            'time_of_day': self.time_of_day,
            'grip_multiplier': self.get_grip_multiplier(),
            'recommended_tire': self.get_optimal_tire_compound(),
        }

    def apply_weather_preset(self, preset: str):
        """
        Apply a weather preset.

        Args:
            preset: One of 'dry_hot', 'dry_cold', 'wet', 'changing'
        """
        if preset == 'dry_hot':
            self.weather = WeatherCondition.DRY
            self.air_temperature = 30.0
            self.track_temperature = 45.0
            self.track_wetness = 0.0
            self.rain_intensity = 0.0

        elif preset == 'dry_cold':
            self.weather = WeatherCondition.DRY
            self.air_temperature = 12.0
            self.track_temperature = 20.0
            self.track_wetness = 0.0
            self.rain_intensity = 0.0

        elif preset == 'wet':
            self.weather = WeatherCondition.HEAVY_RAIN
            self.air_temperature = 15.0
            self.track_temperature = 18.0
            self.track_wetness = 0.8
            self.rain_intensity = 0.7

        elif preset == 'changing':
            self.weather = WeatherCondition.CHANGING
            self.air_temperature = 18.0
            self.track_temperature = 25.0
            self.track_wetness = 0.3
            self.rain_intensity = 0.2

    def get_telemetry(self) -> Dict[str, float]:
        """Get conditions telemetry."""
        return {
            'weather_code': self.weather.value,
            'air_temp_c': self.air_temperature,
            'track_temp_c': self.track_temperature,
            'track_wetness': self.track_wetness,
            'rain_intensity': self.rain_intensity,
            'rubber_buildup': self.rubber_buildup,
            'grip_multiplier': self.get_grip_multiplier(),
            'time_of_day': self.time_of_day,
        }

    def __repr__(self) -> str:
        return (f"Conditions(weather={self.weather.name}, "
                f"air_temp={self.air_temperature:.1f}°C, "
                f"track_temp={self.track_temperature:.1f}°C, "
                f"grip={self.get_grip_multiplier():.2f}x)")
