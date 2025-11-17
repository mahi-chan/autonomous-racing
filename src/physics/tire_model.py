"""
Advanced Tire Model for F1 Simulation

Implements compound-specific tire behavior including:
- Grip levels (C1-C5 compounds)
- Temperature-dependent performance
- Degradation models
- Wear patterns
- Optimal operating windows
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class TireCompound(Enum):
    """F1 tire compounds (Pirelli 2024)."""
    C1 = "C1"  # Hardest
    C2 = "C2"
    C3 = "C3"  # Medium
    C4 = "C4"
    C5 = "C5"  # Softest
    INTERMEDIATE = "INTER"
    WET = "WET"


@dataclass
class TireCompoundCharacteristics:
    """Characteristics of each tire compound."""

    # Grip levels
    peak_grip: float  # Peak friction coefficient
    optimal_temp_min: float  # °C
    optimal_temp_max: float  # °C

    # Degradation
    wear_rate: float  # Base wear rate per km
    cliff_wear_threshold: float  # Wear level where performance cliff occurs
    cliff_grip_loss: float  # Additional grip loss after cliff

    # Thermal
    heat_generation_rate: float  # How quickly tire heats up
    cooling_rate: float  # How quickly tire cools down

    # Performance
    warmup_laps: float  # Laps needed to reach optimal temp
    optimal_pressure: float  # PSI


# Tire compound database (based on F1 data)
TIRE_COMPOUNDS_DB = {
    TireCompound.C1: TireCompoundCharacteristics(
        peak_grip=1.75,
        optimal_temp_min=100.0,
        optimal_temp_max=115.0,
        wear_rate=0.03,
        cliff_wear_threshold=0.85,
        cliff_grip_loss=0.15,
        heat_generation_rate=0.8,
        cooling_rate=1.2,
        warmup_laps=3.0,
        optimal_pressure=23.5,
    ),
    TireCompound.C2: TireCompoundCharacteristics(
        peak_grip=1.80,
        optimal_temp_min=95.0,
        optimal_temp_max=110.0,
        wear_rate=0.05,
        cliff_wear_threshold=0.80,
        cliff_grip_loss=0.18,
        heat_generation_rate=1.0,
        cooling_rate=1.1,
        warmup_laps=2.5,
        optimal_pressure=23.0,
    ),
    TireCompound.C3: TireCompoundCharacteristics(
        peak_grip=1.85,
        optimal_temp_min=90.0,
        optimal_temp_max=105.0,
        wear_rate=0.08,
        cliff_wear_threshold=0.75,
        cliff_grip_loss=0.20,
        heat_generation_rate=1.2,
        cooling_rate=1.0,
        warmup_laps=2.0,
        optimal_pressure=22.5,
    ),
    TireCompound.C4: TireCompoundCharacteristics(
        peak_grip=1.90,
        optimal_temp_min=85.0,
        optimal_temp_max=100.0,
        wear_rate=0.12,
        cliff_wear_threshold=0.70,
        cliff_grip_loss=0.22,
        heat_generation_rate=1.4,
        cooling_rate=0.9,
        warmup_laps=1.5,
        optimal_pressure=22.0,
    ),
    TireCompound.C5: TireCompoundCharacteristics(
        peak_grip=2.00,
        optimal_temp_min=80.0,
        optimal_temp_max=95.0,
        wear_rate=0.18,
        cliff_wear_threshold=0.65,
        cliff_grip_loss=0.25,
        heat_generation_rate=1.6,
        cooling_rate=0.8,
        warmup_laps=1.0,
        optimal_pressure=21.5,
    ),
    TireCompound.INTERMEDIATE: TireCompoundCharacteristics(
        peak_grip=1.50,  # Wet conditions
        optimal_temp_min=70.0,
        optimal_temp_max=90.0,
        wear_rate=0.10,
        cliff_wear_threshold=0.75,
        cliff_grip_loss=0.12,
        heat_generation_rate=0.9,
        cooling_rate=1.3,
        warmup_laps=1.5,
        optimal_pressure=22.0,
    ),
    TireCompound.WET: TireCompoundCharacteristics(
        peak_grip=1.30,  # Very wet conditions
        optimal_temp_min=60.0,
        optimal_temp_max=80.0,
        wear_rate=0.08,
        cliff_wear_threshold=0.80,
        cliff_grip_loss=0.10,
        heat_generation_rate=0.7,
        cooling_rate=1.5,
        warmup_laps=1.0,
        optimal_pressure=21.0,
    ),
}


class TireModel:
    """
    Advanced tire model simulating F1 tire behavior.

    Tracks four tires independently: FL, FR, RL, RR
    """

    def __init__(self, compound: TireCompound = TireCompound.C3):
        self.compound = compound
        self.characteristics = TIRE_COMPOUNDS_DB[compound]

        # State for each tire [FL, FR, RL, RR]
        self.temperatures = np.array([20.0, 20.0, 20.0, 20.0])  # °C
        self.wear = np.array([0.0, 0.0, 0.0, 0.0])  # 0=new, 1=destroyed
        self.pressures = np.ones(4) * self.characteristics.optimal_pressure  # PSI

        # Age tracking
        self.distance_traveled = 0.0  # km
        self.laps_completed = 0.0

        # Thermal memory (for realistic heat buildup)
        self.core_temps = np.array([20.0, 20.0, 20.0, 20.0])

    def reset(self):
        """Reset to new tires."""
        self.temperatures = np.array([20.0, 20.0, 20.0, 20.0])
        self.wear = np.array([0.0, 0.0, 0.0, 0.0])
        self.pressures = np.ones(4) * self.characteristics.optimal_pressure
        self.core_temps = np.array([20.0, 20.0, 20.0, 20.0])
        self.distance_traveled = 0.0
        self.laps_completed = 0.0

    def update(
        self,
        speed: float,
        lateral_accel: float,
        longitudinal_accel: float,
        steering_angle: float,
        brake_pressure: float,
        downforce: float,
        track_temp: float,
        ambient_temp: float,
        dt: float
    ):
        """
        Update tire state.

        Args:
            speed: Vehicle speed [m/s]
            lateral_accel: Lateral acceleration [m/s²]
            longitudinal_accel: Longitudinal acceleration [m/s²]
            steering_angle: Steering input [-1, 1]
            brake_pressure: Brake pressure [0, 1]
            downforce: Total downforce [N]
            track_temp: Track surface temperature [°C]
            ambient_temp: Air temperature [°C]
            dt: Timestep [s]
        """
        # Update distance
        self.distance_traveled += speed * dt / 1000.0  # Convert to km

        # Calculate load on each tire
        tire_loads = self._calculate_tire_loads(
            longitudinal_accel,
            lateral_accel,
            downforce
        )

        # Calculate slip for each tire
        tire_slips = self._calculate_tire_slips(
            speed,
            lateral_accel,
            steering_angle,
            brake_pressure
        )

        # Update temperatures
        self._update_temperatures(
            tire_loads,
            tire_slips,
            speed,
            track_temp,
            ambient_temp,
            dt
        )

        # Update wear
        self._update_wear(
            tire_loads,
            tire_slips,
            speed,
            dt
        )

        # Update pressures (increase with temperature)
        self._update_pressures()

    def get_grip_coefficient(self, tire_index: int = None) -> float:
        """
        Get current grip coefficient.

        Args:
            tire_index: Specific tire (0-3), or None for average

        Returns:
            Grip coefficient (mu)
        """
        if tire_index is not None:
            return self._calculate_grip(tire_index)
        else:
            # Return average grip
            grips = [self._calculate_grip(i) for i in range(4)]
            return np.mean(grips)

    def _calculate_grip(self, tire_index: int) -> float:
        """Calculate grip for a specific tire."""
        temp = self.temperatures[tire_index]
        wear = self.wear[tire_index]

        # Temperature effect
        opt_min = self.characteristics.optimal_temp_min
        opt_max = self.characteristics.optimal_temp_max

        if opt_min <= temp <= opt_max:
            temp_factor = 1.0
        elif temp < opt_min:
            # Cold tire - reduced grip
            temp_factor = 0.6 + 0.4 * (temp / opt_min)
        else:  # temp > opt_max
            # Overheated - reduced grip
            overheat = temp - opt_max
            temp_factor = 1.0 - 0.005 * overheat

        temp_factor = np.clip(temp_factor, 0.4, 1.0)

        # Wear effect
        if wear < self.characteristics.cliff_wear_threshold:
            # Linear degradation before cliff
            wear_factor = 1.0 - 0.1 * (wear / self.characteristics.cliff_wear_threshold)
        else:
            # Cliff - rapid performance loss
            base_factor = 0.9
            cliff_progress = ((wear - self.characteristics.cliff_wear_threshold) /
                            (1.0 - self.characteristics.cliff_wear_threshold))
            wear_factor = base_factor - self.characteristics.cliff_grip_loss * cliff_progress

        wear_factor = np.clip(wear_factor, 0.3, 1.0)

        # Pressure effect (simplified)
        pressure = self.pressures[tire_index]
        optimal_pressure = self.characteristics.optimal_pressure
        pressure_diff = abs(pressure - optimal_pressure)
        pressure_factor = 1.0 - 0.01 * pressure_diff
        pressure_factor = np.clip(pressure_factor, 0.85, 1.0)

        # Combined grip
        grip = (self.characteristics.peak_grip *
                temp_factor *
                wear_factor *
                pressure_factor)

        return grip

    def _calculate_tire_loads(
        self,
        long_accel: float,
        lat_accel: float,
        downforce: float
    ) -> np.ndarray:
        """
        Calculate normal load on each tire.

        Returns: [FL, FR, RL, RR] loads in Newtons
        """
        # Simplified load calculation
        # Assume 800kg car
        car_mass = 800.0
        g = 9.81

        # Static load (50/50 distribution for simplicity)
        static_load = car_mass * g / 4.0

        # Longitudinal weight transfer
        # Acceleration -> rear, Braking -> front
        long_transfer = car_mass * long_accel * 0.3 / 2.0  # 30cm CG height approx

        # Lateral weight transfer
        lat_transfer = car_mass * abs(lat_accel) * 0.3 / 2.0

        # Downforce (split 40/60 front/rear typically)
        df_front = downforce * 0.4 / 2.0
        df_rear = downforce * 0.6 / 2.0

        loads = np.array([
            static_load + long_transfer + df_front,  # FL
            static_load + long_transfer + df_front,  # FR
            static_load - long_transfer + df_rear,   # RL
            static_load - long_transfer + df_rear,   # RR
        ])

        # Add lateral transfer (simplified - just affects left/right)
        if lat_accel > 0:  # Right turn
            loads[0] += lat_transfer  # FL
            loads[2] += lat_transfer  # RL
            loads[1] -= lat_transfer  # FR
            loads[3] -= lat_transfer  # RR
        else:
            loads[0] -= lat_transfer
            loads[2] -= lat_transfer
            loads[1] += lat_transfer
            loads[3] += lat_transfer

        return np.clip(loads, 1000.0, 20000.0)

    def _calculate_tire_slips(
        self,
        speed: float,
        lat_accel: float,
        steering: float,
        brake: float
    ) -> np.ndarray:
        """
        Calculate slip for each tire.

        Returns: [FL, FR, RL, RR] slip magnitudes
        """
        # Simplified slip calculation

        # Longitudinal slip from braking/acceleration
        long_slip = brake * 0.3  # Max 30% slip

        # Lateral slip from cornering
        if speed > 1.0:
            lat_slip = abs(lat_accel) / (speed * 0.1)  # Simplified
        else:
            lat_slip = 0.0

        # Front tires have more slip when steering
        front_slip = np.sqrt(long_slip**2 + (lat_slip * (1.0 + abs(steering)))**2)
        rear_slip = np.sqrt(long_slip**2 + lat_slip**2)

        slips = np.array([front_slip, front_slip, rear_slip, rear_slip])

        return np.clip(slips, 0.0, 1.0)

    def _update_temperatures(
        self,
        loads: np.ndarray,
        slips: np.ndarray,
        speed: float,
        track_temp: float,
        ambient_temp: float,
        dt: float
    ):
        """Update tire temperatures."""
        # Heat generation from slip
        heat_gen = (slips * speed * loads *
                   self.characteristics.heat_generation_rate * 0.0001)

        # Heat from track
        track_heat = (track_temp - self.temperatures) * 0.1 * dt

        # Cooling from air
        cooling = ((self.temperatures - ambient_temp) *
                  self.characteristics.cooling_rate * dt)

        # Update surface temperature
        self.temperatures += heat_gen * dt + track_heat - cooling
        self.temperatures = np.clip(self.temperatures, ambient_temp, 150.0)

        # Core temperature (slower to change)
        core_diff = (self.temperatures - self.core_temps) * 0.1 * dt
        self.core_temps += core_diff

    def _update_wear(
        self,
        loads: np.ndarray,
        slips: np.ndarray,
        speed: float,
        dt: float
    ):
        """Update tire wear."""
        # Wear rate depends on slip, load, and speed
        wear_rate = (slips * (loads / 5000.0) * (speed / 50.0) *
                    self.characteristics.wear_rate * 0.0001)

        # Increased wear if overheated
        overheat_factor = np.where(
            self.temperatures > self.characteristics.optimal_temp_max + 10,
            2.0,
            1.0
        )

        wear_rate *= overheat_factor

        self.wear += wear_rate * dt
        self.wear = np.clip(self.wear, 0.0, 1.0)

    def _update_pressures(self):
        """Update tire pressures based on temperature."""
        # Simplified: pressure increases ~0.1 PSI per 10°C
        temp_increase = self.temperatures - 20.0  # Baseline 20°C
        pressure_increase = temp_increase * 0.01

        self.pressures = self.characteristics.optimal_pressure + pressure_increase
        self.pressures = np.clip(self.pressures, 15.0, 30.0)

    def get_state(self) -> Dict:
        """Get tire state for observation/logging."""
        return {
            'compound': self.compound.value,
            'temperatures': self.temperatures.copy(),
            'wear': self.wear.copy(),
            'pressures': self.pressures.copy(),
            'distance_km': self.distance_traveled,
            'laps': self.laps_completed,
            'grip_coefficients': np.array([self._calculate_grip(i) for i in range(4)]),
        }

    def is_critical_wear(self) -> bool:
        """Check if any tire has critical wear."""
        return np.any(self.wear > 0.95)

    def get_wear_percentage(self) -> float:
        """Get average wear as percentage."""
        return np.mean(self.wear) * 100.0
