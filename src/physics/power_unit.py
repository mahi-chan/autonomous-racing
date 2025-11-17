"""
F1 Power Unit Model

Simulates the complete F1 hybrid power unit:
- ICE (Internal Combustion Engine) - V6 1.6L turbocharged
- MGU-K (Motor Generator Unit - Kinetic) - 120kW
- MGU-H (Motor Generator Unit - Heat) - from turbocharger
- Energy Store (Battery)
- Control strategies
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


class ERSMode(Enum):
    """ERS deployment modes."""
    HARVEST_MAX = -1.0  # Maximum harvesting
    BALANCED = 0.0  # Neutral
    DEPLOY_MEDIUM = 0.5  # Medium deployment
    DEPLOY_MAX = 1.0  # Maximum deployment (overtake mode)


@dataclass
class PowerUnitConfig:
    """Power unit configuration (2024 F1 regulations)."""

    # ICE specifications
    ice_max_power: float = 550000.0  # W (550 kW / ~740 hp)
    ice_max_rpm: float = 15000.0  # RPM (regulated limit)
    ice_idle_rpm: float = 5000.0  # Minimum RPM
    ice_optimal_rpm: float = 12000.0  # Peak power RPM

    # Fuel
    max_fuel_flow_rate: float = 100.0  # kg/hour (FIA limit)
    fuel_energy_density: float = 44.0e6  # J/kg (MJ/kg)
    ice_efficiency: float = 0.50  # 50% thermal efficiency (modern F1)

    # MGU-K (kinetic recovery)
    mguk_max_power_deploy: float = 120000.0  # W (120 kW / ~160 hp)
    mguk_max_power_harvest: float = 120000.0  # W
    mguk_max_energy_per_lap: float = 4.0e6  # J (4 MJ per lap - deploy limit)
    mguk_max_harvest_per_lap: float = 2.0e6  # J (2 MJ from braking)

    # MGU-H (heat recovery from turbo)
    mguh_max_power: float = 100000.0  # W (no regulatory limit, but practical)
    mguh_recovery_efficiency: float = 0.30  # 30% of exhaust energy

    # Energy Store (battery)
    es_max_capacity: float = 4.0e6  # J (4 MJ max storage)
    es_charge_efficiency: float = 0.95  # 95% efficient
    es_discharge_efficiency: float = 0.95

    # Turbocharger
    turbo_max_boost: float = 3.5  # bar (absolute pressure)
    turbo_lag: float = 0.2  # seconds
    turbo_efficiency: float = 0.70

    def __post_init__(self):
        """Validate configuration."""
        assert self.ice_max_power > 0
        assert self.mguk_max_power_deploy <= 120000.0  # FIA limit
        assert self.es_max_capacity <= 4.0e6  # FIA limit


class PowerUnit:
    """
    Complete F1 hybrid power unit simulation.

    Manages:
    - ICE power output
    - ERS deployment and harvesting
    - Fuel consumption
    - Turbo boost
    - Thermal management
    """

    def __init__(self, config: Optional[PowerUnitConfig] = None):
        self.config = config or PowerUnitConfig()

        # State variables
        self.reset()

    def reset(self):
        """Reset power unit to initial state."""
        # ICE state
        self.engine_rpm = self.config.ice_idle_rpm
        self.engine_temp = 90.0  # °C (operating temperature ~90-100°C)

        # Fuel
        self.fuel_used_lap = 0.0  # kg
        self.fuel_flow_rate = 0.0  # kg/hour

        # ERS state
        self.es_energy = self.config.es_max_capacity  # Start with full battery
        self.mguk_energy_deployed_lap = 0.0  # J
        self.mguk_energy_harvested_lap = 0.0  # J

        # Turbo state
        self.turbo_boost_pressure = 1.0  # bar (atmospheric)
        self.turbo_rpm = 0.0

        # Temperatures
        self.turbo_temp = 200.0  # °C
        self.mgu_temp = 60.0  # °C

        # Lap tracking
        self.lap_started = False

    def calculate_power(
        self,
        throttle: float,
        rpm: float,
        ers_mode: float = 0.0,
        brake_pressure: float = 0.0,
        speed: float = 0.0,
        dt: float = 0.01
    ) -> Dict[str, float]:
        """
        Calculate total power output and update state.

        Args:
            throttle: Throttle position [0, 1]
            rpm: Wheel RPM (used to calculate engine RPM with gear ratio)
            ers_mode: ERS mode [-1 to 1] (-1=harvest, 0=neutral, 1=deploy)
            brake_pressure: Brake pressure [0, 1] (for MGU-K harvesting)
            speed: Vehicle speed [m/s]
            dt: Timestep [s]

        Returns:
            Dictionary with power components
        """
        self.engine_rpm = rpm

        # === ICE POWER ===
        ice_power = self._calculate_ice_power(throttle)

        # === MGU-K ===
        mguk_power = self._calculate_mguk_power(ers_mode, brake_pressure, dt)

        # === MGU-H ===
        mguh_power = self._calculate_mguh_power(throttle, dt)

        # === TOTAL POWER ===
        total_power = ice_power + mguk_power

        # === FUEL CONSUMPTION ===
        self._update_fuel_consumption(throttle, ice_power, dt)

        # === THERMAL MANAGEMENT ===
        self._update_temperatures(throttle, ice_power, dt)

        # === TURBO ===
        self._update_turbo(throttle, dt)

        return {
            'total_power': total_power,
            'ice_power': ice_power,
            'mguk_power': mguk_power,
            'mguh_power': mguh_power,
            'ers_mode': ers_mode,
            'fuel_flow_rate': self.fuel_flow_rate,
            'es_energy': self.es_energy,
            'es_soc': self.es_energy / self.config.es_max_capacity,  # State of charge
        }

    def _calculate_ice_power(self, throttle: float) -> float:
        """Calculate ICE power output."""
        # RPM-based power curve (simplified)
        rpm = self.engine_rpm

        if rpm < self.config.ice_idle_rpm:
            return 0.0

        if rpm > self.config.ice_max_rpm:
            # Over-rev protection
            return self.config.ice_max_power * 0.3

        # Power curve: peaks at optimal RPM
        rpm_normalized = (rpm - self.config.ice_idle_rpm) / (
            self.config.ice_max_rpm - self.config.ice_idle_rpm
        )

        # Simplified power curve (bell-shaped, peaks at ~80% of max RPM)
        optimal_normalized = (self.config.ice_optimal_rpm - self.config.ice_idle_rpm) / (
            self.config.ice_max_rpm - self.config.ice_idle_rpm
        )

        # Gaussian-like curve
        power_factor = np.exp(-((rpm_normalized - optimal_normalized) ** 2) / 0.1)

        # Turbo boost effect
        boost_multiplier = 0.7 + 0.3 * (self.turbo_boost_pressure / self.config.turbo_max_boost)

        # Throttle response
        ice_power = (self.config.ice_max_power *
                    power_factor *
                    boost_multiplier *
                    throttle)

        return ice_power

    def _calculate_mguk_power(
        self,
        ers_mode: float,
        brake_pressure: float,
        dt: float
    ) -> float:
        """
        Calculate MGU-K power (deployment or harvesting).

        Returns:
            Power in Watts (positive for deployment, negative for harvesting)
        """
        if ers_mode > 0.0:
            # DEPLOYMENT mode
            # Check if we have energy and haven't exceeded lap limit
            if (self.es_energy > 100.0 and
                self.mguk_energy_deployed_lap < self.config.mguk_max_energy_per_lap):

                # Power output
                deploy_power = self.config.mguk_max_power_deploy * ers_mode

                # Energy consumed from battery
                energy_consumed = deploy_power * dt / self.config.es_discharge_efficiency

                # Check limits
                energy_available = min(
                    self.es_energy,
                    self.config.mguk_max_energy_per_lap - self.mguk_energy_deployed_lap
                )

                actual_energy = min(energy_consumed, energy_available)
                actual_power = actual_energy / dt * self.config.es_discharge_efficiency

                # Update state
                self.es_energy -= actual_energy
                self.mguk_energy_deployed_lap += actual_energy

                return actual_power
            else:
                return 0.0

        elif ers_mode < 0.0 or brake_pressure > 0.3:
            # HARVESTING mode (from braking)
            if (brake_pressure > 0.3 and
                self.mguk_energy_harvested_lap < self.config.mguk_max_harvest_per_lap and
                self.es_energy < self.config.es_max_capacity):

                # Harvest power (limited by brake pressure and MGU-K limit)
                harvest_power = min(
                    self.config.mguk_max_power_harvest * brake_pressure,
                    self.config.mguk_max_power_harvest
                )

                # Energy recovered (with efficiency loss)
                energy_recovered = harvest_power * dt * self.config.es_charge_efficiency

                # Check limits
                energy_available = min(
                    energy_recovered,
                    self.config.mguk_max_harvest_per_lap - self.mguk_energy_harvested_lap,
                    self.config.es_max_capacity - self.es_energy
                )

                # Update state
                self.es_energy += energy_available
                self.mguk_energy_harvested_lap += energy_recovered

                # Return negative power (it's being harvested, not delivered)
                return -harvest_power

        return 0.0

    def _calculate_mguh_power(self, throttle: float, dt: float) -> float:
        """
        Calculate MGU-H power (always harvesting from exhaust).

        MGU-H has no deployment limit - can harvest unlimited energy.
        """
        if throttle < 0.1:
            return 0.0

        # Exhaust energy available (roughly proportional to fuel flow)
        exhaust_energy_rate = (self.fuel_flow_rate / 3600.0 *
                              self.config.fuel_energy_density *
                              (1.0 - self.config.ice_efficiency))

        # MGU-H harvests portion of exhaust energy
        harvest_rate = min(
            exhaust_energy_rate * self.config.mguh_recovery_efficiency,
            self.config.mguh_max_power
        )

        # Charge battery (if space available)
        if self.es_energy < self.config.es_max_capacity:
            energy_recovered = harvest_rate * dt * self.config.es_charge_efficiency
            energy_to_store = min(
                energy_recovered,
                self.config.es_max_capacity - self.es_energy
            )
            self.es_energy += energy_to_store

        return harvest_rate

    def _update_fuel_consumption(self, throttle: float, ice_power: float, dt: float):
        """Update fuel consumption."""
        if throttle < 0.01:
            self.fuel_flow_rate = 0.0
            return

        # Fuel flow based on power output and efficiency
        # FIA limit: 100 kg/hour max
        required_flow = (ice_power / self.config.ice_efficiency /
                        self.config.fuel_energy_density * 3600.0)

        # Apply FIA limit
        self.fuel_flow_rate = min(required_flow, self.config.max_fuel_flow_rate)

        # Update lap fuel usage
        fuel_used = (self.fuel_flow_rate / 3600.0) * dt
        self.fuel_used_lap += fuel_used

    def _update_temperatures(self, throttle: float, power: float, dt: float):
        """Update component temperatures."""
        # Engine temperature
        # Heat generation from power output
        heat_gen = power * (1.0 - self.config.ice_efficiency) * 0.0001

        # Cooling
        cooling = (self.engine_temp - 90.0) * 0.1 * dt

        self.engine_temp += heat_gen * dt - cooling
        self.engine_temp = np.clip(self.engine_temp, 60.0, 130.0)

        # Turbo temperature (very hot - up to 1000°C, but we model turbine inlet)
        if throttle > 0.1:
            self.turbo_temp += throttle * 100.0 * dt
        self.turbo_temp -= 5.0 * dt  # Cooling
        self.turbo_temp = np.clip(self.turbo_temp, 100.0, 950.0)

        # MGU temperature
        # Heats up during deployment/harvesting
        self.mgu_temp += abs(self.mguk_energy_deployed_lap / 1e6) * 0.5 * dt
        self.mgu_temp -= 1.0 * dt  # Cooling
        self.mgu_temp = np.clip(self.mgu_temp, 40.0, 120.0)

    def _update_turbo(self, throttle: float, dt: float):
        """Update turbocharger state."""
        # Target boost based on throttle and RPM
        if throttle > 0.1 and self.engine_rpm > self.config.ice_idle_rpm:
            rpm_factor = min(1.0, self.engine_rpm / self.config.ice_optimal_rpm)
            target_boost = 1.0 + (self.config.turbo_max_boost - 1.0) * throttle * rpm_factor
        else:
            target_boost = 1.0  # Atmospheric

        # Turbo lag - exponential approach to target
        tau = self.config.turbo_lag  # Time constant
        self.turbo_boost_pressure += (target_boost - self.turbo_boost_pressure) * dt / tau

        self.turbo_boost_pressure = np.clip(
            self.turbo_boost_pressure,
            1.0,
            self.config.turbo_max_boost
        )

        # Turbo RPM (very high - up to 125,000 RPM in F1)
        self.turbo_rpm = (self.turbo_boost_pressure - 1.0) / (
            self.config.turbo_max_boost - 1.0
        ) * 125000.0

    def on_lap_complete(self):
        """Reset lap-based counters."""
        self.mguk_energy_deployed_lap = 0.0
        self.mguk_energy_harvested_lap = 0.0
        self.fuel_used_lap = 0.0

    def get_ers_recommendation(
        self,
        lap_progress: float,
        next_sector_type: str = "mixed"
    ) -> float:
        """
        Recommend ERS mode based on lap progress and upcoming sector.

        Args:
            lap_progress: Progress through lap [0, 1]
            next_sector_type: 'straight', 'corners', 'mixed'

        Returns:
            Recommended ERS mode [-1, 1]
        """
        soc = self.es_energy / self.config.es_max_capacity

        # Energy management strategy
        if next_sector_type == "straight":
            # Deploy on straights for overtaking/defense
            if soc > 0.3:
                return 1.0  # Full deployment
            elif soc > 0.1:
                return 0.5  # Partial deployment
            else:
                return 0.0  # Save energy

        elif next_sector_type == "corners":
            # Harvest in braking zones
            return -0.5

        else:  # mixed
            # Balanced approach
            if soc > 0.7:
                return 0.5  # Deploy excess
            elif soc < 0.3:
                return -0.5  # Harvest more
            else:
                return 0.0  # Neutral

    def get_telemetry(self) -> Dict[str, float]:
        """Get power unit telemetry."""
        return {
            'engine_rpm': self.engine_rpm,
            'engine_temp': self.engine_temp,
            'turbo_boost_bar': self.turbo_boost_pressure,
            'turbo_rpm': self.turbo_rpm,
            'turbo_temp': self.turbo_temp,
            'fuel_flow_kg_h': self.fuel_flow_rate,
            'fuel_used_lap_kg': self.fuel_used_lap,
            'es_energy_mj': self.es_energy / 1e6,
            'es_soc_percent': (self.es_energy / self.config.es_max_capacity) * 100,
            'mguk_deployed_lap_mj': self.mguk_energy_deployed_lap / 1e6,
            'mguk_harvested_lap_mj': self.mguk_energy_harvested_lap / 1e6,
            'mgu_temp': self.mgu_temp,
        }

    def check_reliability(self) -> Dict[str, bool]:
        """Check for component reliability issues."""
        issues = {
            'engine_overheat': self.engine_temp > 120.0,
            'turbo_overheat': self.turbo_temp > 900.0,
            'mgu_overheat': self.mgu_temp > 110.0,
            'over_rev': self.engine_rpm > self.config.ice_max_rpm,
            'fuel_flow_violation': self.fuel_flow_rate > self.config.max_fuel_flow_rate,
        }

        return issues
