"""
F1 Car Physics Model

Comprehensive physics simulation of an F1 car including:
- Aerodynamics (downforce, drag, DRS)
- Tire model (grip, degradation, temperature)
- Power unit (ICE + ERS)
- Weight distribution and fuel load
- Suspension and ride height
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import yaml


@dataclass
class F1CarConfig:
    """F1 car configuration parameters based on 2024 regulations."""

    # Mass and inertia
    min_weight: float = 798.0  # kg (with driver, 2024 regulations)
    weight_distribution_front: float = 0.46  # 46% front
    moment_of_inertia_yaw: float = 1200.0  # kg⋅m²
    moment_of_inertia_pitch: float = 1800.0  # kg⋅m²
    moment_of_inertia_roll: float = 400.0  # kg⋅m²

    # Dimensions
    wheelbase: float = 3.6  # meters
    track_width_front: float = 1.6  # meters
    track_width_rear: float = 1.55  # meters
    center_of_gravity_height: float = 0.30  # meters

    # Aerodynamics
    frontal_area: float = 1.4  # m²
    drag_coefficient: float = 0.70  # Cd
    downforce_coefficient: float = 3.5  # Cl
    drs_drag_reduction: float = 0.15  # 15% reduction
    drs_downforce_reduction: float = 0.10  # 10% reduction

    # Aero balance (% front)
    aero_balance_front: float = 0.38  # 38% front downforce

    # Power unit
    max_power_ice: float = 550000.0  # W (550 kW / ~740 hp)
    max_power_ers_deploy: float = 120000.0  # W (120 kW / ~160 hp)
    ers_max_energy: float = 4.0e6  # J (4 MJ per lap)
    ers_max_recovery: float = 2.0e6  # J (2 MJ per lap)
    ers_max_deploy_rate: float = 120000.0  # W
    ers_max_recovery_rate: float = 120000.0  # W

    # Fuel
    max_fuel_load: float = 110.0  # kg
    fuel_consumption_rate: float = 100.0  # kg/hour at full power

    # Brakes
    max_brake_force: float = 20000.0  # N
    brake_bias_front: float = 0.58  # 58% front
    brake_cooling_factor: float = 0.95

    # Tires
    tire_radius: float = 0.36  # meters
    tire_width_front: float = 0.305  # meters
    tire_width_rear: float = 0.405  # meters

    # Gear ratios (8-speed gearbox)
    gear_ratios: Tuple[float, ...] = (3.2, 2.4, 1.9, 1.5, 1.25, 1.05, 0.92, 0.82)
    final_drive: float = 3.0

    # Differential
    differential_lock_accel: float = 0.45  # 45% locking on acceleration
    differential_lock_decel: float = 0.30  # 30% locking on deceleration

    # DRS zones
    drs_available: bool = True

    @classmethod
    def from_yaml(cls, filepath: str) -> 'F1CarConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, filepath: str):
        """Save configuration to YAML file."""
        config_dict = self.__dict__
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


class F1Car:
    """
    High-fidelity F1 car physics model.

    State vector:
        - position: (x, y) [m]
        - velocity: (vx, vy) [m/s]
        - heading: θ [rad]
        - angular_velocity: ω [rad/s]
        - wheel_speeds: [FL, FR, RL, RR] [rad/s]
        - fuel_load: [kg]
        - ers_energy: [J]
        - tire_temps: [FL, FR, RL, RR] [°C]
        - tire_wear: [FL, FR, RL, RR] [0-1]

    Control inputs:
        - throttle: [0, 1]
        - brake: [0, 1]
        - steering: [-1, 1] (normalized angle)
        - gear: [0-8] (0=neutral, 1-8=gears)
        - ers_mode: [-1, 0, 1] (harvest, neutral, deploy)
        - drs_active: [0, 1]
    """

    def __init__(self, config: Optional[F1CarConfig] = None):
        self.config = config or F1CarConfig()

        # Initialize state
        self.reset_state()

        # Physics constants
        self.g = 9.81  # m/s²
        self.air_density = 1.225  # kg/m³

    def reset_state(self):
        """Reset car to initial state."""
        self.position = np.array([0.0, 0.0])  # x, y
        self.velocity = np.array([0.0, 0.0])  # vx, vy (local frame)
        self.heading = 0.0  # radians
        self.angular_velocity = 0.0  # rad/s

        # Wheel speeds (rad/s)
        self.wheel_speeds = np.array([0.0, 0.0, 0.0, 0.0])  # FL, FR, RL, RR

        # Fuel and energy
        self.fuel_load = self.config.max_fuel_load
        self.ers_energy = self.config.ers_max_energy
        self.ers_deployed = 0.0
        self.ers_recovered = 0.0

        # Tire state
        self.tire_temps = np.array([80.0, 80.0, 80.0, 80.0])  # °C (optimal ~90-110°C)
        self.tire_wear = np.array([0.0, 0.0, 0.0, 0.0])  # 0=new, 1=completely worn

        # Brake temps
        self.brake_temps = np.array([200.0, 200.0, 200.0, 200.0])  # °C

        # Current gear
        self.current_gear = 1

    def get_state(self) -> Dict[str, np.ndarray]:
        """Get current state as dictionary."""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'heading': self.heading,
            'angular_velocity': self.angular_velocity,
            'wheel_speeds': self.wheel_speeds.copy(),
            'fuel_load': self.fuel_load,
            'ers_energy': self.ers_energy,
            'tire_temps': self.tire_temps.copy(),
            'tire_wear': self.tire_wear.copy(),
            'brake_temps': self.brake_temps.copy(),
            'current_gear': self.current_gear,
        }

    def step(self, controls: Dict[str, float], dt: float = 0.01) -> Dict[str, np.ndarray]:
        """
        Advance physics simulation by one timestep.

        Args:
            controls: Dictionary with keys: throttle, brake, steering, gear, ers_mode, drs_active
            dt: Time step in seconds

        Returns:
            Updated state dictionary
        """
        # Extract controls
        throttle = np.clip(controls.get('throttle', 0.0), 0.0, 1.0)
        brake = np.clip(controls.get('brake', 0.0), 0.0, 1.0)
        steering = np.clip(controls.get('steering', 0.0), -1.0, 1.0)
        gear_cmd = int(controls.get('gear', self.current_gear))
        ers_mode = np.clip(controls.get('ers_mode', 0.0), -1.0, 1.0)
        drs_active = bool(controls.get('drs_active', False))

        # Update gear
        self.current_gear = np.clip(gear_cmd, 0, len(self.config.gear_ratios))

        # Calculate speed
        speed = np.linalg.norm(self.velocity)

        # === AERODYNAMICS ===
        aero_forces = self._calculate_aero_forces(speed, drs_active)
        downforce = aero_forces['downforce']
        drag = aero_forces['drag']

        # === TIRE FORCES ===
        tire_forces = self._calculate_tire_forces(
            self.velocity,
            steering,
            downforce,
            brake
        )

        # === POWER UNIT ===
        engine_force = self._calculate_engine_force(
            throttle,
            speed,
            self.current_gear,
            ers_mode
        )

        # === BRAKE FORCES ===
        brake_force = self._calculate_brake_force(brake, speed)

        # === TOTAL FORCES ===
        # Longitudinal force (local x-axis)
        fx_total = engine_force - drag - brake_force + tire_forces['fx']

        # Lateral force (local y-axis)
        fy_total = tire_forces['fy']

        # === DYNAMICS ===
        # Current mass (including fuel)
        current_mass = self.config.min_weight + self.fuel_load

        # Accelerations in local frame
        ax = fx_total / current_mass
        ay = fy_total / current_mass

        # Update velocities
        self.velocity[0] += ax * dt  # vx
        self.velocity[1] += ay * dt  # vy

        # Yaw moment from steering and tire forces
        yaw_moment = tire_forces['moment']
        angular_accel = yaw_moment / self.config.moment_of_inertia_yaw
        self.angular_velocity += angular_accel * dt

        # Update heading
        self.heading += self.angular_velocity * dt
        self.heading = self._normalize_angle(self.heading)

        # Update position (convert to global frame)
        global_velocity = self._local_to_global(self.velocity, self.heading)
        self.position += global_velocity * dt

        # === UPDATE SUBSYSTEMS ===
        self._update_fuel(throttle, dt)
        self._update_ers(ers_mode, brake, speed, dt)
        self._update_tires(speed, steering, brake, dt)
        self._update_brakes(brake, speed, dt)
        self._update_wheel_speeds(speed, dt)

        return self.get_state()

    def _calculate_aero_forces(self, speed: float, drs_active: bool) -> Dict[str, float]:
        """Calculate aerodynamic forces."""
        cd = self.config.drag_coefficient
        cl = self.config.downforce_coefficient

        if drs_active and self.config.drs_available:
            cd *= (1.0 - self.config.drs_drag_reduction)
            cl *= (1.0 - self.config.drs_downforce_reduction)

        # Dynamic pressure
        q = 0.5 * self.air_density * speed**2

        # Forces
        drag = q * self.config.frontal_area * cd
        downforce = q * self.config.frontal_area * cl

        return {
            'drag': drag,
            'downforce': downforce,
            'downforce_front': downforce * self.config.aero_balance_front,
            'downforce_rear': downforce * (1.0 - self.config.aero_balance_front),
        }

    def _calculate_tire_forces(
        self,
        velocity: np.ndarray,
        steering: float,
        downforce: float,
        brake: float
    ) -> Dict[str, float]:
        """
        Calculate tire forces using simplified Pacejka model.

        Returns longitudinal, lateral forces and yaw moment.
        """
        speed = np.linalg.norm(velocity)

        if speed < 0.1:
            return {'fx': 0.0, 'fy': 0.0, 'moment': 0.0}

        # Slip angle (simplified)
        if abs(velocity[0]) > 0.1:
            slip_angle = np.arctan2(velocity[1], velocity[0])
        else:
            slip_angle = 0.0

        # Steering angle (max ~20 degrees at wheels)
        max_steer_angle = np.radians(20)
        steer_angle = steering * max_steer_angle

        # Effective slip angle for front tires
        alpha_front = slip_angle - steer_angle
        alpha_rear = slip_angle

        # Normal forces (simplified, includes downforce)
        current_mass = self.config.min_weight + self.fuel_load
        static_load = current_mass * self.g

        # Weight transfer under braking
        brake_weight_transfer = brake * 0.3 * static_load  # 30% transfer

        # Front and rear loads
        fz_front = (static_load * self.config.weight_distribution_front +
                    downforce * self.config.aero_balance_front +
                    brake_weight_transfer)
        fz_rear = (static_load * (1.0 - self.config.weight_distribution_front) +
                   downforce * (1.0 - self.config.aero_balance_front) -
                   brake_weight_transfer)

        # Tire grip coefficient (decreases with wear and non-optimal temp)
        grip_mult = self._calculate_grip_multiplier()

        # Peak friction coefficient
        mu_peak = 1.8 * grip_mult  # F1 slicks can exceed 1.5-2.0

        # Simplified Pacejka magic formula
        # Lateral force
        C_alpha = 1.3  # Shape factor
        fy_front = -mu_peak * fz_front * np.sin(C_alpha * np.arctan(3.0 * alpha_front))
        fy_rear = -mu_peak * fz_rear * np.sin(C_alpha * np.arctan(3.0 * alpha_rear))

        fy_total = fy_front + fy_rear

        # Yaw moment (simplified)
        moment = (fy_front * self.config.wheelbase * 0.5 -
                  fy_rear * self.config.wheelbase * 0.5)

        # Longitudinal force (simplified - from tire friction during cornering)
        # In reality, this is coupled with lateral force via friction circle
        fx = 0.0  # Handled separately by engine/brakes

        return {
            'fx': fx,
            'fy': fy_total,
            'moment': moment,
        }

    def _calculate_engine_force(
        self,
        throttle: float,
        speed: float,
        gear: int,
        ers_mode: float
    ) -> float:
        """Calculate engine force including ICE and ERS."""
        if gear == 0:
            return 0.0

        # Get gear ratio
        gear_ratio = self.config.gear_ratios[gear - 1] * self.config.final_drive

        # Calculate engine RPM (simplified)
        wheel_rpm = (speed / (2 * np.pi * self.config.tire_radius)) * 60
        engine_rpm = wheel_rpm * gear_ratio

        # RPM limits
        rpm_min = 4000
        rpm_max = 15000  # Modern F1 limit

        if engine_rpm < rpm_min or engine_rpm > rpm_max:
            engine_power = self.config.max_power_ice * 0.5  # Reduced power outside optimal range
        else:
            # Power curve (simplified - peaks around 12000 RPM)
            rpm_normalized = (engine_rpm - rpm_min) / (rpm_max - rpm_min)
            power_curve = 1.0 - 0.3 * (rpm_normalized - 0.8)**2  # Peaks at 80% of range
            engine_power = self.config.max_power_ice * power_curve * throttle

        # ERS deployment
        ers_power = 0.0
        if ers_mode > 0 and self.ers_energy > 0:
            ers_power = self.config.max_power_ers_deploy * ers_mode

        total_power = engine_power + ers_power

        # Convert power to force
        if speed > 1.0:
            force = total_power / speed
        else:
            # At low speeds, limit force to avoid unrealistic acceleration
            force = total_power / 1.0

        # Apply to rear wheels only (RWD - wait, modern F1 is technically RWD for mechanical,
        # but ERS can provide front recovery... simplifying as RWD)
        return force

    def _calculate_brake_force(self, brake: float, speed: float) -> float:
        """Calculate braking force."""
        # Maximum brake force depends on tire grip and downforce
        # Simplified: use configured max brake force
        brake_force = brake * self.config.max_brake_force

        # Reduce effectiveness at very low speeds
        if speed < 2.0:
            brake_force *= speed / 2.0

        return brake_force

    def _calculate_grip_multiplier(self) -> float:
        """Calculate tire grip multiplier based on temperature and wear."""
        # Optimal tire temperature: 90-110°C
        temp_avg = np.mean(self.tire_temps)

        if 90 <= temp_avg <= 110:
            temp_mult = 1.0
        elif temp_avg < 90:
            temp_mult = 0.7 + 0.3 * (temp_avg / 90)
        else:  # temp > 110
            temp_mult = 1.0 - 0.002 * (temp_avg - 110)

        temp_mult = np.clip(temp_mult, 0.5, 1.0)

        # Wear effect
        wear_avg = np.mean(self.tire_wear)
        wear_mult = 1.0 - 0.3 * wear_avg  # Up to 30% grip loss when fully worn

        return temp_mult * wear_mult

    def _update_fuel(self, throttle: float, dt: float):
        """Update fuel consumption."""
        # Fuel consumption in kg/s
        fuel_rate = (self.config.fuel_consumption_rate / 3600) * throttle
        fuel_consumed = fuel_rate * dt
        self.fuel_load = max(0.0, self.fuel_load - fuel_consumed)

    def _update_ers(self, ers_mode: float, brake: float, speed: float, dt: float):
        """Update ERS energy."""
        if ers_mode > 0:
            # Deploying
            energy_deployed = self.config.ers_max_deploy_rate * ers_mode * dt
            self.ers_energy = max(0.0, self.ers_energy - energy_deployed)
            self.ers_deployed += energy_deployed
        elif ers_mode < 0 or brake > 0.3:
            # Harvesting (from braking or MGU-H)
            energy_recovered = self.config.ers_max_recovery_rate * abs(ers_mode) * dt
            # Also harvest from braking
            if brake > 0.3 and speed > 10.0:
                energy_recovered += brake * speed * 5000 * dt  # Simplified

            self.ers_energy = min(self.config.ers_max_energy,
                                  self.ers_energy + energy_recovered)
            self.ers_recovered += energy_recovered

    def _update_tires(self, speed: float, steering: float, brake: float, dt: float):
        """Update tire temperature and wear."""
        # Temperature increase from friction
        friction_factor = abs(steering) * speed * 0.01 + brake * speed * 0.02
        temp_increase = friction_factor * dt

        # Temperature decrease from cooling
        temp_decrease = 0.5 * dt  # Cooling rate

        # Update temperatures
        self.tire_temps += temp_increase - temp_decrease
        self.tire_temps = np.clip(self.tire_temps, 20.0, 150.0)

        # Wear rate depends on speed, steering, and temperature
        wear_rate = (speed * 0.00001 +
                     abs(steering) * speed * 0.00002 +
                     brake * speed * 0.00003)

        # Increased wear if overheated
        if np.mean(self.tire_temps) > 120:
            wear_rate *= 2.0

        self.tire_wear += wear_rate * dt
        self.tire_wear = np.clip(self.tire_wear, 0.0, 1.0)

    def _update_brakes(self, brake: float, speed: float, dt: float):
        """Update brake temperatures."""
        # Heat from braking
        heat_input = brake * speed * 50.0 * dt

        # Cooling
        cooling = 10.0 * dt

        self.brake_temps += heat_input - cooling
        self.brake_temps = np.clip(self.brake_temps, 50.0, 1000.0)

    def _update_wheel_speeds(self, speed: float, dt: float):
        """Update wheel speeds (simplified)."""
        if speed > 0.1:
            angular_speed = speed / self.config.tire_radius
            self.wheel_speeds[:] = angular_speed
        else:
            self.wheel_speeds[:] = 0.0

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-π, π]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    @staticmethod
    def _local_to_global(local_vec: np.ndarray, heading: float) -> np.ndarray:
        """Convert vector from local to global frame."""
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

        global_vec = np.array([
            cos_h * local_vec[0] - sin_h * local_vec[1],
            sin_h * local_vec[0] + cos_h * local_vec[1]
        ])

        return global_vec

    def get_telemetry(self) -> Dict[str, float]:
        """Get current telemetry data (for logging/visualization)."""
        speed = np.linalg.norm(self.velocity)
        speed_kmh = speed * 3.6

        return {
            'speed_kmh': speed_kmh,
            'position_x': self.position[0],
            'position_y': self.position[1],
            'heading_deg': np.degrees(self.heading),
            'fuel_kg': self.fuel_load,
            'ers_energy_mj': self.ers_energy / 1e6,
            'tire_temp_avg': np.mean(self.tire_temps),
            'tire_wear_avg': np.mean(self.tire_wear) * 100,
            'brake_temp_avg': np.mean(self.brake_temps),
            'gear': self.current_gear,
        }
