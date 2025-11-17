"""
F1 Aerodynamics Model

Simulates:
- Downforce generation (front/rear wings, floor)
- Drag forces
- DRS (Drag Reduction System)
- Ground effect
- Aero balance
- Dirty air effects (when following another car)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AeroConfig:
    """Aerodynamic configuration parameters."""

    # Wing settings (adjustable)
    front_wing_angle: float = 5.0  # degrees (typically 2-8)
    rear_wing_angle: float = 10.0  # degrees (typically 5-15)

    # Base coefficients (at reference angles)
    base_drag_coefficient: float = 0.70
    base_downforce_coefficient: float = 3.5

    # Component breakdown
    # Downforce distribution
    front_wing_downforce_ratio: float = 0.30  # 30% from front wing
    floor_downforce_ratio: float = 0.45  # 45% from floor/ground effect
    rear_wing_downforce_ratio: float = 0.25  # 25% from rear wing

    # Drag distribution
    front_wing_drag_ratio: float = 0.25
    rear_wing_drag_ratio: float = 0.30
    body_drag_ratio: float = 0.45

    # DRS
    drs_drag_reduction: float = 0.15  # 15% drag reduction
    drs_downforce_reduction: float = 0.10  # 10% downforce reduction (mostly rear)

    # Ground effect
    ride_height_reference: float = 50.0  # mm
    ground_effect_sensitivity: float = 0.02  # How much downforce changes per mm

    # Dirty air (following another car)
    dirty_air_distance_threshold: float = 20.0  # meters
    dirty_air_downforce_loss: float = 0.25  # Up to 25% loss in dirty air

    # Reference values
    frontal_area: float = 1.4  # m²
    reference_velocity: float = 80.0  # m/s (288 km/h)

    def adjust_front_wing(self, angle_deg: float):
        """Adjust front wing angle (within legal limits)."""
        self.front_wing_angle = np.clip(angle_deg, 2.0, 8.0)

    def adjust_rear_wing(self, angle_deg: float):
        """Adjust rear wing angle (within legal limits)."""
        self.rear_wing_angle = np.clip(angle_deg, 5.0, 15.0)


class AerodynamicsModel:
    """
    Advanced aerodynamics model for F1 car.

    Calculates forces based on:
    - Speed
    - Ride height
    - Wing angles
    - DRS activation
    - Proximity to other cars (dirty air)
    """

    def __init__(self, config: Optional[AeroConfig] = None):
        self.config = config or AeroConfig()

        # Air properties
        self.air_density = 1.225  # kg/m³ at sea level, 15°C

        # Current state
        self.current_ride_height = self.config.ride_height_reference  # mm
        self.drs_active = False

        # Dirty air tracking
        self.in_dirty_air = False
        self.dirty_air_intensity = 0.0  # 0=clean, 1=maximum dirty air

    def set_environmental_conditions(
        self,
        altitude: float = 0.0,
        temperature: float = 15.0,
        pressure: float = 101325.0
    ):
        """
        Update air density based on environmental conditions.

        Args:
            altitude: Meters above sea level
            temperature: Celsius
            pressure: Pascals
        """
        # Ideal gas law
        R = 287.05  # Specific gas constant for dry air (J/(kg·K))
        T_kelvin = temperature + 273.15

        self.air_density = pressure / (R * T_kelvin)

    def calculate_forces(
        self,
        velocity: float,
        ride_height: float = None,
        drs_active: bool = False,
        dirty_air_intensity: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate aerodynamic forces.

        Args:
            velocity: Speed in m/s
            ride_height: Ride height in mm (None uses current)
            drs_active: Is DRS active
            dirty_air_intensity: How much dirty air (0-1)

        Returns:
            Dictionary with forces and coefficients
        """
        if ride_height is not None:
            self.current_ride_height = ride_height

        self.drs_active = drs_active
        self.dirty_air_intensity = dirty_air_intensity

        # Dynamic pressure
        q = 0.5 * self.air_density * velocity**2

        # Calculate coefficients
        cd, cl = self._calculate_coefficients()

        # Apply dirty air effect (reduces downforce)
        if dirty_air_intensity > 0:
            cl *= (1.0 - dirty_air_intensity * self.config.dirty_air_downforce_loss)

        # Calculate forces
        drag = q * self.config.frontal_area * cd
        downforce = q * self.config.frontal_area * cl

        # Downforce distribution
        downforce_front = downforce * self._calculate_front_downforce_ratio()
        downforce_rear = downforce - downforce_front

        # Aero balance (% front)
        aero_balance = downforce_front / downforce if downforce > 0 else 0.5

        return {
            'drag': drag,
            'downforce': downforce,
            'downforce_front': downforce_front,
            'downforce_rear': downforce_rear,
            'aero_balance': aero_balance,
            'drag_coefficient': cd,
            'downforce_coefficient': cl,
            'dynamic_pressure': q,
        }

    def _calculate_coefficients(self) -> tuple[float, float]:
        """
        Calculate drag and downforce coefficients.

        Returns:
            (cd, cl) tuple
        """
        # Base coefficients
        cd = self.config.base_drag_coefficient
        cl = self.config.base_downforce_coefficient

        # Wing angle effects
        # More wing angle = more downforce but also more drag
        front_wing_factor = self.config.front_wing_angle / 5.0  # Normalized to reference
        rear_wing_factor = self.config.rear_wing_angle / 10.0

        # Adjust coefficients based on wing settings
        cl_adjustment = (
            (front_wing_factor - 1.0) * 0.15 +  # Front wing contribution
            (rear_wing_factor - 1.0) * 0.20      # Rear wing contribution
        )

        cd_adjustment = (
            (front_wing_factor - 1.0) * 0.05 +
            (rear_wing_factor - 1.0) * 0.08
        )

        cl += cl_adjustment
        cd += cd_adjustment

        # Ground effect (more downforce closer to ground, but bottoming out reduces it)
        ride_height_delta = self.current_ride_height - self.config.ride_height_reference

        if self.current_ride_height > 10.0:  # Normal operation
            # Lower ride height = more ground effect
            ground_effect_multiplier = 1.0 - (ride_height_delta * self.config.ground_effect_sensitivity)
            ground_effect_multiplier = np.clip(ground_effect_multiplier, 0.7, 1.3)
        else:  # Bottoming out or porpoising
            # Stalled floor
            ground_effect_multiplier = 0.5

        cl *= ground_effect_multiplier

        # DRS effect
        if self.drs_active:
            cd *= (1.0 - self.config.drs_drag_reduction)
            # DRS mainly affects rear wing downforce
            cl_reduction = (self.config.drs_downforce_reduction *
                          self.config.rear_wing_downforce_ratio)
            cl *= (1.0 - cl_reduction)

        return cd, cl

    def _calculate_front_downforce_ratio(self) -> float:
        """
        Calculate what percentage of total downforce is at front.

        This depends on wing balance and ground effect.
        """
        # Base distribution
        front_ratio = self.config.front_wing_downforce_ratio

        # Ground effect affects floor (which we'll assume is 60% rear, 40% front)
        # So more ground effect = more rear bias

        # Front wing angle effect
        # More front wing = more front downforce
        wing_balance_shift = (self.config.front_wing_angle - 5.0) * 0.01

        front_ratio += wing_balance_shift

        return np.clip(front_ratio, 0.25, 0.45)

    def calculate_dirty_air_effect(self, distance_to_car_ahead: float) -> float:
        """
        Calculate dirty air intensity based on distance to car ahead.

        Args:
            distance_to_car_ahead: Distance in meters

        Returns:
            Dirty air intensity (0-1)
        """
        if distance_to_car_ahead > self.config.dirty_air_distance_threshold:
            return 0.0

        # Intensity decreases with distance
        # Maximum effect at 0m, decreases to 0 at threshold distance
        intensity = 1.0 - (distance_to_car_ahead / self.config.dirty_air_distance_threshold)

        # Non-linear falloff (more realistic)
        intensity = intensity ** 1.5

        return np.clip(intensity, 0.0, 1.0)

    def can_use_drs(
        self,
        distance_to_car_ahead: float,
        drs_detection_zone: bool = True
    ) -> bool:
        """
        Check if DRS can be used (racing rules).

        Args:
            distance_to_car_ahead: Distance to car ahead in meters
            drs_detection_zone: Is car in DRS detection zone

        Returns:
            True if DRS is allowed
        """
        drs_gap_threshold = 1.0  # Must be within 1 second (roughly 40-80m)

        # Simple distance-based check (in race, would be time-based)
        within_gap = distance_to_car_ahead < 80.0

        return drs_detection_zone and within_gap

    def optimize_setup_for_track(
        self,
        track_characteristics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Suggest optimal wing angles for a given track.

        Args:
            track_characteristics: Dict with 'high_speed_ratio', 'corner_speed_avg', etc.

        Returns:
            Recommended wing angles
        """
        high_speed_ratio = track_characteristics.get('high_speed_ratio', 0.5)
        corner_count = track_characteristics.get('corner_count', 15)
        avg_corner_speed = track_characteristics.get('avg_corner_speed', 100.0)  # km/h

        # High speed tracks (Monza) -> low downforce
        # Low speed tracks (Monaco, Hungary) -> high downforce

        # Speed-based adjustment
        if high_speed_ratio > 0.7:  # High-speed track
            front_wing = 3.0
            rear_wing = 6.0
        elif high_speed_ratio < 0.3:  # Low-speed track
            front_wing = 7.0
            rear_wing = 14.0
        else:  # Medium-speed track
            front_wing = 5.0
            rear_wing = 10.0

        # Fine-tune based on corner speed
        if avg_corner_speed < 80.0:
            front_wing += 1.0
            rear_wing += 2.0
        elif avg_corner_speed > 150.0:
            front_wing -= 1.0
            rear_wing -= 2.0

        # Ensure within legal limits
        front_wing = np.clip(front_wing, 2.0, 8.0)
        rear_wing = np.clip(rear_wing, 5.0, 15.0)

        return {
            'front_wing_angle': front_wing,
            'rear_wing_angle': rear_wing,
            'expected_top_speed_kmh': self._estimate_top_speed(rear_wing),
            'expected_cornering_g': self._estimate_cornering(front_wing, rear_wing),
        }

    def _estimate_top_speed(self, rear_wing_angle: float) -> float:
        """Estimate top speed based on wing configuration."""
        # Lower wing = higher top speed
        # Rough estimation
        base_speed = 330.0  # km/h
        wing_penalty = (rear_wing_angle - 5.0) * 3.0  # ~3 km/h per degree

        return base_speed - wing_penalty

    def _estimate_cornering(self, front_wing: float, rear_wing: float) -> float:
        """Estimate average cornering G-force."""
        # More downforce = more cornering
        avg_wing = (front_wing + rear_wing) / 2.0
        base_g = 3.5
        wing_boost = (avg_wing - 7.0) * 0.1

        return base_g + wing_boost

    def get_telemetry(self) -> Dict[str, float]:
        """Get aerodynamic telemetry data."""
        current_forces = self.calculate_forces(
            velocity=self.config.reference_velocity,
            drs_active=self.drs_active,
            dirty_air_intensity=self.dirty_air_intensity
        )

        return {
            'front_wing_angle': self.config.front_wing_angle,
            'rear_wing_angle': self.config.rear_wing_angle,
            'ride_height_mm': self.current_ride_height,
            'drs_active': int(self.drs_active),
            'dirty_air_intensity': self.dirty_air_intensity,
            'drag_coefficient': current_forces['drag_coefficient'],
            'downforce_coefficient': current_forces['downforce_coefficient'],
            'aero_balance': current_forces['aero_balance'],
        }
