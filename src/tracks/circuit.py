"""
Base Circuit class for racing tracks.

Defines track geometry, racing line, and characteristics.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum
import json


class TrackType(Enum):
    """Type of track segment."""
    STRAIGHT = "straight"
    CORNER_SLOW = "corner_slow"  # < 100 km/h
    CORNER_MEDIUM = "corner_medium"  # 100-150 km/h
    CORNER_FAST = "corner_fast"  # > 150 km/h
    CHICANE = "chicane"
    COMPLEX = "complex"


class SurfaceType(Enum):
    """Track surface type."""
    ASPHALT = "asphalt"
    CONCRETE = "concrete"
    KERB = "kerb"
    GRASS = "grass"
    GRAVEL = "gravel"


@dataclass
class TrackSegment:
    """
    Represents a segment of the racing circuit.
    """
    # Identification
    number: int
    name: str
    type: TrackType

    # Geometry
    start_distance: float  # meters from start line
    length: float  # meters
    curve_radius: float  # meters (infinity for straights)
    banking_angle: float = 0.0  # degrees

    # Racing line
    racing_line_offset: float = 0.0  # meters from centerline
    optimal_speed: float = 0.0  # km/h (calculated)

    # Braking zones
    braking_point: Optional[float] = None  # distance from segment start
    brake_pressure: Optional[float] = None  # optimal pressure [0-1]

    # DRS
    drs_zone: bool = False
    drs_detection_point: Optional[float] = None  # distance on track

    # Surface
    surface: SurfaceType = SurfaceType.ASPHALT
    grip_level: float = 1.0  # multiplier (1.0 = normal)

    # Elevation
    elevation: float = 0.0  # meters above sea level
    gradient: float = 0.0  # percentage

    def get_centerline_position(self, distance: float) -> Tuple[float, float]:
        """
        Get (x, y) position on centerline.

        Args:
            distance: Distance along segment [0, length]

        Returns:
            (x, y) coordinates
        """
        # Simplified - assume segment starts at origin
        if self.type == TrackType.STRAIGHT:
            # Straight line
            return (distance, 0.0)
        else:
            # Circular arc approximation
            if self.curve_radius > 0:
                angle = distance / self.curve_radius
                x = self.curve_radius * np.sin(angle)
                y = self.curve_radius * (1 - np.cos(angle))
                return (x, y)
            else:
                return (distance, 0.0)

    def get_racing_line_position(self, distance: float) -> Tuple[float, float]:
        """Get position on racing line (offset from centerline)."""
        cx, cy = self.get_centerline_position(distance)
        # Apply offset (simplified - perpendicular to centerline)
        return (cx, cy + self.racing_line_offset)


class Circuit:
    """
    Base class for racing circuits.

    Defines track layout, characteristics, and provides utilities
    for querying track properties at any position.
    """

    def __init__(self):
        # Track metadata
        self.name: str = "Generic Circuit"
        self.country: str = "Unknown"
        self.length: float = 5000.0  # meters
        self.lap_record: Optional[float] = None  # seconds
        self.num_corners: int = 10

        # Track characteristics
        self.track_width: float = 12.0  # meters
        self.pit_lane_length: float = 300.0  # meters
        self.start_finish_line: Tuple[float, float] = (0.0, 0.0)

        # Environmental (defaults)
        self.altitude: float = 0.0  # meters
        self.typical_temp: float = 20.0  # °C
        self.typical_track_temp: float = 30.0  # °C

        # Track segments
        self.segments: List[TrackSegment] = []

        # Track centerline (sampled points)
        self.centerline: np.ndarray = np.array([])  # Nx2 array of (x, y)
        self.racing_line: np.ndarray = np.array([])  # Nx2 array of (x, y)

        # Distance markers
        self.distance_to_segment: List[Tuple[int, float]] = []  # (segment_idx, distance_in_segment)

    def build_track(self):
        """
        Build track geometry from segments.

        Should be called after defining segments in subclass.
        """
        if not self.segments:
            raise ValueError("No segments defined")

        # Sample the track
        sample_distance = 1.0  # meters
        num_samples = int(self.length / sample_distance)

        centerline_points = []
        racing_line_points = []
        current_distance = 0.0

        for segment in self.segments:
            # Sample this segment
            segment_samples = int(segment.length / sample_distance)

            for i in range(segment_samples):
                dist_in_segment = i * sample_distance

                # Get positions
                cx, cy = segment.get_centerline_position(dist_in_segment)
                rx, ry = segment.get_racing_line_position(dist_in_segment)

                centerline_points.append([cx, cy])
                racing_line_points.append([rx, ry])

                # Track distance mapping
                self.distance_to_segment.append((segment.number, dist_in_segment))

                current_distance += sample_distance

        self.centerline = np.array(centerline_points)
        self.racing_line = np.array(racing_line_points)

    def get_segment_at_distance(self, distance: float) -> TrackSegment:
        """
        Get track segment at a given distance from start.

        Args:
            distance: Distance in meters (wraps around for multiple laps)

        Returns:
            TrackSegment
        """
        # Wrap around for multiple laps
        distance = distance % self.length

        for segment in self.segments:
            segment_end = segment.start_distance + segment.length
            if segment.start_distance <= distance < segment_end:
                return segment

        # If not found, return last segment
        return self.segments[-1]

    def get_racing_line_at_distance(self, distance: float) -> Tuple[float, float]:
        """
        Get racing line position at distance.

        Args:
            distance: Distance along track [m]

        Returns:
            (x, y) position
        """
        distance = distance % self.length
        idx = int((distance / self.length) * len(self.racing_line))
        idx = min(idx, len(self.racing_line) - 1)

        return tuple(self.racing_line[idx])

    def get_track_direction_at_distance(self, distance: float) -> float:
        """
        Get track heading angle at distance.

        Args:
            distance: Distance along track [m]

        Returns:
            Heading angle in radians
        """
        distance = distance % self.length
        idx = int((distance / self.length) * len(self.racing_line))
        idx = min(idx, len(self.racing_line) - 2)

        # Calculate direction from adjacent points
        p1 = self.racing_line[idx]
        p2 = self.racing_line[idx + 1]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        return np.arctan2(dy, dx)

    def find_nearest_racing_line_point(
        self,
        position: Tuple[float, float]
    ) -> Tuple[int, float, float]:
        """
        Find nearest point on racing line to given position.

        Args:
            position: (x, y) position

        Returns:
            (index, distance_to_line, distance_along_track)
        """
        pos = np.array(position)
        distances = np.linalg.norm(self.racing_line - pos, axis=1)
        nearest_idx = np.argmin(distances)

        distance_to_line = distances[nearest_idx]
        distance_along_track = (nearest_idx / len(self.racing_line)) * self.length

        return nearest_idx, distance_to_line, distance_along_track

    def is_off_track(self, position: Tuple[float, float], tolerance: float = 2.0) -> bool:
        """
        Check if position is off the track.

        Args:
            position: (x, y) position
            tolerance: Additional meters beyond track width

        Returns:
            True if off track
        """
        _, distance_to_line, _ = self.find_nearest_racing_line_point(position)

        max_distance = (self.track_width / 2.0) + tolerance

        return distance_to_line > max_distance

    def calculate_lap_time_theoretical(self, car_config: Dict) -> float:
        """
        Calculate theoretical best lap time for given car.

        Args:
            car_config: Car configuration parameters

        Returns:
            Lap time in seconds
        """
        # Simplified calculation based on segment speeds
        lap_time = 0.0

        for segment in self.segments:
            if segment.optimal_speed > 0:
                # Time = distance / speed
                time = segment.length / (segment.optimal_speed / 3.6)  # Convert km/h to m/s
            else:
                # Estimate based on segment type
                if segment.type == TrackType.STRAIGHT:
                    speed = 300.0 / 3.6  # 300 km/h
                elif segment.type == TrackType.CORNER_FAST:
                    speed = 200.0 / 3.6
                elif segment.type == TrackType.CORNER_MEDIUM:
                    speed = 120.0 / 3.6
                else:  # CORNER_SLOW or CHICANE
                    speed = 80.0 / 3.6

                time = segment.length / speed

            lap_time += time

        return lap_time

    def get_track_characteristics(self) -> Dict[str, float]:
        """
        Get track characteristics for setup optimization.

        Returns:
            Dictionary with track metrics
        """
        # Analyze segments
        total_straight_length = 0.0
        total_corner_length = 0.0
        slow_corners = 0
        fast_corners = 0

        for segment in self.segments:
            if segment.type == TrackType.STRAIGHT:
                total_straight_length += segment.length
            else:
                total_corner_length += segment.length
                if segment.type == TrackType.CORNER_SLOW:
                    slow_corners += 1
                elif segment.type == TrackType.CORNER_FAST:
                    fast_corners += 1

        high_speed_ratio = total_straight_length / self.length

        # Average corner speed (simplified)
        avg_corner_speed = 100.0  # Base
        if fast_corners > slow_corners * 2:
            avg_corner_speed = 150.0
        elif slow_corners > fast_corners * 2:
            avg_corner_speed = 80.0

        return {
            'length_km': self.length / 1000.0,
            'num_corners': self.num_corners,
            'high_speed_ratio': high_speed_ratio,
            'avg_corner_speed': avg_corner_speed,
            'straight_length_total': total_straight_length,
            'corner_count_slow': slow_corners,
            'corner_count_fast': fast_corners,
            'altitude_m': self.altitude,
        }

    def export_to_json(self, filepath: str):
        """Export track data to JSON file."""
        data = {
            'name': self.name,
            'country': self.country,
            'length': self.length,
            'lap_record': self.lap_record,
            'segments': [
                {
                    'number': seg.number,
                    'name': seg.name,
                    'type': seg.type.value,
                    'length': seg.length,
                    'curve_radius': seg.curve_radius,
                }
                for seg in self.segments
            ],
            'racing_line': self.racing_line.tolist(),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def __repr__(self) -> str:
        return (f"Circuit(name='{self.name}', length={self.length:.0f}m, "
                f"corners={self.num_corners}, segments={len(self.segments)})")
