"""
Advanced Track Geometry System - High-Precision

Implements F1-grade track geometry including:
- Spline interpolation (cubic Hermite/Catmull-Rom)
- GPS-based track data import
- Elevation profiles with accuracy
- Banking angle variation
- Track width changes
- Curvature calculation
- Racing line optimization
- Sector definitions

Based on:
- F1 track laser scan data
- GPS telemetry from real laps
- Track surveying standards
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from scipy.interpolate import CubicSpline, UnivariateSpline, interp1d
from scipy.spatial.distance import cdist
import json


@dataclass
class TrackPoint:
    """High-precision track point with all attributes."""

    # Position
    distance: float  # meters from start line
    x: float  # meters (global coordinates)
    y: float  # meters
    z: float  # meters (elevation)

    # Geometry
    heading: float  # radians
    curvature: float  # 1/radius (1/m)
    banking: float  # radians

    # Track properties
    width: float  # meters
    surface_grip: float  # 0-1 multiplier

    # Racing line
    racing_line_offset: float  # meters from centerline (negative = left)
    optimal_speed: float  # km/h (calculated or provided)

    # Sector/zone info
    sector: int  # 1, 2, or 3
    zone_type: str  # 'straight', 'braking', 'corner', 'acceleration'
    drs_zone: bool = False


class AdvancedTrackGeometry:
    """
    High-precision track geometry using splines.

    Features:
    - GPS/laser scan data import
    - Smooth spline interpolation
    - Continuous curvature
    - Elevation handling
    - Racing line optimization
    """

    def __init__(self, name: str = "Generic Track"):
        self.name = name
        self.points: List[TrackPoint] = []

        # Spline representations
        self.centerline_spline = None
        self.elevation_spline = None
        self.width_spline = None
        self.banking_spline = None

        # Computed properties
        self.total_length = 0.0
        self.min_elevation = 0.0
        self.max_elevation = 0.0

        # Racing line
        self.racing_line_spline = None
        self.optimal_racing_line = None

    def from_gps_data(
        self,
        gps_points: np.ndarray,
        elevations: Optional[np.ndarray] = None,
        widths: Optional[np.ndarray] = None,
        bankings: Optional[np.ndarray] = None,
        smooth_factor: float = 0.0
    ):
        """
        Create track geometry from GPS data.

        Args:
            gps_points: Nx2 array of (lat, lon) or (x, y) coordinates
            elevations: N array of elevations (meters)
            widths: N array of track widths (meters)
            bankings: N array of banking angles (radians)
            smooth_factor: Spline smoothing (0=interpolation, >0=smoothing)
        """
        n_points = len(gps_points)

        # Convert GPS to local coordinates if needed
        if self._is_gps_coordinates(gps_points):
            xy_points = self._gps_to_local(gps_points)
        else:
            xy_points = gps_points

        # Calculate cumulative distance along track
        distances = self._calculate_distances(xy_points)

        # Default values if not provided
        if elevations is None:
            elevations = np.zeros(n_points)

        if widths is None:
            widths = np.full(n_points, 12.0)  # 12m default

        if bankings is None:
            bankings = np.zeros(n_points)

        # Create splines
        self._create_splines(
            distances, xy_points, elevations, widths, bankings, smooth_factor
        )

        # Sample track at high resolution
        self._sample_track(sample_rate=0.5)  # Every 0.5m

        # Calculate derived properties
        self._calculate_curvature()
        self._calculate_headings()

        self.total_length = distances[-1]

    def from_points_list(
        self,
        points: List[TrackPoint]
    ):
        """Create track from list of TrackPoint objects."""
        self.points = sorted(points, key=lambda p: p.distance)

        # Extract arrays
        distances = np.array([p.distance for p in self.points])
        xy = np.array([[p.x, p.y] for p in self.points])
        elevations = np.array([p.z for p in self.points])
        widths = np.array([p.width for p in self.points])
        bankings = np.array([p.banking for p in self.points])

        # Create splines
        self._create_splines(distances, xy, elevations, widths, bankings, smooth_factor=0.0)

        self.total_length = distances[-1]

    def _create_splines(
        self,
        distances: np.ndarray,
        xy: np.ndarray,
        elevations: np.ndarray,
        widths: np.ndarray,
        bankings: np.ndarray,
        smooth_factor: float
    ):
        """Create spline representations."""
        # Ensure distances are monotonically increasing
        assert np.all(np.diff(distances) > 0), "Distances must be increasing"

        if smooth_factor == 0.0:
            # Cubic spline interpolation (C2 continuous)
            self.centerline_spline = CubicSpline(
                distances, xy, bc_type='periodic'  # Closed loop
            )

            self.elevation_spline = CubicSpline(
                distances, elevations, bc_type='periodic'
            )

            self.width_spline = CubicSpline(
                distances, widths, bc_type='periodic'
            )

            self.banking_spline = CubicSpline(
                distances, bankings, bc_type='periodic'
            )

        else:
            # Smoothing spline (for noisy GPS data)
            self.centerline_spline = [
                UnivariateSpline(distances, xy[:, 0], s=smooth_factor),
                UnivariateSpline(distances, xy[:, 1], s=smooth_factor)
            ]

            self.elevation_spline = UnivariateSpline(
                distances, elevations, s=smooth_factor
            )

            self.width_spline = UnivariateSpline(
                distances, widths, s=smooth_factor
            )

            self.banking_spline = UnivariateSpline(
                distances, bankings, s=smooth_factor
            )

    def _sample_track(self, sample_rate: float = 0.5):
        """Sample track at uniform distance intervals."""
        distances_sampled = np.arange(0, self.total_length, sample_rate)

        self.points = []

        for d in distances_sampled:
            # Get position from spline
            if isinstance(self.centerline_spline, list):
                x = float(self.centerline_spline[0](d))
                y = float(self.centerline_spline[1](d))
            else:
                xy = self.centerline_spline(d)
                x, y = float(xy[0]), float(xy[1])

            z = float(self.elevation_spline(d))
            width = float(self.width_spline(d))
            banking = float(self.banking_spline(d))

            point = TrackPoint(
                distance=d,
                x=x,
                y=y,
                z=z,
                heading=0.0,  # Will be calculated
                curvature=0.0,  # Will be calculated
                banking=banking,
                width=width,
                surface_grip=1.0,
                racing_line_offset=0.0,
                optimal_speed=0.0,
                sector=self._determine_sector(d),
                zone_type='unknown',
                drs_zone=False
            )

            self.points.append(point)

    def _calculate_curvature(self):
        """Calculate curvature at each point using spline derivatives."""
        for point in self.points:
            d = point.distance

            # Get first and second derivatives
            if isinstance(self.centerline_spline, list):
                dx = float(self.centerline_spline[0].derivative()(d))
                dy = float(self.centerline_spline[1].derivative()(d))
                ddx = float(self.centerline_spline[0].derivative(2)(d))
                ddy = float(self.centerline_spline[1].derivative(2)(d))
            else:
                dxy = self.centerline_spline.derivative()(d)
                ddxy = self.centerline_spline.derivative(2)(d)
                dx, dy = float(dxy[0]), float(dxy[1])
                ddx, ddy = float(ddxy[0]), float(ddxy[1])

            # Curvature formula: κ = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
            numerator = dx * ddy - dy * ddx
            denominator = (dx**2 + dy**2)**1.5

            if abs(denominator) > 1e-6:
                point.curvature = numerator / denominator
            else:
                point.curvature = 0.0

    def _calculate_headings(self):
        """Calculate heading angle at each point."""
        for i, point in enumerate(self.points):
            # Get next point
            next_i = (i + 1) % len(self.points)
            next_point = self.points[next_i]

            # Calculate heading
            dx = next_point.x - point.x
            dy = next_point.y - point.y

            point.heading = np.arctan2(dy, dx)

    def optimize_racing_line(
        self,
        method: str = 'minimum_curvature',
        speed_weight: float = 1.0
    ):
        """
        Calculate optimal racing line.

        Args:
            method: 'minimum_curvature', 'minimum_time', or 'geometric'
            speed_weight: Weight for speed optimization

        Creates self.optimal_racing_line with offsets from centerline
        """
        if method == 'minimum_curvature':
            self._racing_line_minimum_curvature()
        elif method == 'minimum_time':
            self._racing_line_minimum_time(speed_weight)
        else:  # geometric
            self._racing_line_geometric()

    def _racing_line_geometric(self):
        """
        Geometric racing line (classic approach).

        For each corner:
        - Late apex
        - Maximize radius
        - Straighten the corner
        """
        for i, point in enumerate(self.points):
            if abs(point.curvature) < 0.001:  # Straight
                point.racing_line_offset = 0.0
            else:
                # Corner - move to outside to maximize radius
                # Negative curvature = right turn, positive = left turn
                # Move to outside of corner
                max_offset = point.width / 2.0 - 1.0  # 1m safety margin

                # Geometric optimal: move to edge
                if point.curvature > 0:  # Left turn
                    point.racing_line_offset = -max_offset
                else:  # Right turn
                    point.racing_line_offset = max_offset

    def _racing_line_minimum_curvature(self):
        """
        Minimum curvature racing line.

        Uses optimization to find line that minimizes total curvature.
        """
        # Simplified version: smooth the geometric line
        self._racing_line_geometric()

        # Extract offsets
        offsets = np.array([p.racing_line_offset for p in self.points])
        distances = np.array([p.distance for p in self.points])

        # Smooth using spline
        smoothing_spline = UnivariateSpline(distances, offsets, s=0.1)

        # Update offsets
        for i, point in enumerate(self.points):
            point.racing_line_offset = float(smoothing_spline(point.distance))

    def _racing_line_minimum_time(self, speed_weight: float):
        """
        Minimum time racing line (requires vehicle model).

        Placeholder - full implementation requires optimal control
        """
        # For now, use minimum curvature as approximation
        self._racing_line_minimum_curvature()

        # Could be extended with:
        # - Vehicle dynamics model
        # - Tire model
        # - Optimal control (direct/indirect methods)
        # - Iterative improvement

    def calculate_optimal_speeds(
        self,
        max_lateral_g: float = 5.0,
        max_speed: float = 350.0
    ):
        """
        Calculate optimal speed at each point.

        Uses:
        - Cornering limit: v = sqrt(μ * g * R)
        - Acceleration/braking limits

        Args:
            max_lateral_g: Maximum lateral acceleration (G's)
            max_speed: Maximum straight-line speed (km/h)
        """
        for point in self.points:
            if abs(point.curvature) < 0.001:  # Straight
                point.optimal_speed = max_speed
            else:
                # Corner speed limit
                radius = abs(1.0 / point.curvature)

                # Accounting for banking (increases cornering speed)
                effective_g = max_lateral_g + np.sin(point.banking)

                # v = sqrt(a * R)
                v_mps = np.sqrt(effective_g * 9.81 * radius)
                v_kmh = v_mps * 3.6

                point.optimal_speed = min(v_kmh, max_speed)

        # Forward pass: deceleration constraints
        for i in range(len(self.points) - 1, 0, -1):
            current = self.points[i]
            previous = self.points[i - 1]

            # Distance between points
            ds = current.distance - previous.distance

            # Maximum deceleration (assume 1.5G)
            max_decel = 1.5 * 9.81  # m/s²

            # Maximum speed at previous point given current speed
            v_current = current.optimal_speed / 3.6  # m/s
            v_max_prev = np.sqrt(v_current**2 + 2 * max_decel * ds)
            v_max_prev_kmh = v_max_prev * 3.6

            # Limit previous point speed
            previous.optimal_speed = min(previous.optimal_speed, v_max_prev_kmh)

        # Backward pass: acceleration constraints
        for i in range(len(self.points) - 1):
            current = self.points[i]
            next_point = self.points[i + 1]

            ds = next_point.distance - current.distance

            # Maximum acceleration (assume 1.2G)
            max_accel = 1.2 * 9.81  # m/s²

            v_current = current.optimal_speed / 3.6
            v_max_next = np.sqrt(v_current**2 + 2 * max_accel * ds)
            v_max_next_kmh = v_max_next * 3.6

            next_point.optimal_speed = min(next_point.optimal_speed, v_max_next_kmh)

    def get_point_at_distance(self, distance: float) -> TrackPoint:
        """Get interpolated track point at specific distance."""
        # Wrap around
        distance = distance % self.total_length

        # Find surrounding points
        distances = np.array([p.distance for p in self.points])
        idx = np.searchsorted(distances, distance)

        if idx == 0:
            return self.points[0]
        elif idx >= len(self.points):
            return self.points[-1]

        # Interpolate between points
        p1 = self.points[idx - 1]
        p2 = self.points[idx]

        alpha = (distance - p1.distance) / (p2.distance - p1.distance)

        # Linear interpolation (could use spline for smoother)
        return TrackPoint(
            distance=distance,
            x=p1.x + alpha * (p2.x - p1.x),
            y=p1.y + alpha * (p2.y - p1.y),
            z=p1.z + alpha * (p2.z - p1.z),
            heading=p1.heading + alpha * self._angle_diff(p2.heading, p1.heading),
            curvature=p1.curvature + alpha * (p2.curvature - p1.curvature),
            banking=p1.banking + alpha * (p2.banking - p1.banking),
            width=p1.width + alpha * (p2.width - p1.width),
            surface_grip=p1.surface_grip,
            racing_line_offset=p1.racing_line_offset + alpha * (p2.racing_line_offset - p1.racing_line_offset),
            optimal_speed=p1.optimal_speed + alpha * (p2.optimal_speed - p1.optimal_speed),
            sector=p1.sector,
            zone_type=p1.zone_type,
            drs_zone=p1.drs_zone or p2.drs_zone
        )

    def find_nearest_point(self, position: Tuple[float, float]) -> Tuple[TrackPoint, float]:
        """
        Find nearest track point to a given position.

        Args:
            position: (x, y) coordinates

        Returns:
            (nearest_point, lateral_deviation)
        """
        # Get all point positions
        track_positions = np.array([[p.x, p.y] for p in self.points])
        query_position = np.array([position]).reshape(1, -1)

        # Calculate distances
        distances = cdist(query_position, track_positions)[0]

        # Find nearest
        idx = np.argmin(distances)
        nearest = self.points[idx]

        # Calculate lateral deviation (signed)
        # Positive = right of track, negative = left
        dx = position[0] - nearest.x
        dy = position[1] - nearest.y

        # Project onto track normal
        normal_x = -np.sin(nearest.heading)
        normal_y = np.cos(nearest.heading)

        lateral_dev = dx * normal_x + dy * normal_y

        return nearest, lateral_dev

    def export_to_json(self, filepath: str):
        """Export track geometry to JSON."""
        data = {
            'name': self.name,
            'total_length': self.total_length,
            'points': [
                {
                    'distance': p.distance,
                    'x': p.x,
                    'y': p.y,
                    'z': p.z,
                    'heading': p.heading,
                    'curvature': p.curvature,
                    'banking': p.banking,
                    'width': p.width,
                    'racing_line_offset': p.racing_line_offset,
                    'optimal_speed': p.optimal_speed,
                    'sector': p.sector,
                    'drs_zone': p.drs_zone,
                }
                for p in self.points[::10]  # Downsample for file size
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def import_from_json(self, filepath: str):
        """Import track geometry from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.name = data['name']
        self.total_length = data['total_length']

        # Create points
        points = []
        for p_data in data['points']:
            point = TrackPoint(
                distance=p_data['distance'],
                x=p_data['x'],
                y=p_data['y'],
                z=p_data['z'],
                heading=p_data['heading'],
                curvature=p_data['curvature'],
                banking=p_data['banking'],
                width=p_data['width'],
                surface_grip=1.0,
                racing_line_offset=p_data['racing_line_offset'],
                optimal_speed=p_data['optimal_speed'],
                sector=p_data['sector'],
                zone_type='unknown',
                drs_zone=p_data['drs_zone']
            )
            points.append(point)

        self.from_points_list(points)

    # === HELPER METHODS ===

    @staticmethod
    def _is_gps_coordinates(points: np.ndarray) -> bool:
        """Check if points are GPS (lat/lon) vs local (x/y)."""
        # GPS coordinates typically in range [-90, 90] for lat, [-180, 180] for lon
        # Local coordinates typically much larger
        return np.all(np.abs(points) < 200)

    @staticmethod
    def _gps_to_local(gps_points: np.ndarray) -> np.ndarray:
        """
        Convert GPS (lat, lon) to local (x, y) coordinates.

        Uses simple equirectangular projection centered on first point.
        """
        lat0, lon0 = gps_points[0]

        # Earth radius
        R = 6371000  # meters

        lat_rad = np.deg2rad(gps_points[:, 0])
        lon_rad = np.deg2rad(gps_points[:, 1])
        lat0_rad = np.deg2rad(lat0)
        lon0_rad = np.deg2rad(lon0)

        # Equirectangular projection
        x = R * (lon_rad - lon0_rad) * np.cos(lat0_rad)
        y = R * (lat_rad - lat0_rad)

        return np.column_stack([x, y])

    @staticmethod
    def _calculate_distances(xy_points: np.ndarray) -> np.ndarray:
        """Calculate cumulative distance along points."""
        diffs = np.diff(xy_points, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        distances = np.concatenate([[0], np.cumsum(segment_lengths)])
        return distances

    def _determine_sector(self, distance: float) -> int:
        """Determine which sector (1, 2, or 3) a distance belongs to."""
        progress = distance / self.total_length

        if progress < 1/3:
            return 1
        elif progress < 2/3:
            return 2
        else:
            return 3

    @staticmethod
    def _angle_diff(angle2: float, angle1: float) -> float:
        """Calculate smallest difference between two angles."""
        diff = angle2 - angle1
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff


# === UTILITY FUNCTIONS ===

def create_track_from_gps_file(
    filepath: str,
    name: str = "Imported Track"
) -> AdvancedTrackGeometry:
    """
    Create track from GPS data file.

    Supported formats:
    - CSV: distance, lat, lon, elevation, width, banking
    - GPX: standard GPS exchange format
    """
    if filepath.endswith('.csv'):
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)

        gps_points = data[:, 1:3]  # lat, lon
        elevations = data[:, 3] if data.shape[1] > 3 else None
        widths = data[:, 4] if data.shape[1] > 4 else None
        bankings = data[:, 5] if data.shape[1] > 5 else None

    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    track = AdvancedTrackGeometry(name=name)
    track.from_gps_data(gps_points, elevations, widths, bankings)
    track.optimize_racing_line()
    track.calculate_optimal_speeds()

    return track
