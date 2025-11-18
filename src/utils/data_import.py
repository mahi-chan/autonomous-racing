"""
Data Import Utilities for Real F1 Data

Supports importing:
- GPS telemetry (GPX, CSV formats)
- Lap telemetry (speed, throttle, brake, gear, etc.)
- Track data (laser scans, surveyed points)
- Setup sheets (car configuration)
- Tire data (temperature, pressure, wear sensors)

Enables validation against real F1 data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import xml.etree.ElementTree as ET


@dataclass
class TelemetryData:
    """Container for lap telemetry data."""

    time: np.ndarray  # seconds
    distance: np.ndarray  # meters
    speed: np.ndarray  # km/h
    throttle: np.ndarray  # 0-100%
    brake: np.ndarray  # 0-100%
    steering: np.ndarray  # degrees
    gear: np.ndarray  # 1-8
    rpm: np.ndarray  # RPM
    drs: np.ndarray  # 0/1

    # Optional tire data
    tire_temp_fl: Optional[np.ndarray] = None
    tire_temp_fr: Optional[np.ndarray] = None
    tire_temp_rl: Optional[np.ndarray] = None
    tire_temp_rr: Optional[np.ndarray] = None

    # Optional additional data
    lat_accel: Optional[np.ndarray] = None
    long_accel: Optional[np.ndarray] = None
    fuel_remaining: Optional[np.ndarray] = None


class TelemetryImporter:
    """Import telemetry data from various formats."""

    @staticmethod
    def from_csv(filepath: str, format_spec: Optional[Dict] = None) -> TelemetryData:
        """
        Import telemetry from CSV file.

        Args:
            filepath: Path to CSV file
            format_spec: Column mapping dictionary
                Example: {'time': 0, 'speed': 1, 'throttle': 2, ...}

        Returns:
            TelemetryData object
        """
        df = pd.read_csv(filepath)

        if format_spec is None:
            # Try to auto-detect columns
            format_spec = TelemetryImporter._auto_detect_columns(df)

        # Extract data
        telemetry = TelemetryData(
            time=df.iloc[:, format_spec['time']].values if 'time' in format_spec else np.arange(len(df)) * 0.01,
            distance=df.iloc[:, format_spec['distance']].values if 'distance' in format_spec else np.zeros(len(df)),
            speed=df.iloc[:, format_spec['speed']].values,
            throttle=df.iloc[:, format_spec['throttle']].values if 'throttle' in format_spec else np.zeros(len(df)),
            brake=df.iloc[:, format_spec['brake']].values if 'brake' in format_spec else np.zeros(len(df)),
            steering=df.iloc[:, format_spec['steering']].values if 'steering' in format_spec else np.zeros(len(df)),
            gear=df.iloc[:, format_spec['gear']].values.astype(int) if 'gear' in format_spec else np.ones(len(df)),
            rpm=df.iloc[:, format_spec['rpm']].values if 'rpm' in format_spec else np.zeros(len(df)),
            drs=df.iloc[:, format_spec['drs']].values if 'drs' in format_spec else np.zeros(len(df)),
        )

        # Optional columns
        if 'tire_temp_fl' in format_spec:
            telemetry.tire_temp_fl = df.iloc[:, format_spec['tire_temp_fl']].values

        return telemetry

    @staticmethod
    def from_f1_game(filepath: str) -> TelemetryData:
        """
        Import telemetry from F1 game (Codemasters format).

        Supports .csv exports from F1 2023/2024 games.
        """
        df = pd.read_csv(filepath)

        # F1 game column names (standard)
        return TelemetryData(
            time=df['time'].values if 'time' in df.columns else np.arange(len(df)) * 0.01,
            distance=df['distance'].values if 'distance' in df.columns else np.zeros(len(df)),
            speed=df['speed'].values,
            throttle=df['throttle'].values * 100 if 'throttle' in df.columns else np.zeros(len(df)),
            brake=df['brake'].values * 100 if 'brake' in df.columns else np.zeros(len(df)),
            steering=df['steer'].values if 'steer' in df.columns else np.zeros(len(df)),
            gear=df['gear'].values.astype(int) if 'gear' in df.columns else np.ones(len(df)),
            rpm=df['engineRPM'].values if 'engineRPM' in df.columns else np.zeros(len(df)),
            drs=df['drs'].values if 'drs' in df.columns else np.zeros(len(df)),
        )

    @staticmethod
    def _auto_detect_columns(df: pd.DataFrame) -> Dict[str, int]:
        """Auto-detect column mapping from dataframe."""
        mapping = {}

        # Common column name patterns
        patterns = {
            'time': ['time', 't', 'seconds', 'sec'],
            'distance': ['distance', 'dist', 'd', 'lap_dist'],
            'speed': ['speed', 'velocity', 'velo', 'spd', 'v'],
            'throttle': ['throttle', 'throt', 'thr', 'gas'],
            'brake': ['brake', 'brk'],
            'steering': ['steering', 'steer', 'str'],
            'gear': ['gear', 'g'],
            'rpm': ['rpm', 'engine_rpm', 'engine'],
        }

        for key, pattern_list in patterns.items():
            for col_idx, col_name in enumerate(df.columns):
                col_lower = str(col_name).lower()
                if any(pattern in col_lower for pattern in pattern_list):
                    mapping[key] = col_idx
                    break

        return mapping


class GPSTrackImporter:
    """Import track geometry from GPS data."""

    @staticmethod
    def from_gpx(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Import track from GPX file.

        Args:
            filepath: Path to GPX file

        Returns:
            (gps_points, elevations) tuple
            gps_points: Nx2 array of (lat, lon)
            elevations: N array of elevations (meters)
        """
        tree = ET.parse(filepath)
        root = tree.getroot()

        # GPX namespace
        ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}

        # Extract track points
        points = []
        elevations = []

        for trkpt in root.findall('.//gpx:trkpt', ns):
            lat = float(trkpt.get('lat'))
            lon = float(trkpt.get('lon'))
            points.append([lat, lon])

            # Elevation (optional)
            ele = trkpt.find('gpx:ele', ns)
            if ele is not None:
                elevations.append(float(ele.text))
            else:
                elevations.append(0.0)

        return np.array(points), np.array(elevations)

    @staticmethod
    def from_csv(filepath: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Import track from CSV file.

        Expected format: lat, lon, [elevation], [width], [banking]
        """
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)

        gps_points = data[:, :2]  # lat, lon
        elevations = data[:, 2] if data.shape[1] > 2 else None

        return gps_points, elevations


class SetupImporter:
    """Import car setup data."""

    @staticmethod
    def from_json(filepath: str) -> Dict:
        """Import setup from JSON file."""
        with open(filepath, 'r') as f:
            setup = json.load(f)
        return setup

    @staticmethod
    def from_f1_game_setup(filepath: str) -> Dict:
        """
        Import setup from F1 game setup file.

        Converts to simulator format.
        """
        # F1 game setups are XML
        tree = ET.parse(filepath)
        root = tree.getroot()

        setup = {
            'aero': {
                'front_wing': int(root.find('.//FrontWing').text),
                'rear_wing': int(root.find('.//RearWing').text),
            },
            'suspension': {
                'front_ride_height': int(root.find('.//FrontRideHeight').text),
                'rear_ride_height': int(root.find('.//RearRideHeight').text),
            },
            'brakes': {
                'brake_pressure': int(root.find('.//BrakePressure').text),
                'brake_bias': int(root.find('.//BrakeBias').text),
            }
        }

        return setup


class ValidationDataset:
    """
    Dataset for validation against real F1 data.

    Enables comparing simulation outputs to real telemetry.
    """

    def __init__(self):
        self.telemetry_laps: List[TelemetryData] = []
        self.lap_times: List[float] = []
        self.track_name: str = ""

    def add_lap(self, telemetry: TelemetryData, lap_time: float):
        """Add a lap to the dataset."""
        self.telemetry_laps.append(telemetry)
        self.lap_times.append(lap_time)

    def get_reference_lap(self) -> Tuple[TelemetryData, float]:
        """Get fastest lap (reference lap)."""
        best_idx = np.argmin(self.lap_times)
        return self.telemetry_laps[best_idx], self.lap_times[best_idx]

    def calculate_statistics(self) -> Dict:
        """Calculate statistics across all laps."""
        speeds = []
        throttle_usage = []

        for lap in self.telemetry_laps:
            speeds.extend(lap.speed)
            throttle_usage.extend(lap.throttle)

        return {
            'mean_speed': np.mean(speeds),
            'max_speed': np.max(speeds),
            'mean_throttle': np.mean(throttle_usage),
            'num_laps': len(self.telemetry_laps),
            'best_lap_time': min(self.lap_times),
            'average_lap_time': np.mean(self.lap_times),
        }

    def export_to_hdf5(self, filepath: str):
        """Export dataset to HDF5 for efficient storage."""
        try:
            import h5py

            with h5py.File(filepath, 'w') as f:
                # Metadata
                f.attrs['track_name'] = self.track_name
                f.attrs['num_laps'] = len(self.telemetry_laps)

                # Create group for each lap
                for i, (telemetry, lap_time) in enumerate(zip(self.telemetry_laps, self.lap_times)):
                    grp = f.create_group(f'lap_{i}')
                    grp.attrs['lap_time'] = lap_time

                    # Store arrays
                    grp.create_dataset('time', data=telemetry.time)
                    grp.create_dataset('distance', data=telemetry.distance)
                    grp.create_dataset('speed', data=telemetry.speed)
                    grp.create_dataset('throttle', data=telemetry.throttle)
                    grp.create_dataset('brake', data=telemetry.brake)
                    grp.create_dataset('steering', data=telemetry.steering)
                    grp.create_dataset('gear', data=telemetry.gear)
                    grp.create_dataset('rpm', data=telemetry.rpm)

            print(f"Dataset exported to {filepath}")

        except ImportError:
            print("h5py not installed - cannot export to HDF5")


# === EXAMPLE USAGE ===

def example_import_telemetry():
    """Example: Import telemetry from CSV."""
    # Import lap telemetry
    telemetry = TelemetryImporter.from_csv(
        'data/lap_telemetry.csv',
        format_spec={
            'time': 0,
            'distance': 1,
            'speed': 2,
            'throttle': 3,
            'brake': 4,
            'gear': 5,
        }
    )

    print(f"Imported {len(telemetry.time)} samples")
    print(f"Max speed: {np.max(telemetry.speed):.1f} km/h")
    print(f"Lap time: {telemetry.time[-1]:.2f} s")


def example_import_track():
    """Example: Import track from GPS data."""
    from src.tracks.track_geometry_advanced import AdvancedTrackGeometry

    # Import GPS data
    gps_points, elevations = GPSTrackImporter.from_gpx('data/silverstone.gpx')

    # Create track geometry
    track = AdvancedTrackGeometry(name="Silverstone")
    track.from_gps_data(gps_points, elevations)
    track.optimize_racing_line()
    track.calculate_optimal_speeds()

    print(f"Track imported: {track.total_length:.0f}m")
    print(f"Elevation change: {track.max_elevation - track.min_elevation:.1f}m")


def example_create_validation_dataset():
    """Example: Create validation dataset."""
    dataset = ValidationDataset()
    dataset.track_name = "Silverstone"

    # Import multiple laps
    for i in range(5):
        telemetry = TelemetryImporter.from_csv(f'data/lap_{i}.csv')
        lap_time = telemetry.time[-1]
        dataset.add_lap(telemetry, lap_time)

    # Get statistics
    stats = dataset.calculate_statistics()
    print(f"Best lap: {stats['best_lap_time']:.3f}s")
    print(f"Average: {stats['average_lap_time']:.3f}s")

    # Export
    dataset.export_to_hdf5('data/validation_dataset.h5')


if __name__ == "__main__":
    print("Data import utilities")
    print("="*50)
    print("\nExamples:")
    print("1. example_import_telemetry()")
    print("2. example_import_track()")
    print("3. example_create_validation_dataset()")
