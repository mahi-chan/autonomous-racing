"""
Circuit de Monaco - Monaco

Street circuit through Monte Carlo
- Length: 3.337 km
- Corners: 19
- Lap record: 1:12.909 (Lewis Hamilton, 2021)
- Characteristics: Slow, technical, narrow
"""

from src.tracks.circuit import Circuit, TrackSegment, TrackType
import numpy as np


class Monaco(Circuit):
    """Circuit de Monaco implementation."""

    def __init__(self):
        super().__init__()

        self.name = "Circuit de Monaco"
        self.country = "Monaco"
        self.length = 3337.0
        self.lap_record = 72.909
        self.num_corners = 19

        self.track_width = 9.0  # Very narrow
        self.altitude = 10.0
        self.typical_temp = 22.0
        self.typical_track_temp = 35.0

        self._build_monaco_segments()
        self.build_track()

    def _build_monaco_segments(self):
        """Define Monaco's track segments (simplified)."""
        self.segments = [
            # Sainte-Dévote (T1)
            TrackSegment(1, "Sainte-Dévote", TrackType.CORNER_SLOW,
                        0.0, 80.0, 45.0, optimal_speed=85.0,
                        braking_point=30.0, brake_pressure=0.9),

            # Beau Rivage (Straight)
            TrackSegment(2, "Beau Rivage", TrackType.STRAIGHT,
                        80.0, 180.0, np.inf, optimal_speed=260.0),

            # Massenet (T2-3)
            TrackSegment(3, "Massenet", TrackType.CORNER_MEDIUM,
                        260.0, 90.0, 80.0, optimal_speed=110.0,
                        braking_point=25.0, brake_pressure=0.7),

            # Casino Square (T4-5)
            TrackSegment(4, "Casino Square", TrackType.COMPLEX,
                        350.0, 120.0, 60.0, optimal_speed=95.0),

            # Mirabeau (T6)
            TrackSegment(5, "Mirabeau", TrackType.CORNER_SLOW,
                        470.0, 70.0, 35.0, optimal_speed=70.0,
                        braking_point=35.0, brake_pressure=0.95),

            # Grand Hotel Hairpin (T8 - slowest corner in F1)
            TrackSegment(6, "Grand Hotel Hairpin", TrackType.CORNER_SLOW,
                        540.0, 100.0, 25.0, optimal_speed=50.0,
                        braking_point=40.0, brake_pressure=1.0),

            # Tunnel section
            TrackSegment(7, "Tunnel", TrackType.STRAIGHT,
                        640.0, 300.0, np.inf, optimal_speed=280.0),

            # Chicane (T13-14)
            TrackSegment(8, "Chicane", TrackType.CHICANE,
                        940.0, 110.0, 50.0, optimal_speed=90.0,
                        braking_point=40.0, brake_pressure=0.85),

            # Tabac (T15)
            TrackSegment(9, "Tabac", TrackType.CORNER_MEDIUM,
                        1050.0, 90.0, 70.0, optimal_speed=115.0),

            # Swimming Pool complex (T16-17-18)
            TrackSegment(10, "Swimming Pool", TrackType.COMPLEX,
                        1140.0, 150.0, 55.0, optimal_speed=100.0),

            # La Rascasse (T19)
            TrackSegment(11, "La Rascasse", TrackType.CORNER_SLOW,
                        1290.0, 80.0, 40.0, optimal_speed=75.0,
                        braking_point=25.0, brake_pressure=0.8),

            # Anthony Noghès (T20)
            TrackSegment(12, "Anthony Noghès", TrackType.CORNER_MEDIUM,
                        1370.0, 100.0, 65.0, optimal_speed=105.0),

            # Final section
            TrackSegment(13, "Final Approach", TrackType.STRAIGHT,
                        1470.0, 1867.0, np.inf, optimal_speed=240.0),
        ]

    def get_setup_recommendations(self) -> dict:
        """Monaco-specific setup."""
        return {
            'wing_levels': {
                'front_wing_angle': 8.0,  # Maximum downforce
                'rear_wing_angle': 15.0,  # Maximum downforce
                'reasoning': 'Slow-speed circuit, need maximum downforce'
            },
            'suspension': {
                'front_stiffness': 'soft',
                'rear_stiffness': 'soft',
                'ride_height': 60,  # Higher for bumps and kerbs
            },
            'tire_strategy': {
                'race': 'Zero-stop or one-stop if needed',
                'compound': 'Typically softest compounds',
            },
            'critical_corners': [
                'Grand Hotel Hairpin - slowest corner in F1',
                'Swimming Pool - technical complex',
                'Sainte-Dévote - lap 1 chaos'
            ]
        }
