"""
Circuit de Spa-Francorchamps - Belgium

Classic high-speed circuit in the Ardennes
- Length: 7.004 km (longest on calendar)
- Corners: 19
- Lap record: 1:46.286 (Valtteri Bottas, 2018)
- Characteristics: High-speed, elevation changes, unpredictable weather
"""

from src.tracks.circuit import Circuit, TrackSegment, TrackType
import numpy as np


class Spa(Circuit):
    """Spa-Francorchamps Circuit implementation."""

    def __init__(self):
        super().__init__()

        self.name = "Circuit de Spa-Francorchamps"
        self.country = "Belgium"
        self.length = 7004.0
        self.lap_record = 106.286
        self.num_corners = 19

        self.track_width = 14.0
        self.altitude = 420.0  # High altitude
        self.typical_temp = 16.0  # Often cool
        self.typical_track_temp = 25.0

        self._build_spa_segments()
        self.build_track()

    def _build_spa_segments(self):
        """Define Spa's track segments (simplified)."""
        self.segments = [
            # La Source (T1 - hairpin)
            TrackSegment(1, "La Source", TrackType.CORNER_SLOW,
                        0.0, 100.0, 40.0, optimal_speed=75.0,
                        braking_point=45.0, brake_pressure=0.95),

            # Eau Rouge approach
            TrackSegment(2, "Straight to Eau Rouge", TrackType.STRAIGHT,
                        100.0, 450.0, np.inf, optimal_speed=310.0),

            # Eau Rouge (T3 - famous uphill left)
            TrackSegment(3, "Eau Rouge", TrackType.CORNER_FAST,
                        550.0, 150.0, 250.0, optimal_speed=280.0,
                        elevation=50.0, gradient=15.0),

            # Raidillon (T4 - continues uphill)
            TrackSegment(4, "Raidillon", TrackType.CORNER_FAST,
                        700.0, 120.0, 200.0, optimal_speed=295.0,
                        elevation=80.0, gradient=12.0),

            # Kemmel Straight
            TrackSegment(5, "Kemmel Straight", TrackType.STRAIGHT,
                        820.0, 980.0, np.inf, optimal_speed=345.0,
                        drs_zone=True, drs_detection_point=700.0),

            # Les Combes (T5-T6-T7)
            TrackSegment(6, "Les Combes", TrackType.COMPLEX,
                        1800.0, 200.0, 90.0, optimal_speed=130.0,
                        braking_point=50.0, brake_pressure=0.85),

            # Malmedy/Bruxelles approach
            TrackSegment(7, "Approach to Bruxelles", TrackType.STRAIGHT,
                        2000.0, 380.0, np.inf, optimal_speed=290.0),

            # Bruxelles/Turn 9 (slow right)
            TrackSegment(8, "Bruxelles", TrackType.CORNER_SLOW,
                        2380.0, 90.0, 50.0, optimal_speed=95.0,
                        braking_point=40.0, brake_pressure=0.9),

            # Turn 10-11 (Pouhon - famous fast left)
            TrackSegment(9, "Pouhon", TrackType.CORNER_FAST,
                        2470.0, 180.0, 180.0, optimal_speed=240.0),

            # Turn 12-13 (Campus/Pif-Paf chicane)
            TrackSegment(10, "Campus", TrackType.CHICANE,
                        2650.0, 140.0, 60.0, optimal_speed=105.0,
                        braking_point=35.0, brake_pressure=0.75),

            # Back section approach
            TrackSegment(11, "Straight to Blanchimont", TrackType.STRAIGHT,
                        2790.0, 1200.0, np.inf, optimal_speed=315.0),

            # Blanchimont (T17 - very fast left)
            TrackSegment(12, "Blanchimont", TrackType.CORNER_FAST,
                        3990.0, 200.0, 300.0, optimal_speed=305.0),

            # Approach to Bus Stop
            TrackSegment(13, "Approach to Bus Stop", TrackType.STRAIGHT,
                        4190.0, 650.0, np.inf, optimal_speed=330.0),

            # Bus Stop Chicane (T18-19)
            TrackSegment(14, "Bus Stop", TrackType.CHICANE,
                        4840.0, 160.0, 55.0, optimal_speed=100.0,
                        braking_point=45.0, brake_pressure=0.85),

            # Final straight back to La Source
            TrackSegment(15, "Final Straight", TrackType.STRAIGHT,
                        5000.0, 2004.0, np.inf, optimal_speed=325.0,
                        drs_zone=True, drs_detection_point=4900.0),
        ]

    def get_setup_recommendations(self) -> dict:
        """Spa-specific setup."""
        return {
            'wing_levels': {
                'front_wing_angle': 4.0,  # Low downforce
                'rear_wing_angle': 7.0,   # Low downforce
                'reasoning': 'Long straights (Kemmel, final), need high top speed'
            },
            'suspension': {
                'front_stiffness': 'medium',
                'rear_stiffness': 'medium',
                'ride_height': 42,  # Medium - for Eau Rouge/Raidillon
            },
            'tire_strategy': {
                'race': 'Usually one-stop, medium degradation',
                'weather': 'Be prepared for rain - very common at Spa!',
            },
            'critical_corners': [
                'Eau Rouge/Raidillon - iconic, flat-out in modern F1',
                'Pouhon - high-speed left, very demanding',
                'Blanchimont - 300+ km/h, commitment required'
            ],
            'weather_note': 'Spa is famous for unpredictable weather - can be raining in one sector and dry in another!'
        }
