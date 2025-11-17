"""
Silverstone Circuit - UK

Full Grand Prix layout (since 2011)
- Length: 5.891 km
- Corners: 18
- Lap record: 1:27.097 (Lewis Hamilton, 2020)
"""

import numpy as np
from src.tracks.circuit import Circuit, TrackSegment, TrackType, SurfaceType


class Silverstone(Circuit):
    """Silverstone Circuit implementation."""

    def __init__(self):
        super().__init__()

        # Metadata
        self.name = "Silverstone Circuit"
        self.country = "United Kingdom"
        self.length = 5891.0  # meters
        self.lap_record = 87.097  # seconds (Hamilton, 2020)
        self.num_corners = 18

        # Track properties
        self.track_width = 15.0  # meters (relatively wide)
        self.altitude = 150.0  # meters
        self.typical_temp = 18.0  # °C
        self.typical_track_temp = 28.0  # °C

        # Define track segments
        self._build_silverstone_segments()

        # Build geometry
        self.build_track()

    def _build_silverstone_segments(self):
        """Define Silverstone's track segments."""

        self.segments = [
            # Turn 1 - Abbey (Fast right)
            TrackSegment(
                number=1,
                name="Abbey",
                type=TrackType.CORNER_FAST,
                start_distance=0.0,
                length=120.0,
                curve_radius=200.0,
                racing_line_offset=-2.0,
                optimal_speed=240.0,
                drs_zone=False,
            ),

            # Turn 2 - Farm Curve (Fast left)
            TrackSegment(
                number=2,
                name="Farm Curve",
                type=TrackType.CORNER_FAST,
                start_distance=120.0,
                length=110.0,
                curve_radius=180.0,
                racing_line_offset=1.5,
                optimal_speed=235.0,
            ),

            # Turn 3 - Village (Medium left)
            TrackSegment(
                number=3,
                name="Village",
                type=TrackType.CORNER_MEDIUM,
                start_distance=230.0,
                length=90.0,
                curve_radius=100.0,
                racing_line_offset=2.0,
                optimal_speed=180.0,
                braking_point=20.0,
                brake_pressure=0.6,
            ),

            # Straight - The Loop approach
            TrackSegment(
                number=4,
                name="Straight to Loop",
                type=TrackType.STRAIGHT,
                start_distance=320.0,
                length=250.0,
                curve_radius=np.inf,
                optimal_speed=310.0,
            ),

            # Turn 4-5 - The Loop (Slow left hairpin)
            TrackSegment(
                number=5,
                name="The Loop",
                type=TrackType.CORNER_SLOW,
                start_distance=570.0,
                length=130.0,
                curve_radius=50.0,
                racing_line_offset=3.0,
                optimal_speed=90.0,
                braking_point=40.0,
                brake_pressure=0.9,
            ),

            # Turn 6 - Aintree (Medium right)
            TrackSegment(
                number=6,
                name="Aintree",
                type=TrackType.CORNER_MEDIUM,
                start_distance=700.0,
                length=100.0,
                curve_radius=120.0,
                racing_line_offset=-2.0,
                optimal_speed=150.0,
            ),

            # Wellington Straight
            TrackSegment(
                number=7,
                name="Wellington Straight",
                type=TrackType.STRAIGHT,
                start_distance=800.0,
                length=650.0,
                curve_radius=np.inf,
                optimal_speed=325.0,
                drs_zone=True,  # DRS zone
                drs_detection_point=750.0,
            ),

            # Turn 7 - Brooklands (Medium right)
            TrackSegment(
                number=8,
                name="Brooklands",
                type=TrackType.CORNER_MEDIUM,
                start_distance=1450.0,
                length=110.0,
                curve_radius=110.0,
                racing_line_offset=-2.5,
                optimal_speed=140.0,
                braking_point=35.0,
                brake_pressure=0.8,
            ),

            # Turn 8 - Luffield (Slow left-right complex)
            TrackSegment(
                number=9,
                name="Luffield",
                type=TrackType.COMPLEX,
                start_distance=1560.0,
                length=140.0,
                curve_radius=60.0,
                optimal_speed=110.0,
                braking_point=25.0,
                brake_pressure=0.7,
            ),

            # Turn 9 - Woodcote (Fast right)
            TrackSegment(
                number=10,
                name="Woodcote",
                type=TrackType.CORNER_FAST,
                start_distance=1700.0,
                length=90.0,
                curve_radius=170.0,
                racing_line_offset=-1.5,
                optimal_speed=210.0,
            ),

            # Pit Straight
            TrackSegment(
                number=11,
                name="Pit Straight",
                type=TrackType.STRAIGHT,
                start_distance=1790.0,
                length=770.0,
                curve_radius=np.inf,
                optimal_speed=330.0,
            ),

            # Turn 10 - Copse (Fast right)
            TrackSegment(
                number=12,
                name="Copse",
                type=TrackType.CORNER_FAST,
                start_distance=2560.0,
                length=120.0,
                curve_radius=190.0,
                racing_line_offset=-2.0,
                optimal_speed=260.0,
                braking_point=15.0,
                brake_pressure=0.4,
            ),

            # Maggotts straight
            TrackSegment(
                number=13,
                name="Approach to Maggotts",
                type=TrackType.STRAIGHT,
                start_distance=2680.0,
                length=180.0,
                curve_radius=np.inf,
                optimal_speed=290.0,
            ),

            # Turn 11-12 - Maggotts (Fast left)
            TrackSegment(
                number=14,
                name="Maggotts",
                type=TrackType.CORNER_FAST,
                start_distance=2860.0,
                length=130.0,
                curve_radius=220.0,
                racing_line_offset=2.0,
                optimal_speed=270.0,
            ),

            # Turn 13 - Becketts (Fast right-left complex)
            TrackSegment(
                number=15,
                name="Becketts",
                type=TrackType.COMPLEX,
                start_distance=2990.0,
                length=180.0,
                curve_radius=150.0,
                optimal_speed=250.0,
            ),

            # Turn 14 - Chapel (Fast left)
            TrackSegment(
                number=16,
                name="Chapel",
                type=TrackType.CORNER_FAST,
                start_distance=3170.0,
                length=100.0,
                curve_radius=200.0,
                racing_line_offset=1.5,
                optimal_speed=245.0,
            ),

            # Hangar Straight
            TrackSegment(
                number=17,
                name="Hangar Straight",
                type=TrackType.STRAIGHT,
                start_distance=3270.0,
                length=900.0,
                curve_radius=np.inf,
                optimal_speed=335.0,
                drs_zone=True,
                drs_detection_point=3200.0,
            ),

            # Turn 15 - Stowe (Slow right)
            TrackSegment(
                number=18,
                name="Stowe",
                type=TrackType.CORNER_SLOW,
                start_distance=4170.0,
                length=110.0,
                curve_radius=55.0,
                racing_line_offset=-2.5,
                optimal_speed=95.0,
                braking_point=45.0,
                brake_pressure=0.95,
            ),

            # Turn 16 - Vale (Medium left)
            TrackSegment(
                number=19,
                name="Vale",
                type=TrackType.CORNER_MEDIUM,
                start_distance=4280.0,
                length=130.0,
                curve_radius=100.0,
                racing_line_offset=2.0,
                optimal_speed=135.0,
            ),

            # Turn 17 - Club (Slow right-left chicane)
            TrackSegment(
                number=20,
                name="Club",
                type=TrackType.CHICANE,
                start_distance=4410.0,
                length=160.0,
                curve_radius=70.0,
                optimal_speed=115.0,
                braking_point=30.0,
                brake_pressure=0.75,
            ),

            # Final section back to Abbey
            TrackSegment(
                number=21,
                name="Final Approach",
                type=TrackType.STRAIGHT,
                start_distance=4570.0,
                length=1321.0,  # To complete the lap
                curve_radius=np.inf,
                optimal_speed=305.0,
            ),
        ]

    def get_ideal_racing_line_description(self) -> str:
        """
        Get description of ideal racing line for Silverstone.

        Returns:
            String describing the optimal racing line
        """
        return """
        Silverstone Ideal Racing Line:

        Turn 1 (Abbey): Fast right-hander, stay right on entry, clip apex, run wide on exit
        Turn 2 (Farm): Fast left, smooth arc, minimal steering input
        Turn 3 (Village): Medium left, brake hard, late apex for straight to The Loop

        The Loop: Heavy braking from 300+ km/h to ~90 km/h, wide entry, late apex
        Aintree: Medium right, smooth through for Wellington Straight

        Brooklands: Important for Luffield-Woodcote, brake hard, clip apex
        Luffield: Complex, sacrifice entry for good exit onto pit straight

        Copse: Critical corner, 260+ km/h, minimal braking, smooth arc
        Maggotts-Becketts: High-speed complex, maintain momentum, precision crucial
        Chapel: Fast left before Hangar Straight, prioritize exit speed

        Stowe: Hard braking point, ~95 km/h, crucial for lap time
        Vale-Club: Technical section, smooth inputs, maintain rhythm

        Key overtaking zones: Into Brooklands (post-DRS), Into Stowe (post-DRS)
        """

    def get_setup_recommendations(self) -> dict:
        """
        Get recommended setup for Silverstone.

        Returns:
            Dictionary with setup recommendations
        """
        return {
            'wing_levels': {
                'front_wing_angle': 5.5,  # Medium-high downforce
                'rear_wing_angle': 9.5,
                'reasoning': 'Balance between high-speed straights and fast corners'
            },
            'suspension': {
                'front_stiffness': 'medium-high',
                'rear_stiffness': 'medium-high',
                'ride_height': 45,  # mm - medium for Maggotts-Becketts
            },
            'brake_balance': {
                'front_bias': 58,  # percent
                'reasoning': 'Heavy braking into Stowe and The Loop'
            },
            'tire_strategy': {
                'race': 'One-stop: Soft → Medium or Medium → Hard',
                'qualifying': 'Soft compound for Q3',
                'degradation': 'Medium to High (Maggotts-Becketts is demanding)'
            },
            'ers_strategy': {
                'deploy_zones': ['Wellington Straight', 'Hangar Straight', 'Pit Straight'],
                'harvest_zones': ['Into The Loop', 'Into Brooklands', 'Into Stowe']
            },
            'critical_corners': [
                'Copse - high-speed entry, sets up Maggotts-Becketts',
                'Maggotts-Becketts - most demanding sequence, requires precision',
                'Stowe - major braking, overtaking opportunity'
            ]
        }
