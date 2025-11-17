"""
Visualization Tools for Advanced F1 Models

Provides comprehensive visualization of:
- Advanced tire model behavior
- CFD aerodynamics maps
- Track geometry and racing lines
- Training progress and performance
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional, Tuple
from pathlib import Path
import seaborn as sns


class TireModelVisualizer:
    """Visualize advanced tire model characteristics."""

    def __init__(self, tire_model):
        self.tire_model = tire_model

    def plot_friction_circle(self, output_path: str = "tire_friction_circle.png"):
        """
        Plot tire friction circle (combined slip limits).

        Shows maximum Fx-Fy combinations.
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # Generate slip angles and ratios
        alphas = np.linspace(-15, 15, 30)  # deg
        kappas = np.linspace(-0.3, 0.3, 30)

        Fz = 5000.0  # Nominal load
        gamma = 0.0
        V = 100.0 / 3.6  # 100 km/h

        Fx_max_list = []
        Fy_max_list = []

        # Calculate force envelope
        for alpha_deg in alphas:
            alpha = np.deg2rad(alpha_deg)
            Fx_max = 0
            Fy_max = 0

            for kappa in kappas:
                Fx, Fy, _, _ = self.tire_model.calculate_forces(
                    Fz=Fz, kappa=kappa, alpha=alpha, gamma=gamma, V=V, dt=0.01
                )

                if abs(Fx) > abs(Fx_max):
                    Fx_max = Fx
                if abs(Fy) > abs(Fy_max):
                    Fy_max = Fy

            Fx_max_list.append(Fx_max)
            Fy_max_list.append(Fy_max)

        # Normalize to show as g-forces (approx)
        mass_per_tire = 250  # kg (approx 1000kg / 4)
        g = 9.81

        Fx_g = np.array(Fx_max_list) / (mass_per_tire * g)
        Fy_g = np.array(Fy_max_list) / (mass_per_tire * g)

        # Plot
        ax.plot(Fx_g, Fy_g, 'b-', linewidth=2, label='Tire limit')
        ax.plot(-Fx_g, Fy_g, 'b-', linewidth=2)
        ax.plot(Fx_g, -Fy_g, 'b-', linewidth=2)
        ax.plot(-Fx_g, -Fy_g, 'b-', linewidth=2)

        # Add circle for reference
        theta = np.linspace(0, 2*np.pi, 100)
        max_g = max(max(abs(Fx_g)), max(abs(Fy_g)))
        ax.plot(max_g * np.cos(theta), max_g * np.sin(theta),
               'r--', linewidth=1, alpha=0.5, label='Perfect circle')

        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Longitudinal Force (g)', fontsize=11)
        ax.set_ylabel('Lateral Force (g)', fontsize=11)
        ax.set_title('Tire Friction Circle (Combined Slip)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    def plot_load_sensitivity(self, output_path: str = "tire_load_sensitivity.png"):
        """Plot tire force vs vertical load (load sensitivity)."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        Fz_range = np.linspace(2000, 8000, 50)
        kappa = 0.1  # 10% slip
        alpha = np.deg2rad(5)  # 5 deg slip angle
        gamma = 0.0
        V = 100.0 / 3.6

        Fx_list = []
        Fy_list = []

        for Fz in Fz_range:
            Fx, Fy, _, _ = self.tire_model.calculate_forces(
                Fz=Fz, kappa=kappa, alpha=alpha, gamma=gamma, V=V, dt=0.01
            )
            Fx_list.append(Fx)
            Fy_list.append(Fy)

        Fx_array = np.array(Fx_list)
        Fy_array = np.array(Fy_list)

        # Plot longitudinal force
        axes[0].plot(Fz_range / 1000, Fx_array / 1000, 'b-', linewidth=2)
        axes[0].set_xlabel('Vertical Load (kN)', fontsize=11)
        axes[0].set_ylabel('Longitudinal Force (kN)', fontsize=11)
        axes[0].set_title(f'Load Sensitivity - Fx (κ={kappa:.2f})', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # Plot lateral force
        axes[1].plot(Fz_range / 1000, Fy_array / 1000, 'r-', linewidth=2)
        axes[1].set_xlabel('Vertical Load (kN)', fontsize=11)
        axes[1].set_ylabel('Lateral Force (kN)', fontsize=11)
        axes[1].set_title(f'Load Sensitivity - Fy (α={np.rad2deg(alpha):.1f}°)', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    def plot_slip_curves(self, output_path: str = "tire_slip_curves.png"):
        """Plot classic tire slip curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        Fz = 5000.0
        gamma = 0.0
        V = 100.0 / 3.6

        # Longitudinal slip curve
        kappa_range = np.linspace(-0.3, 0.3, 100)
        Fx_list = []

        for kappa in kappa_range:
            Fx, _, _, _ = self.tire_model.calculate_forces(
                Fz=Fz, kappa=kappa, alpha=0.0, gamma=gamma, V=V, dt=0.01
            )
            Fx_list.append(Fx)

        axes[0].plot(kappa_range * 100, np.array(Fx_list) / 1000, 'b-', linewidth=2)
        axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[0].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        axes[0].set_xlabel('Longitudinal Slip (%)', fontsize=11)
        axes[0].set_ylabel('Longitudinal Force (kN)', fontsize=11)
        axes[0].set_title('Longitudinal Slip Curve', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # Lateral slip curve
        alpha_range = np.linspace(-15, 15, 100)  # degrees
        Fy_list = []

        for alpha_deg in alpha_range:
            alpha = np.deg2rad(alpha_deg)
            _, Fy, _, _ = self.tire_model.calculate_forces(
                Fz=Fz, kappa=0.0, alpha=alpha, gamma=gamma, V=V, dt=0.01
            )
            Fy_list.append(Fy)

        axes[1].plot(alpha_range, np.array(Fy_list) / 1000, 'r-', linewidth=2)
        axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        axes[1].set_xlabel('Slip Angle (deg)', fontsize=11)
        axes[1].set_ylabel('Lateral Force (kN)', fontsize=11)
        axes[1].set_title('Lateral Slip Curve', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


class AeroVisualizer:
    """Visualize CFD-based aerodynamics model."""

    def __init__(self, aero_model):
        self.aero_model = aero_model

    def plot_downforce_map(self, output_path: str = "aero_downforce_map.png"):
        """Plot downforce coefficient vs ride height and rake."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ride_heights = np.linspace(10, 80, 50)  # mm
        rakes = np.linspace(-0.5, 1.5, 50)  # degrees

        # Create meshgrid
        RH, RAKE = np.meshgrid(ride_heights, rakes)
        CL = np.zeros_like(RH)
        CD = np.zeros_like(RH)

        speed = 250.0 / 3.6  # 250 km/h
        yaw = 0.0

        # Calculate downforce for each combination
        for i in range(len(rakes)):
            for j in range(len(ride_heights)):
                rh_front = ride_heights[j]
                rh_rear = ride_heights[j] + rakes[i] * 3000 / 1000  # Approximate

                downforce, drag, _, _, _, _ = self.aero_model.calculate_forces(
                    speed=speed,
                    ride_height_front=rh_front / 1000,  # Convert to meters
                    ride_height_rear=rh_rear / 1000,
                    yaw_angle=yaw,
                    drs_active=False,
                    pitch_angle=rakes[i] * np.pi/180,
                    roll_angle=0.0
                )

                # Convert to coefficients (approximate)
                q = 0.5 * 1.225 * speed**2  # Dynamic pressure
                A_ref = 1.5  # Reference area (m^2)
                CL[i, j] = -downforce / (q * A_ref) if q > 0 else 0
                CD[i, j] = drag / (q * A_ref) if q > 0 else 0

        # Plot downforce coefficient
        contour1 = axes[0].contourf(RH, RAKE, CL, levels=20, cmap='viridis')
        axes[0].set_xlabel('Front Ride Height (mm)', fontsize=11)
        axes[0].set_ylabel('Rake (deg)', fontsize=11)
        axes[0].set_title('Downforce Coefficient (Cl)', fontsize=12, fontweight='bold')
        plt.colorbar(contour1, ax=axes[0], label='Cl')
        axes[0].grid(True, alpha=0.3)

        # Plot drag coefficient
        contour2 = axes[1].contourf(RH, RAKE, CD, levels=20, cmap='plasma')
        axes[1].set_xlabel('Front Ride Height (mm)', fontsize=11)
        axes[1].set_ylabel('Rake (deg)', fontsize=11)
        axes[1].set_title('Drag Coefficient (Cd)', fontsize=12, fontweight='bold')
        plt.colorbar(contour2, ax=axes[1], label='Cd')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    def plot_aero_balance(self, output_path: str = "aero_balance.png"):
        """Plot aerodynamic balance vs ride height."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ride_heights = np.linspace(10, 80, 50)  # mm
        balance_list = []

        speed = 250.0 / 3.6
        rake = 0.5  # deg

        for rh_front in ride_heights:
            rh_rear = rh_front + rake * 3000 / 1000

            downforce, drag, _, pitch_moment, _, _ = self.aero_model.calculate_forces(
                speed=speed,
                ride_height_front=rh_front / 1000,
                ride_height_rear=rh_rear / 1000,
                yaw_angle=0.0,
                drs_active=False,
                pitch_angle=rake * np.pi/180,
                roll_angle=0.0
            )

            # Balance = front downforce percentage (estimated from pitch moment)
            # More negative pitch moment = more front downforce
            # This is simplified - real calculation would need front/rear split
            balance = 50 + pitch_moment / 1000  # Approximate
            balance_list.append(np.clip(balance, 30, 70))

        ax.plot(ride_heights, balance_list, 'b-', linewidth=2)
        ax.axhline(y=50, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Neutral (50%)')
        ax.fill_between(ride_heights, 45, 55, alpha=0.2, color='green', label='Target range')

        ax.set_xlabel('Front Ride Height (mm)', fontsize=11)
        ax.set_ylabel('Aero Balance (% Front)', fontsize=11)
        ax.set_title('Aerodynamic Balance vs Ride Height', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


class TrackVisualizer:
    """Visualize track geometry and racing lines."""

    def __init__(self, track_geometry):
        self.track = track_geometry

    def plot_track_layout(self, output_path: str = "track_layout.png"):
        """Plot 2D track layout with racing line."""
        fig, ax = plt.subplots(figsize=(12, 10))

        # Extract coordinates
        x_coords = [seg.position[0] for seg in self.track.segments]
        y_coords = [seg.position[1] for seg in self.track.segments]

        # Racing line
        racing_x = [seg.racing_line_position[0] for seg in self.track.segments if seg.racing_line_position]
        racing_y = [seg.racing_line_position[1] for seg in self.track.segments if seg.racing_line_position]

        # Plot track centerline
        ax.plot(x_coords, y_coords, 'k-', linewidth=2, label='Centerline', alpha=0.5)

        # Plot racing line
        if racing_x and racing_y:
            ax.plot(racing_x, racing_y, 'r-', linewidth=3, label='Racing Line', alpha=0.8)

        # Mark start/finish
        ax.plot(x_coords[0], y_coords[0], 'go', markersize=15, label='Start/Finish')

        # Color segments by speed
        speeds = [seg.optimal_speed for seg in self.track.segments]
        if speeds and max(speeds) > 0:
            norm_speeds = np.array(speeds) / max(speeds)
            colors = plt.cm.RdYlGn(norm_speeds)

            for i in range(len(x_coords)-1):
                ax.plot([x_coords[i], x_coords[i+1]],
                       [y_coords[i], y_coords[i+1]],
                       color=colors[i], linewidth=8, alpha=0.3)

        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_title(f'{self.track.name} - Track Layout', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    def plot_track_profile(self, output_path: str = "track_profile.png"):
        """Plot elevation and curvature profiles."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        distances = [seg.distance_from_start for seg in self.track.segments]
        elevations = [seg.elevation for seg in self.track.segments]
        curvatures = [seg.curvature for seg in self.track.segments]
        speeds = [seg.optimal_speed for seg in self.track.segments]

        # Convert distance to km
        dist_km = np.array(distances) / 1000

        # Plot elevation
        axes[0].fill_between(dist_km, elevations, alpha=0.5)
        axes[0].plot(dist_km, elevations, 'b-', linewidth=2)
        axes[0].set_ylabel('Elevation (m)', fontsize=11)
        axes[0].set_title(f'{self.track.name} - Track Profile', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Plot curvature
        axes[1].plot(dist_km, np.array(curvatures) * 1000, 'r-', linewidth=2)
        axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1].set_ylabel('Curvature (1/km)', fontsize=11)
        axes[1].grid(True, alpha=0.3)

        # Plot optimal speed
        axes[2].fill_between(dist_km, speeds, alpha=0.5, color='green')
        axes[2].plot(dist_km, speeds, 'g-', linewidth=2)
        axes[2].set_xlabel('Distance (km)', fontsize=11)
        axes[2].set_ylabel('Optimal Speed (km/h)', fontsize=11)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


class TrainingVisualizer:
    """Visualize RL training progress."""

    @staticmethod
    def plot_training_curves(
        episode_rewards: List[float],
        episode_lengths: List[float],
        lap_times: List[float],
        output_path: str = "training_progress.png"
    ):
        """Plot training progress over episodes."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        episodes = np.arange(len(episode_rewards))

        # Plot rewards
        axes[0].plot(episodes, episode_rewards, 'b-', alpha=0.3)
        # Add moving average
        window = min(50, len(episode_rewards) // 10)
        if window > 1:
            rewards_smooth = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            axes[0].plot(episodes[window-1:], rewards_smooth, 'b-', linewidth=2, label='Moving avg')

        axes[0].set_ylabel('Episode Reward', fontsize=11)
        axes[0].set_title('Training Progress', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot episode lengths
        axes[1].plot(episodes, episode_lengths, 'g-', alpha=0.3)
        if window > 1:
            lengths_smooth = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
            axes[1].plot(episodes[window-1:], lengths_smooth, 'g-', linewidth=2, label='Moving avg')

        axes[1].set_ylabel('Episode Length (steps)', fontsize=11)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot lap times
        valid_lap_times = [t for t in lap_times if t > 0]
        valid_episodes = episodes[:len(valid_lap_times)]

        if valid_lap_times:
            axes[2].plot(valid_episodes, valid_lap_times, 'r-', alpha=0.3)
            if window > 1 and len(valid_lap_times) > window:
                times_smooth = np.convolve(valid_lap_times, np.ones(window)/window, mode='valid')
                axes[2].plot(valid_episodes[window-1:], times_smooth, 'r-', linewidth=2, label='Moving avg')

            axes[2].set_ylabel('Lap Time (s)', fontsize=11)
            axes[2].set_xlabel('Episode', fontsize=11)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    @staticmethod
    def plot_performance_heatmap(
        lap_times: np.ndarray,
        track_sections: int = 10,
        output_path: str = "performance_heatmap.png"
    ):
        """Plot performance heatmap showing improvement over track sections."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create heatmap data
        # Assuming lap_times is shape (episodes, track_sections)
        sns.heatmap(lap_times, cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Section Time (s)'})

        ax.set_xlabel('Track Section', fontsize=11)
        ax.set_ylabel('Episode', fontsize=11)
        ax.set_title('Performance Heatmap - Section Times', fontsize=13, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


# === EXAMPLE USAGE ===

def example_visualize_tire_model():
    """Example: Visualize advanced tire model."""
    from src.physics.tire_model_advanced import AdvancedTireModel

    tire = AdvancedTireModel(compound="SOFT")
    visualizer = TireModelVisualizer(tire)

    visualizer.plot_friction_circle("output/tire_friction_circle.png")
    visualizer.plot_load_sensitivity("output/tire_load_sensitivity.png")
    visualizer.plot_slip_curves("output/tire_slip_curves.png")

    print("Tire model visualizations created!")


def example_visualize_aero():
    """Example: Visualize aerodynamics model."""
    from src.physics.aerodynamics_advanced import AdvancedAeroModel

    aero = AdvancedAeroModel()
    visualizer = AeroVisualizer(aero)

    visualizer.plot_downforce_map("output/aero_downforce_map.png")
    visualizer.plot_aero_balance("output/aero_balance.png")

    print("Aerodynamics visualizations created!")


def example_visualize_track():
    """Example: Visualize track geometry."""
    from src.tracks.track_geometry_advanced import AdvancedTrackGeometry

    track = AdvancedTrackGeometry(name="Silverstone")
    # Assume track is loaded with GPS data
    # track.from_gps_data(...)

    visualizer = TrackVisualizer(track)
    visualizer.plot_track_layout("output/track_layout.png")
    visualizer.plot_track_profile("output/track_profile.png")

    print("Track visualizations created!")


if __name__ == "__main__":
    print("Visualization Tools for F1 Advanced Models")
    print("=" * 50)
    print("\nExamples:")
    print("1. example_visualize_tire_model()")
    print("2. example_visualize_aero()")
    print("3. example_visualize_track()")
