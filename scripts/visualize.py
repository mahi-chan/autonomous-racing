#!/usr/bin/env python3
"""
Visualization script for F1 Racing telemetry and analysis.

Usage:
    python scripts/visualize.py --telemetry data/telemetry/episode_001.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize F1 Racing Telemetry")

    parser.add_argument(
        "--telemetry",
        type=str,
        required=True,
        help="Path to telemetry CSV file"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for plot (if None, displays interactively)"
    )

    return parser.parse_args()


def plot_telemetry(df: pd.DataFrame, output_path: str = None):
    """
    Create comprehensive telemetry visualization.

    Args:
        df: DataFrame with telemetry data
        output_path: Optional path to save plot
    """
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle('F1 Racing Telemetry Analysis', fontsize=16, fontweight='bold')

    # Filter to single episode if multiple
    if 'episode' in df.columns:
        episode_0 = df[df['episode'] == 0]
    else:
        episode_0 = df

    time = episode_0['time'].values if 'time' in episode_0.columns else np.arange(len(episode_0))

    # 1. Speed profile
    ax = axes[0, 0]
    if 'speed_kmh' in episode_0.columns:
        ax.plot(time, episode_0['speed_kmh'], 'b-', linewidth=1.5)
        ax.set_ylabel('Speed (km/h)', fontweight='bold')
        ax.set_title('Speed Profile')
        ax.grid(True, alpha=0.3)
        ax.fill_between(time, 0, episode_0['speed_kmh'], alpha=0.2)

    # 2. Tire temperatures
    ax = axes[0, 1]
    tire_temp_cols = [col for col in episode_0.columns if 'tire_temp' in col.lower()]
    if tire_temp_cols:
        for col in tire_temp_cols:
            ax.plot(time, episode_0[col], label=col, linewidth=1.5)
        ax.set_ylabel('Temperature (°C)', fontweight='bold')
        ax.set_title('Tire Temperatures')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=90, color='g', linestyle='--', alpha=0.5, label='Optimal Min')
        ax.axhline(y=110, color='r', linestyle='--', alpha=0.5, label='Optimal Max')

    # 3. Tire wear
    ax = axes[1, 0]
    if 'tire_wear_avg' in episode_0.columns:
        ax.plot(time, episode_0['tire_wear_avg'], 'r-', linewidth=2)
        ax.set_ylabel('Wear (%)', fontweight='bold')
        ax.set_title('Tire Wear')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=75, color='orange', linestyle='--', alpha=0.5, label='Cliff Zone')
        ax.legend()
        ax.fill_between(time, 0, episode_0['tire_wear_avg'], color='red', alpha=0.2)

    # 4. Fuel and ERS
    ax = axes[1, 1]
    ax2 = ax.twinx()
    if 'fuel_kg' in episode_0.columns:
        ax.plot(time, episode_0['fuel_kg'], 'g-', linewidth=1.5, label='Fuel (kg)')
        ax.set_ylabel('Fuel (kg)', fontweight='bold', color='g')
        ax.tick_params(axis='y', labelcolor='g')

    if 'ers_energy_mj' in episode_0.columns:
        ax2.plot(time, episode_0['ers_energy_mj'], 'orange', linewidth=1.5, label='ERS (MJ)')
        ax2.set_ylabel('ERS Energy (MJ)', fontweight='bold', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

    ax.set_title('Fuel & ERS Management')
    ax.grid(True, alpha=0.3)

    # 5. Engine RPM and gear
    ax = axes[2, 0]
    ax2 = ax.twinx()
    if 'engine_rpm' in episode_0.columns:
        ax.plot(time, episode_0['engine_rpm'], 'b-', linewidth=1.5)
        ax.set_ylabel('Engine RPM', fontweight='bold', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.axhline(y=15000, color='r', linestyle='--', alpha=0.5, label='RPM Limit')

    if 'gear' in episode_0.columns:
        ax2.plot(time, episode_0['gear'], 'r-', linewidth=1.5, drawstyle='steps-post')
        ax2.set_ylabel('Gear', fontweight='bold', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim(0, 9)

    ax.set_title('Engine RPM & Gear')
    ax.grid(True, alpha=0.3)

    # 6. Brake and throttle inputs
    ax = axes[2, 1]
    # Note: These might not be in telemetry depending on logging
    ax.set_ylabel('Input [0-1]', fontweight='bold')
    ax.set_title('Driver Inputs')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.5, 'Input telemetry not available',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=12, style='italic', alpha=0.5)

    # 7. Track position
    ax = axes[3, 0]
    if 'position_x' in episode_0.columns and 'position_y' in episode_0.columns:
        scatter = ax.scatter(
            episode_0['position_x'],
            episode_0['position_y'],
            c=episode_0['speed_kmh'] if 'speed_kmh' in episode_0.columns else time,
            cmap='jet',
            s=5,
            alpha=0.6
        )
        ax.set_xlabel('X Position (m)', fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontweight='bold')
        ax.set_title('Track Map (colored by speed)')
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, label='Speed (km/h)')
        ax.grid(True, alpha=0.3)

    # 8. Summary statistics
    ax = axes[3, 1]
    ax.axis('off')

    # Calculate summary stats
    summary_text = "=== Session Summary ===\n\n"

    if 'speed_kmh' in episode_0.columns:
        summary_text += f"Max Speed: {episode_0['speed_kmh'].max():.1f} km/h\n"
        summary_text += f"Avg Speed: {episode_0['speed_kmh'].mean():.1f} km/h\n\n"

    if 'distance' in episode_0.columns:
        summary_text += f"Distance: {episode_0['distance'].max():.0f} m\n\n"

    if 'fuel_kg' in episode_0.columns:
        fuel_used = episode_0['fuel_kg'].iloc[0] - episode_0['fuel_kg'].iloc[-1]
        summary_text += f"Fuel Used: {fuel_used:.2f} kg\n\n"

    if 'tire_wear_avg' in episode_0.columns:
        summary_text += f"Final Tire Wear: {episode_0['tire_wear_avg'].iloc[-1]:.1f}%\n\n"

    if 'time' in episode_0.columns:
        duration = episode_0['time'].iloc[-1]
        summary_text += f"Duration: {duration:.1f} s\n"

    ax.text(0.1, 0.9, summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def main():
    """Main visualization function."""
    args = parse_args()

    print(f"\nLoading telemetry from: {args.telemetry}")

    # Load telemetry
    df = pd.read_csv(args.telemetry)

    print(f"Loaded {len(df)} data points")
    if 'episode' in df.columns:
        print(f"Number of episodes: {df['episode'].nunique()}")

    print("\nGenerating visualization...\n")

    # Create plot
    plot_telemetry(df, args.output)

    print("Done!")


if __name__ == "__main__":
    main()
