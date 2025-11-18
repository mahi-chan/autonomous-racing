#!/usr/bin/env python3
"""
Download Real F1 Telemetry Data using FastF1
=============================================

This script downloads official F1 telemetry data from the FIA API
using the FastF1 library.

Usage:
    python scripts/download_real_data.py

Requirements:
    pip install fastf1

Author: F1 Racing RL System v1.2.0
"""

import argparse
from pathlib import Path
import pandas as pd

try:
    import fastf1
    import fastf1.plotting
except ImportError:
    print("\n❌ FastF1 not installed!")
    print("\nPlease install it:")
    print("  pip install fastf1")
    print("\nThen run this script again.\n")
    exit(1)


def download_lap(year: int, circuit: str, session_type: str, driver: str, output_dir: str = 'data/real_f1'):
    """
    Download real F1 lap telemetry.

    Args:
        year: Season year (2018-2024)
        circuit: Circuit name (e.g., 'Silverstone', 'Monaco', 'Spa')
        session_type: 'Q' (Qualifying), 'R' (Race), 'FP1', 'FP2', 'FP3'
        driver: 3-letter driver code (e.g., 'HAM', 'VER', 'LEC')
        output_dir: Where to save the data
    """
    print(f"\n{'='*70}")
    print(f"Downloading F1 Data")
    print(f"{'='*70}")
    print(f"Year: {year}")
    print(f"Circuit: {circuit}")
    print(f"Session: {session_type}")
    print(f"Driver: {driver}")
    print(f"{'='*70}\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load session
        print("Loading session data...")
        session = fastf1.get_session(year, circuit, session_type)
        session.load()
        print("✓ Session loaded")

        # Get driver laps
        print(f"Getting {driver}'s laps...")
        driver_laps = session.laps.pick_driver(driver)

        if len(driver_laps) == 0:
            print(f"❌ No laps found for driver {driver}")
            print(f"Available drivers: {session.laps['Driver'].unique()}")
            return None

        # Get fastest lap
        fastest_lap = driver_laps.pick_fastest()
        print(f"✓ Fastest lap: {fastest_lap['LapTime']}")

        # Get telemetry
        print("Extracting telemetry...")
        telemetry = fastest_lap.get_telemetry()

        # Convert to our format
        export_data = pd.DataFrame({
            'time': telemetry['Time'].dt.total_seconds() - telemetry['Time'].dt.total_seconds().iloc[0],
            'distance': telemetry['Distance'].values,
            'speed': telemetry['Speed'].values,
            'throttle': telemetry['Throttle'].values,
            'brake': telemetry['Brake'].values,
            'gear': telemetry['nGear'].values,
            'drs': telemetry['DRS'].values.astype(int),
            'rpm': telemetry['RPM'].values,
        })

        # Save to CSV
        filename = f"{circuit.lower()}_{driver}_{year}_{session_type.lower()}.csv"
        filepath = output_path / filename

        export_data.to_csv(filepath, index=False)

        print(f"✓ Data saved to: {filepath}")
        print(f"\nLap Statistics:")
        print(f"  Lap Time: {fastest_lap['LapTime']}")
        print(f"  Max Speed: {export_data['speed'].max():.1f} km/h")
        print(f"  Avg Speed: {export_data['speed'].mean():.1f} km/h")
        print(f"  Data Points: {len(export_data)}")

        # Also save track coordinates if available
        try:
            x_coords = telemetry['X'].values
            y_coords = telemetry['Y'].values

            track_data = pd.DataFrame({
                'x': x_coords,
                'y': y_coords,
                'distance': telemetry['Distance'].values,
            })

            track_filename = f"{circuit.lower()}_track_coords.csv"
            track_filepath = output_path / track_filename

            track_data.to_csv(track_filepath, index=False)
            print(f"✓ Track coordinates saved to: {track_filepath}")

        except Exception as e:
            print(f"⚠ Could not save track coordinates: {e}")

        return filepath

    except Exception as e:
        print(f"\n❌ Error downloading data: {e}")
        print("\nPossible issues:")
        print("  - Check circuit name spelling (e.g., 'Silverstone' not 'silverstone')")
        print("  - Check driver code is valid 3-letter code")
        print("  - Check year and session type are valid")
        print("  - Network connection required")
        return None


def download_multiple_circuits():
    """Download data for common F1 circuits."""
    circuits = [
        (2023, 'Silverstone', 'Q', 'HAM'),
        (2023, 'Monaco', 'Q', 'VER'),
        (2023, 'Spa', 'Q', 'LEC'),
    ]

    print("\n" + "="*70)
    print("DOWNLOADING MULTIPLE CIRCUITS")
    print("="*70)

    for year, circuit, session, driver in circuits:
        download_lap(year, circuit, session, driver)
        print("\n")


def main():
    parser = argparse.ArgumentParser(
        description='Download real F1 telemetry data using FastF1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Hamilton's fastest qualifying lap at Silverstone 2023
  python scripts/download_real_data.py --year 2023 --circuit Silverstone --session Q --driver HAM

  # Download Verstappen's race data from Monaco 2023
  python scripts/download_real_data.py --year 2023 --circuit Monaco --session R --driver VER

  # Download multiple common circuits
  python scripts/download_real_data.py --download-all
        """
    )

    parser.add_argument('--year', type=int, default=2023,
                       help='Season year (2018-2024)')
    parser.add_argument('--circuit', type=str, default='Silverstone',
                       help='Circuit name (e.g., Silverstone, Monaco, Spa)')
    parser.add_argument('--session', type=str, default='Q',
                       help='Session type: Q (Qualifying), R (Race), FP1, FP2, FP3')
    parser.add_argument('--driver', type=str, default='HAM',
                       help='3-letter driver code (e.g., HAM, VER, LEC)')
    parser.add_argument('--output-dir', type=str, default='data/real_f1',
                       help='Output directory for downloaded data')
    parser.add_argument('--download-all', action='store_true',
                       help='Download data for multiple common circuits')

    args = parser.parse_args()

    if args.download_all:
        download_multiple_circuits()
    else:
        download_lap(
            year=args.year,
            circuit=args.circuit,
            session_type=args.session,
            driver=args.driver,
            output_dir=args.output_dir
        )

    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    print("\nYou can now use this data in your simulation:")
    print("  1. Edit config.yaml")
    print("  2. Set validate_against_real_data: true")
    print("  3. Set real_data_path to the downloaded CSV file")
    print("  4. Run: python run_f1_simulation.py")
    print()


if __name__ == "__main__":
    main()
