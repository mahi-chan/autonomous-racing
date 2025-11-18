#!/usr/bin/env python3
"""
F1 Racing Simulation - Complete Workflow
=========================================

This script provides a complete end-to-end workflow for F1 racing simulation.
Just edit the configuration section below and run!

Features:
- Test multiple car setups
- Train RL agent to find optimal driving
- Compare lap times and telemetry
- Generate visualizations and reports
- Validate against real F1 data (optional)

Usage:
    python run_f1_simulation.py

Author: F1 Racing RL System v1.2.0
"""

import numpy as np
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION - EDIT THIS SECTION
# ============================================================================

CONFIG = {
    # === CIRCUIT SELECTION ===
    'circuit': 'silverstone',  # Options: 'silverstone', 'monaco', 'spa'

    # === ADVANCED FEATURES (v1.2.0) ===
    'use_advanced_tire_model': True,   # Use Pacejka MF 6.2 (120+ parameters)
    'use_advanced_aero': True,          # Use CFD-based aerodynamics
    'domain_randomization': 'moderate', # Options: None, 'light', 'moderate', 'heavy'

    # === TRAINING SETTINGS ===
    'train_rl_agent': True,             # Train RL agent or use existing model?
    'algorithm': 'sac',                 # Options: 'sac', 'ppo'
    'total_timesteps': 500_000,         # Training steps (500k = ~2-3 hours on GPU)
    'save_model': True,                 # Save trained model?

    # === CAR SETUPS TO TEST ===
    # The system will test each setup and compare lap times
    'car_setups': [
        {
            'name': 'High_Downforce',
            'tire_compound': 'SOFT',     # C1, C2, C3, C4, C5, INTER, WET
            'front_wing_angle': 15,      # degrees
            'rear_wing_angle': 12,
            'ride_height_front': 25,     # mm
            'ride_height_rear': 35,
            'fuel_load': 110,            # kg
        },
        {
            'name': 'Low_Drag',
            'tire_compound': 'MEDIUM',
            'front_wing_angle': 8,
            'rear_wing_angle': 6,
            'ride_height_front': 35,
            'ride_height_rear': 45,
            'fuel_load': 110,
        },
    ],

    # === EVALUATION ===
    'num_evaluation_laps': 5,           # Number of laps to evaluate each setup
    'compare_setups': True,             # Generate comparison report?

    # === VALIDATION (Optional) ===
    'validate_against_real_data': False, # Compare with real F1 data?
    'real_data_path': 'data/real_f1_lap.csv',  # Path to real telemetry
    'real_lap_time': 85.093,            # Real lap time (seconds)

    # === OUTPUT ===
    'output_dir': 'results',            # Where to save results
    'generate_visualizations': True,    # Create plots?
    'generate_report': True,            # Create text report?
}

# ============================================================================
# MAIN SCRIPT - NO NEED TO EDIT BELOW
# ============================================================================

def setup_environment():
    """Create output directories."""
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'telemetry').mkdir(exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    (output_dir / 'reports').mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"F1 RACING SIMULATION - v1.2.0")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Circuit: {CONFIG['circuit'].upper()}")
    print(f"Setups to test: {len(CONFIG['car_setups'])}")
    print(f"{'='*70}\n")

    return output_dir


def test_setup(setup_config: Dict, setup_index: int, output_dir: Path) -> Dict[str, Any]:
    """
    Test a single car setup.

    Args:
        setup_config: Car setup configuration
        setup_index: Index of this setup
        output_dir: Output directory

    Returns:
        Results dictionary with lap time, telemetry, etc.
    """
    from src.envs.f1_racing_env import F1RacingEnv
    from src.physics.tire_model import TireCompound

    setup_name = setup_config['name']

    print(f"\n{'-'*70}")
    print(f"Testing Setup {setup_index + 1}/{len(CONFIG['car_setups'])}: {setup_name}")
    print(f"{'-'*70}")

    # Convert tire compound string to enum
    compound_map = {
        'C1': TireCompound.C1,
        'C2': TireCompound.C2,
        'C3': TireCompound.C3,
        'C4': TireCompound.C4,
        'C5': TireCompound.C5,
        'SOFT': TireCompound.C5,
        'MEDIUM': TireCompound.C3,
        'HARD': TireCompound.C1,
        'INTER': TireCompound.INTERMEDIATE,
        'WET': TireCompound.WET,
    }
    tire_compound = compound_map.get(setup_config.get('tire_compound', 'C3'), TireCompound.C3)

    # Create environment
    env = F1RacingEnv(
        circuit_name=CONFIG['circuit'],
        tire_compound=tire_compound,
        use_advanced_tire_model=CONFIG['use_advanced_tire_model'],
        use_advanced_aero=CONFIG['use_advanced_aero'],
        domain_randomization=CONFIG['domain_randomization'],
    )

    # Train or load model
    model_path = output_dir / 'models' / f"{setup_name}_{CONFIG['algorithm']}.zip"

    if CONFIG['train_rl_agent']:
        print(f"Training {CONFIG['algorithm'].upper()} agent...")
        print(f"Timesteps: {CONFIG['total_timesteps']:,}")

        if CONFIG['algorithm'] == 'sac':
            from src.algorithms.sac_adaptive import train_sac
            model = train_sac(
                env=env,
                total_timesteps=CONFIG['total_timesteps'],
                save_path=str(model_path) if CONFIG['save_model'] else None
            )
        elif CONFIG['algorithm'] == 'ppo':
            from src.algorithms.ppo_lstm import train_ppo
            model = train_ppo(
                env=env,
                total_timesteps=CONFIG['total_timesteps'],
                save_path=str(model_path) if CONFIG['save_model'] else None
            )
        else:
            raise ValueError(f"Unknown algorithm: {CONFIG['algorithm']}")

        print(f"✓ Training complete!")
    else:
        # Load existing model
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        from stable_baselines3 import SAC, PPO
        if CONFIG['algorithm'] == 'sac':
            model = SAC.load(str(model_path))
        else:
            model = PPO.load(str(model_path))

        print(f"✓ Loaded model from {model_path}")

    # Evaluate model
    print(f"\nEvaluating {CONFIG['num_evaluation_laps']} laps...")

    best_lap_time = float('inf')
    best_lap_telemetry = []
    all_lap_times = []

    for lap_num in range(CONFIG['num_evaluation_laps']):
        obs, _ = env.reset()
        lap_telemetry = []
        done = False
        truncated = False
        step_count = 0
        max_steps = 10000

        while not done and not truncated and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            # Record telemetry
            lap_telemetry.append({
                'time': info.get('time', step_count * env.dt),
                'distance': info.get('distance', 0.0),
                'speed': info.get('speed', 0.0),
                'throttle': float(action[0]) * 100,
                'brake': float(action[1]) * 100,
                'steering': float(action[2]),
                'gear': int(action[3] * 8),
            })

            step_count += 1

            if done or truncated:
                lap_time = info.get('lap_time', info.get('time', 0.0))
                all_lap_times.append(lap_time)

                if lap_time < best_lap_time and lap_time > 0:
                    best_lap_time = lap_time
                    best_lap_telemetry = lap_telemetry

                print(f"  Lap {lap_num + 1}: {lap_time:.3f}s")
                break

    # Save telemetry
    if best_lap_telemetry:
        telemetry_path = output_dir / 'telemetry' / f"{setup_name}_telemetry.json"
        with open(telemetry_path, 'w') as f:
            json.dump({
                'setup': setup_config,
                'best_lap_time': best_lap_time,
                'all_lap_times': all_lap_times,
                'telemetry': best_lap_telemetry,
            }, f, indent=2)

        print(f"✓ Saved telemetry to {telemetry_path}")

    print(f"\n✓ Best lap time: {best_lap_time:.3f}s")
    print(f"  Average: {np.mean(all_lap_times):.3f}s")
    print(f"  Std dev: {np.std(all_lap_times):.3f}s")

    return {
        'setup_name': setup_name,
        'setup_config': setup_config,
        'best_lap_time': best_lap_time,
        'avg_lap_time': np.mean(all_lap_times),
        'lap_times': all_lap_times,
        'telemetry': best_lap_telemetry,
    }


def compare_setups(results: List[Dict], output_dir: Path):
    """Generate comparison report and visualizations."""
    print(f"\n{'='*70}")
    print("SETUP COMPARISON")
    print(f"{'='*70}\n")

    # Sort by best lap time
    sorted_results = sorted(results, key=lambda x: x['best_lap_time'])

    # Print comparison table
    print(f"{'Rank':<6} {'Setup Name':<20} {'Best Lap':<12} {'Avg Lap':<12} {'Compound':<10}")
    print(f"{'-'*70}")

    for i, result in enumerate(sorted_results):
        rank = i + 1
        setup_name = result['setup_name']
        best_lap = f"{result['best_lap_time']:.3f}s"
        avg_lap = f"{result['avg_lap_time']:.3f}s"
        compound = result['setup_config'].get('tire_compound', 'N/A')

        print(f"{rank:<6} {setup_name:<20} {best_lap:<12} {avg_lap:<12} {compound:<10}")

    # Calculate time deltas
    best_time = sorted_results[0]['best_lap_time']
    print(f"\n{'-'*70}")
    print(f"Optimal Setup: {sorted_results[0]['setup_name']}")
    print(f"Lap Time: {best_time:.3f}s")

    if len(sorted_results) > 1:
        delta = sorted_results[1]['best_lap_time'] - best_time
        print(f"Advantage over 2nd place: {delta:.3f}s ({delta/best_time*100:.2f}%)")

    # Generate visualizations
    if CONFIG['generate_visualizations']:
        print(f"\nGenerating visualizations...")
        viz_dir = output_dir / 'visualizations'

        # 1. Lap time comparison bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        setup_names = [r['setup_name'] for r in sorted_results]
        lap_times = [r['best_lap_time'] for r in sorted_results]

        bars = ax.bar(setup_names, lap_times, color='steelblue', alpha=0.8)
        bars[0].set_color('gold')  # Highlight best setup

        ax.set_ylabel('Lap Time (s)', fontsize=12)
        ax.set_title(f'Setup Comparison - {CONFIG["circuit"].title()}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}s',
                   ha='center', va='bottom', fontsize=10)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(viz_dir / 'setup_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {viz_dir / 'setup_comparison.png'}")

        # 2. Speed trace comparison
        if all('telemetry' in r and r['telemetry'] for r in results):
            fig, ax = plt.subplots(figsize=(14, 6))

            for result in sorted_results[:3]:  # Plot top 3 setups
                telemetry = result['telemetry']
                distances = [t['distance'] for t in telemetry]
                speeds = [t['speed'] for t in telemetry]

                ax.plot(np.array(distances) / 1000, speeds,
                       label=result['setup_name'], linewidth=2, alpha=0.8)

            ax.set_xlabel('Distance (km)', fontsize=11)
            ax.set_ylabel('Speed (km/h)', fontsize=11)
            ax.set_title('Speed Trace Comparison - Top 3 Setups', fontsize=13, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(viz_dir / 'speed_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  ✓ Saved: {viz_dir / 'speed_comparison.png'}")

    # Generate text report
    if CONFIG['generate_report']:
        report_path = output_dir / 'reports' / 'comparison_report.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("F1 RACING SIMULATION - SETUP COMPARISON REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Circuit: {CONFIG['circuit'].upper()}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Algorithm: {CONFIG['algorithm'].upper()}\n")
            f.write(f"Advanced Tire Model: {CONFIG['use_advanced_tire_model']}\n")
            f.write(f"Advanced Aero: {CONFIG['use_advanced_aero']}\n\n")

            f.write("-" * 70 + "\n")
            f.write("RESULTS\n")
            f.write("-" * 70 + "\n\n")

            for i, result in enumerate(sorted_results):
                f.write(f"Rank {i+1}: {result['setup_name']}\n")
                f.write(f"  Best Lap Time: {result['best_lap_time']:.3f}s\n")
                f.write(f"  Average Lap Time: {result['avg_lap_time']:.3f}s\n")
                f.write(f"  Tire Compound: {result['setup_config'].get('tire_compound', 'N/A')}\n")
                f.write(f"  Front Wing: {result['setup_config'].get('front_wing_angle', 'N/A')}°\n")
                f.write(f"  Rear Wing: {result['setup_config'].get('rear_wing_angle', 'N/A')}°\n")

                if i == 0:
                    f.write(f"\n  ⭐ OPTIMAL SETUP\n")
                elif i > 0:
                    delta = result['best_lap_time'] - sorted_results[0]['best_lap_time']
                    f.write(f"  Time delta: +{delta:.3f}s (+{delta/sorted_results[0]['best_lap_time']*100:.2f}%)\n")

                f.write("\n")

            f.write("=" * 70 + "\n")

        print(f"  ✓ Saved: {report_path}")


def main():
    """Main execution function."""
    # Setup
    output_dir = setup_environment()

    # Test each setup
    results = []
    for i, setup in enumerate(CONFIG['car_setups']):
        try:
            result = test_setup(setup, i, output_dir)
            results.append(result)
        except Exception as e:
            print(f"\n✗ Error testing setup {setup['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compare setups
    if len(results) > 1 and CONFIG['compare_setups']:
        compare_setups(results, output_dir)

    # Validation against real data (optional)
    if CONFIG['validate_against_real_data']:
        try:
            print(f"\n{'='*70}")
            print("VALIDATION AGAINST REAL F1 DATA")
            print(f"{'='*70}\n")

            from src.utils.data_import import TelemetryImporter
            from src.utils.validation import TelemetryValidator

            real_telemetry = TelemetryImporter.from_csv(CONFIG['real_data_path'])

            # Validate best setup
            best_result = min(results, key=lambda x: x['best_lap_time'])

            validator = TelemetryValidator(track_name=CONFIG['circuit'])
            validator.load_real_data(real_telemetry, CONFIG['real_lap_time'])

            # Convert telemetry to compatible format
            # (This is simplified - real implementation would need proper conversion)

            metrics = validator.validate()

            print(f"Validation Score: {metrics.overall_score:.1f}/100")
            print(f"Lap Time Error: {metrics.lap_time_error_percent:.2f}%")
            print(f"Speed Correlation: {metrics.speed_correlation:.4f}")

            # Generate validation plots
            validator.generate_comparison_plots(
                output_dir=str(output_dir / 'visualizations' / 'validation')
            )

        except Exception as e:
            print(f"\n✗ Validation failed: {e}")

    # Summary
    print(f"\n{'='*70}")
    print("SIMULATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir.absolute()}")
    print(f"\nSetups tested: {len(results)}")
    if results:
        best = min(results, key=lambda x: x['best_lap_time'])
        print(f"Best setup: {best['setup_name']}")
        print(f"Best lap time: {best['best_lap_time']:.3f}s")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
