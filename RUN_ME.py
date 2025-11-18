#!/usr/bin/env python3
"""
F1 Racing Simulation - QUICK START
===================================

This is the EASIEST way to run the F1 simulation!

STEP 1: Edit config.yaml to set your parameters
STEP 2: Run this script: python RUN_ME.py

That's it! The system will:
- Test your car setups
- Train RL agents to find optimal driving
- Generate lap time comparisons
- Create visualizations and reports

Author: F1 Racing RL System v1.2.0
"""

import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import json


def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load configuration from YAML file."""
    if not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        print("\nCreating default config.yaml...")

        # Create default config
        default_config = {
            'circuit': 'silverstone',
            'use_advanced_tire_model': True,
            'use_advanced_aero': True,
            'domain_randomization': 'moderate',
            'train_rl_agent': True,
            'algorithm': 'sac',
            'total_timesteps': 100000,  # Quick test
            'save_model': True,
            'car_setups': [
                {
                    'name': 'Balanced',
                    'tire_compound': 'MEDIUM',
                    'front_wing_angle': 11,
                    'rear_wing_angle': 9,
                    'ride_height_front': 30,
                    'ride_height_rear': 40,
                    'fuel_load': 110,
                }
            ],
            'num_evaluation_laps': 3,
            'compare_setups': True,
            'validate_against_real_data': False,
            'real_data_path': 'data/real_f1_lap.csv',
            'real_lap_time': 85.093,
            'output_dir': 'results',
            'generate_visualizations': True,
            'generate_report': True,
        }

        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Created {config_path}")
        print("\nPlease edit config.yaml and run this script again!")
        exit(0)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def setup_environment(config: Dict):
    """Create output directories and print banner."""
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'telemetry').mkdir(exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    (output_dir / 'reports').mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"🏎️  F1 RACING SIMULATION - v1.2.0")
    print(f"{'='*70}")
    print(f"Circuit: {config['circuit'].upper()}")
    print(f"Algorithm: {config['algorithm'].upper()}")
    print(f"Advanced Tire Model: {'✓' if config['use_advanced_tire_model'] else '✗'}")
    print(f"Advanced Aero: {'✓' if config['use_advanced_aero'] else '✗'}")
    print(f"Setups to test: {len(config['car_setups'])}")
    print(f"Output: {output_dir.absolute()}")
    print(f"{'='*70}\n")

    return output_dir


def test_setup(setup_config: Dict, config: Dict, setup_index: int, output_dir: Path) -> Dict[str, Any]:
    """Test a single car setup."""
    from src.envs.f1_racing_env import F1RacingEnv
    from src.physics.tire_model import TireCompound

    setup_name = setup_config['name']

    print(f"\n{'-'*70}")
    print(f"🔧 Testing Setup {setup_index + 1}/{len(config['car_setups'])}: {setup_name}")
    print(f"{'-'*70}")

    # Convert tire compound
    compound_map = {
        'C1': TireCompound.C1, 'C2': TireCompound.C2, 'C3': TireCompound.C3,
        'C4': TireCompound.C4, 'C5': TireCompound.C5,
        'SOFT': TireCompound.C5, 'MEDIUM': TireCompound.C3, 'HARD': TireCompound.C1,
        'INTER': TireCompound.INTERMEDIATE, 'WET': TireCompound.WET,
    }
    tire_compound = compound_map.get(setup_config.get('tire_compound', 'C3'), TireCompound.C3)

    # Create environment
    print("Creating environment...")
    env = F1RacingEnv(
        circuit_name=config['circuit'],
        tire_compound=tire_compound,
        use_advanced_tire_model=config['use_advanced_tire_model'],
        use_advanced_aero=config['use_advanced_aero'],
        domain_randomization=config.get('domain_randomization'),
    )

    # Train or load model
    model_path = output_dir / 'models' / f"{setup_name}_{config['algorithm']}.zip"

    if config['train_rl_agent']:
        print(f"\n🚀 Training {config['algorithm'].upper()} agent...")
        print(f"   Timesteps: {config['total_timesteps']:,}")
        print(f"   (This may take a while...)\n")

        if config['algorithm'] == 'sac':
            from src.algorithms.sac_adaptive import train_sac
            model = train_sac(
                env=env,
                total_timesteps=config['total_timesteps'],
                save_path=str(model_path) if config['save_model'] else None
            )
        elif config['algorithm'] == 'ppo':
            from src.algorithms.ppo_lstm import train_ppo
            model = train_ppo(
                env=env,
                total_timesteps=config['total_timesteps'],
                save_path=str(model_path) if config['save_model'] else None
            )
        else:
            raise ValueError(f"Unknown algorithm: {config['algorithm']}")

        print(f"\n✓ Training complete!")
    else:
        from stable_baselines3 import SAC, PPO
        print(f"Loading model from {model_path}...")
        if config['algorithm'] == 'sac':
            model = SAC.load(str(model_path))
        else:
            model = PPO.load(str(model_path))
        print("✓ Model loaded")

    # Evaluate
    print(f"\n📊 Evaluating {config['num_evaluation_laps']} laps...")

    best_lap_time = float('inf')
    best_lap_telemetry = []
    all_lap_times = []

    for lap_num in range(config['num_evaluation_laps']):
        obs, _ = env.reset()
        lap_telemetry = []
        done = False
        truncated = False
        step_count = 0
        max_steps = 10000

        while not done and not truncated and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

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

    print(f"\n✅ Best lap time: {best_lap_time:.3f}s")
    if len(all_lap_times) > 1:
        print(f"   Average: {np.mean(all_lap_times):.3f}s ± {np.std(all_lap_times):.3f}s")

    return {
        'setup_name': setup_name,
        'setup_config': setup_config,
        'best_lap_time': best_lap_time,
        'avg_lap_time': np.mean(all_lap_times) if all_lap_times else best_lap_time,
        'lap_times': all_lap_times,
        'telemetry': best_lap_telemetry,
    }


def generate_report(results: List[Dict], config: Dict, output_dir: Path):
    """Generate comparison report and visualizations."""
    print(f"\n{'='*70}")
    print("📊 SETUP COMPARISON REPORT")
    print(f"{'='*70}\n")

    # Sort by best lap time
    sorted_results = sorted(results, key=lambda x: x['best_lap_time'])

    # Print table
    print(f"{'Rank':<6} {'Setup':<20} {'Best Lap':<12} {'Compound':<10}")
    print(f"{'-'*70}")

    for i, result in enumerate(sorted_results):
        rank = f"#{i+1}"
        if i == 0:
            rank = "🥇"
        elif i == 1:
            rank = "🥈"
        elif i == 2:
            rank = "🥉"

        print(f"{rank:<6} {result['setup_name']:<20} {result['best_lap_time']:.3f}s    {result['setup_config'].get('tire_compound', 'N/A'):<10}")

    # Highlight best
    best = sorted_results[0]
    print(f"\n{'-'*70}")
    print(f"🏆 OPTIMAL SETUP: {best['setup_name']}")
    print(f"⏱️  LAP TIME: {best['best_lap_time']:.3f}s")

    if len(sorted_results) > 1:
        delta = sorted_results[1]['best_lap_time'] - best['best_lap_time']
        print(f"📈 ADVANTAGE: +{delta:.3f}s over 2nd place ({delta/best['best_lap_time']*100:.2f}%)")

    # Generate visualizations
    if config['generate_visualizations']:
        print(f"\n📈 Generating visualizations...")
        viz_dir = output_dir / 'visualizations'

        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        names = [r['setup_name'] for r in sorted_results]
        times = [r['best_lap_time'] for r in sorted_results]

        bars = ax.bar(names, times, color=['gold' if i == 0 else 'steelblue' for i in range(len(names))], alpha=0.8)

        ax.set_ylabel('Lap Time (s)', fontsize=12, fontweight='bold')
        ax.set_title(f'Setup Comparison - {config["circuit"].title()}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(viz_dir / 'setup_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   ✓ Saved: setup_comparison.png")

    # Generate report
    if config['generate_report']:
        report_path = output_dir / 'reports' / 'report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("F1 RACING SIMULATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Circuit: {config['circuit'].upper()}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Algorithm: {config['algorithm'].upper()}\n")
            f.write(f"Training Steps: {config['total_timesteps']:,}\n\n")

            for i, r in enumerate(sorted_results):
                f.write(f"Rank {i+1}: {r['setup_name']}\n")
                f.write(f"  Best Lap: {r['best_lap_time']:.3f}s\n")
                f.write(f"  Tire: {r['setup_config'].get('tire_compound', 'N/A')}\n\n")

        print(f"   ✓ Saved: report.txt")


def main():
    """Main execution."""
    # Load configuration
    print("Loading configuration from config.yaml...")
    config = load_config('config.yaml')

    # Setup
    output_dir = setup_environment(config)

    # Test each setup
    results = []
    for i, setup in enumerate(config['car_setups']):
        try:
            result = test_setup(setup, config, i, output_dir)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error testing setup {setup['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Generate report
    if results and config['compare_setups']:
        generate_report(results, config, output_dir)

    # Done
    print(f"\n{'='*70}")
    print("✅ SIMULATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir.absolute()}")
    print(f"\nWhat's next?")
    print("  1. Check visualizations in: results/visualizations/")
    print("  2. Check detailed report in: results/reports/report.txt")
    print("  3. Modify config.yaml to test different setups")
    print("  4. Run again: python RUN_ME.py")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
