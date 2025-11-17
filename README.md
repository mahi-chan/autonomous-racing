# F1 Autonomous Racing Agent - Reinforcement Learning System

A state-of-the-art reinforcement learning system designed for F1 racing optimization, enabling teams to find optimal racing lines, braking points, gear selection, and speed profiles for any circuit configuration.

## 🚀 Version 1.2.0 - F1-Grade Physics

**NEW: Advanced Features** (See [docs/ADVANCED_FEATURES.md](docs/ADVANCED_FEATURES.md)):
- ⚡ **Advanced Tire Model**: Pacejka MF 6.2 with 120+ parameters
- 🌊 **CFD Aerodynamics**: Multi-dimensional maps with ground effect
- 🛤️ **Spline Track Geometry**: High-precision GPS-based tracks
- 📊 **Real Data Import**: GPS, telemetry, and validation tools
- 📈 **Comprehensive Visualizations**: Tire, aero, and track analysis

**Previous Updates (v1.1.0)**:
- ✅ Fixed action space (now compatible with all standard RL algorithms)
- ✅ Added domain randomization for robust sim-to-real transfer
- ✅ Implemented opponent AI system
- ✅ Comprehensive integration tests

📖 See [CRITICAL_FIXES.md](CRITICAL_FIXES.md) and [docs/ADVANCED_FEATURES.md](docs/ADVANCED_FEATURES.md) for details.

## Overview

This system simulates F1 cars with realistic physics and allows training RL agents to optimize racing performance. Built for F1 team engineers to test car configurations and racing strategies.

## Key Features

### Advanced RL Algorithms
- **SAC (Soft Actor-Critic)** with adaptive temperature for optimal exploration
- **PPO with LSTM/GRU** for temporal dynamics and memory
- **Model-Based RL** (Dreamer & MBPO) for sample-efficient learning
- **Meta-RL** for rapid adaptation to new tracks and conditions

### Dynamic Environment Modeling
- **Tire degradation** over laps with compound-specific characteristics
- **Fuel load changes** affecting weight and balance
- **Track evolution** (temperature, rubber buildup)
- **Weather conditions** (dry, wet, changing conditions)
- **Dynamic opponent behavior** for race simulation

### F1-Specific Features
- **Advanced Physics Models**:
  - Pacejka Magic Formula 6.2 tire model (120+ parameters)
  - CFD-based aerodynamics with ground effect and porpoising
  - Spline-based track geometry from GPS data
  - Temperature, pressure, and degradation effects
- **Real Circuit Models**: Silverstone, Monaco, Spa with GPS precision
- **Telemetry & Validation**: Import real F1 data, compare with simulation
- **Lap Time Optimization**: Advanced racing line algorithms
- **Power Unit**: DRS, ERS, hybrid power management
- **Strategy**: Tire compound selection and pit stop optimization

## Platform Architecture

### Simulation Engine
- **Gymnasium-based** custom F1 environment
- **Physics Engine**: MuJoCo for accurate dynamics simulation
- **Alternative**: PyBullet for open-source option
- Real-time and accelerated simulation modes

### Training Infrastructure
- **Stable-Baselines3** with custom extensions
- **RLlib (Ray)** for distributed training
- GPU-accelerated training
- Parallel environment execution
- Automatic checkpointing and resuming

### Data Pipeline
- Real-time telemetry streaming
- TensorBoard integration
- Weights & Biases logging
- Custom F1 metrics and KPIs

## Quick Start

### Basic Training

```bash
# Install dependencies
pip install -e .

# Train an agent on Silverstone
python scripts/train.py --circuit silverstone --algorithm sac --config configs/f1_2024.yaml

# Evaluate trained agent
python scripts/evaluate.py --checkpoint models/sac_silverstone_best.zip --circuit silverstone

# Run interactive visualization
python scripts/visualize.py --checkpoint models/sac_silverstone_best.zip
```

### Advanced Features (v1.2.0+)

```python
# Import track from GPS data
from src.tracks.track_geometry_advanced import AdvancedTrackGeometry
from src.utils.data_import import GPSTrackImporter

gps_points, elevations = GPSTrackImporter.from_gpx('data/silverstone.gpx')
track = AdvancedTrackGeometry(name="Silverstone")
track.from_gps_data(gps_points, elevations)
track.optimize_racing_line(method='minimum_time')

# Create environment with advanced physics
from src.envs.f1_racing_env import F1RacingEnv

env = F1RacingEnv(
    circuit_name='silverstone',
    use_advanced_tire_model=True,    # Enable Pacejka MF 6.2
    use_advanced_aero=True,           # Enable CFD aerodynamics
    domain_randomization='moderate'   # Enable sim-to-real transfer
)

# Validate against real F1 data
from src.utils.validation import TelemetryValidator
from src.utils.data_import import TelemetryImporter

real_telemetry = TelemetryImporter.from_csv('data/real_f1_lap.csv')
validator = TelemetryValidator(track_name="Silverstone")
validator.load_real_data(real_telemetry, lap_time=85.093)
validator.load_simulation_data(sim_telemetry, lap_time=sim_lap_time)

metrics = validator.validate()
validator.generate_comparison_plots(output_dir="validation")
print(f"Validation score: {metrics.overall_score:.1f}/100")
```

📖 See [docs/ADVANCED_FEATURES.md](docs/ADVANCED_FEATURES.md) for complete examples.

## System Architecture

```
autonomous-racing/
├── src/
│   ├── envs/              # Racing environments
│   │   ├── f1_racing_env.py          # Main Gymnasium environment
│   │   ├── opponent_ai.py            # AI opponents (v1.1.0)
│   │   └── domain_randomization.py   # Sim-to-real transfer (v1.1.0)
│   ├── agents/            # RL algorithms
│   ├── physics/           # Car physics models
│   │   ├── tire_model_advanced.py    # Pacejka MF 6.2 (v1.2.0)
│   │   ├── aerodynamics_advanced.py  # CFD aero maps (v1.2.0)
│   │   ├── f1_car.py                 # Complete car model
│   │   └── ...
│   ├── tracks/            # Circuit definitions
│   │   ├── track_geometry_advanced.py # GPS-based tracks (v1.2.0)
│   │   ├── silverstone.py, monaco.py, spa.py
│   │   └── ...
│   ├── dynamics/          # Dynamic conditions
│   ├── telemetry/         # Data collection
│   └── utils/             # Utilities
│       ├── data_import.py             # GPS/telemetry import (v1.2.0)
│       ├── validation.py              # Validation tools (v1.2.0)
│       └── visualization.py           # Advanced visualizations (v1.2.0)
├── configs/               # Configuration files (YAML)
├── scripts/               # Training/evaluation scripts
├── tests/                 # Test suite
│   └── test_integration.py   # Integration tests (v1.1.0)
└── docs/                  # Documentation
    ├── ADVANCED_FEATURES.md  # Advanced features guide (v1.2.0)
    └── CRITICAL_FIXES.md     # v1.1.0 fixes
```

## Configuration System

The system uses YAML configuration files to define:
- Car specifications (aero, power unit, suspension)
- Track characteristics
- Training hyperparameters
- Dynamic condition parameters
- Opponent behavior

## Performance

Typical training times (NVIDIA A100):
- SAC: 2-4 hours for lap time optimization
- PPO-LSTM: 4-6 hours for full race strategy
- Model-based: 1-2 hours (sample efficient)
- Meta-RL: 30 min adaptation to new track

## Research References

Based on latest racing RL research:
- Wurman et al. (2022) - Gran Turismo Sophy
- Autonomous Racing Survey papers
- SAC, PPO, Dreamer, MAML implementations

## License

MIT License - See LICENSE file

## Contact

For F1 team integration and support, contact the development team.
