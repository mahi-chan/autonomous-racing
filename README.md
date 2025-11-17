# F1 Autonomous Racing Agent - Reinforcement Learning System

A state-of-the-art reinforcement learning system designed for F1 racing optimization, enabling teams to find optimal racing lines, braking points, gear selection, and speed profiles for any circuit configuration.

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
- Accurate F1 car physics (aero, tire model, power unit)
- Real circuit models (Silverstone, Monaco, Spa, etc.)
- Telemetry data compatible with F1 formats
- Lap time optimization and race strategy
- DRS and ERS management
- Tire strategy optimization

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

## System Architecture

```
autonomous-racing/
├── src/
│   ├── envs/              # Racing environments
│   ├── agents/            # RL algorithms
│   ├── physics/           # Car physics models
│   ├── tracks/            # Circuit definitions
│   ├── dynamics/          # Dynamic conditions
│   ├── telemetry/         # Data collection
│   └── utils/             # Utilities
├── configs/               # Configuration files
├── scripts/               # Training/evaluation scripts
├── tests/                 # Test suite
└── docs/                  # Documentation
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
