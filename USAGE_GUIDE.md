# F1 Autonomous Racing - Usage Guide

Complete guide for training and using the F1 Racing RL system.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Training Agents](#training-agents)
4. [Evaluating Agents](#evaluating-agents)
5. [Configuration](#configuration)
6. [Advanced Usage](#advanced-usage)
7. [Telemetry Analysis](#telemetry-analysis)

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM
- 10GB disk space

### Install Dependencies

```bash
# Clone the repository
cd autonomous-racing

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import src; from src.envs import F1RacingEnv; print('Installation successful!')"
```

---

## Quick Start

### 1. Train a Simple Agent (5 minutes)

```bash
# Train SAC on Silverstone for 100k steps
python scripts/train.py \
    --circuit silverstone \
    --algorithm sac \
    --total-timesteps 100000 \
    --save-dir models/quickstart
```

### 2. Evaluate the Agent

```bash
# Evaluate trained agent
python scripts/evaluate.py \
    --checkpoint models/quickstart/sac_silverstone_final.zip \
    --circuit silverstone \
    --num-episodes 5 \
    --deterministic
```

### 3. Visualize Telemetry

```bash
# If telemetry was saved during evaluation
python scripts/visualize.py --telemetry telemetry_output.csv
```

---

## Training Agents

### SAC (Soft Actor-Critic) with Adaptive Temperature

**Best for**: General racing, lap time optimization

```bash
python scripts/train.py \
    --circuit silverstone \
    --algorithm sac \
    --config configs/f1_2024.yaml \
    --total-timesteps 1000000 \
    --num-envs 4 \
    --save-dir models/sac \
    --log-dir logs/sac
```

**Training time**: ~2-4 hours on RTX 4090

**When to use**:
- Lap time optimization
- Finding optimal racing lines
- Quick training iteration

### PPO with LSTM (Recurrent Policy)

**Best for**: Strategy, tire management, fuel management

```bash
python scripts/train.py \
    --circuit silverstone \
    --algorithm ppo-lstm \
    --config configs/f1_2024.yaml \
    --total-timesteps 2000000 \
    --save-dir models/ppo_lstm \
    --log-dir logs/ppo_lstm
```

**Training time**: ~4-6 hours on RTX 4090

**When to use**:
- Race strategy optimization
- Tire degradation management
- Long-term planning (fuel, tire wear)
- Full race simulation

### Model-Based RL (Dreamer)

**Best for**: Sample efficiency, sim-to-real transfer

```bash
python scripts/train.py \
    --circuit silverstone \
    --algorithm model-based \
    --config configs/f1_2024.yaml \
    --total-timesteps 500000 \
    --save-dir models/dreamer
```

**Training time**: ~1-2 hours on RTX 4090 (sample efficient!)

**When to use**:
- Limited real-world data
- Fast prototyping
- Transfer to new cars/setups

### Meta-RL (MAML)

**Best for**: Multi-track adaptation, quick learning

```bash
# Meta-RL requires special training loop (see advanced usage)
python scripts/train.py \
    --circuit silverstone \
    --algorithm meta-rl \
    --config configs/f1_2024.yaml
```

**When to use**:
- Training on multiple circuits
- Quick adaptation to new tracks
- Few-shot learning scenarios

---

## Evaluating Agents

### Basic Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint models/sac_silverstone_best.zip \
    --circuit silverstone \
    --num-episodes 10 \
    --deterministic
```

### Evaluation with Telemetry Saving

```bash
python scripts/evaluate.py \
    --checkpoint models/sac_silverstone_best.zip \
    --circuit silverstone \
    --num-episodes 5 \
    --deterministic \
    --save-telemetry data/telemetry/sac_silverstone.csv
```

### Cross-Track Evaluation

Test agent trained on Silverstone on Monaco:

```bash
python scripts/evaluate.py \
    --checkpoint models/sac_silverstone_best.zip \
    --circuit monaco \
    --num-episodes 10 \
    --deterministic
```

---

## Configuration

### Creating Custom Configurations

Example: High-downforce setup for Monaco

```yaml
# configs/my_monaco_setup.yaml
circuit: monaco
tire_compound: C5

car:
  min_weight: 798.0
  drag_coefficient: 0.85  # Higher for max downforce
  downforce_coefficient: 4.2  # Maximum downforce
  aero_balance_front: 0.42
  max_brake_force: 22000.0

reward_weights:
  racing_line: 1.0  # Prioritize precision
  speed: 0.2
  crash_penalty: -20.0

sac:
  learning_rate: 0.0001
  batch_size: 128
```

### Configuration Parameters

#### Car Parameters

- `min_weight`: Car weight in kg (798 for 2024 regs)
- `drag_coefficient`: Cd (0.7-0.9 typical)
- `downforce_coefficient`: Cl (3.5-4.5 typical)
- `aero_balance_front`: Front downforce % (0.35-0.45)
- `max_power_ice`: Engine power in Watts (550 kW)
- `max_power_ers_deploy`: ERS power (120 kW)

#### Reward Weights

- `progress`: Forward progress reward (default: 1.0)
- `racing_line`: Deviation penalty (default: 0.5)
- `speed`: Speed reward (default: 0.3)
- `tire_management`: Tire wear penalty (default: 0.3)
- `fuel_efficiency`: Fuel usage penalty (default: 0.2)
- `crash_penalty`: Crash penalty (default: -10.0)

---

## Advanced Usage

### Multi-Environment Training (Parallelization)

```bash
python scripts/train.py \
    --circuit silverstone \
    --algorithm sac \
    --num-envs 8 \
    --total-timesteps 2000000
```

Uses 8 parallel environments for faster training.

### Custom Python Script

```python
from src.envs.f1_racing_env import F1RacingEnv
from src.agents.sac_adaptive import create_sac_agent

# Create environment
env = F1RacingEnv(
    circuit_name='silverstone',
    enable_dynamic_conditions=True
)

# Create agent
agent = create_sac_agent(env)

# Train
agent.train(total_timesteps=1000000)

# Save
agent.save('models/my_agent.zip')

# Evaluate
obs, _ = env.reset()
for _ in range(1000):
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break
```

### Track-Specific Training

```python
# Train on multiple tracks and compare
circuits = ['silverstone', 'monaco', 'spa']

for circuit in circuits:
    env = F1RacingEnv(circuit_name=circuit)
    agent = create_sac_agent(env)
    agent.train(total_timesteps=500000)
    agent.save(f'models/sac_{circuit}.zip')
```

---

## Telemetry Analysis

### Viewing Telemetry

```bash
python scripts/visualize.py --telemetry data/telemetry/lap_data.csv
```

### Analyzing Specific Metrics

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load telemetry
df = pd.read_csv('telemetry.csv')

# Plot tire degradation
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['tire_wear_avg'])
plt.xlabel('Time (s)')
plt.ylabel('Tire Wear (%)')
plt.title('Tire Degradation Over Lap')
plt.grid(True)
plt.show()

# Calculate statistics
print(f"Max speed: {df['speed_kmh'].max():.1f} km/h")
print(f"Fuel used: {df['fuel_kg'].iloc[0] - df['fuel_kg'].iloc[-1]:.2f} kg")
```

---

## Tips & Best Practices

### For Fastest Training

1. Use SAC with multiple parallel environments
2. Start with 100k timesteps for testing
3. Use GPU acceleration
4. Monitor tensorboard for convergence

```bash
tensorboard --logdir logs/
```

### For Best Lap Times

1. Train for 1M+ timesteps
2. Use high `racing_line` reward weight
3. Fine-tune on specific circuit
4. Use deterministic evaluation

### For Race Strategy

1. Use PPO-LSTM for temporal planning
2. Enable dynamic conditions
3. Train for full race distance
4. Weight tire/fuel management highly

### For Multi-Track Performance

1. Use Meta-RL
2. Train on diverse circuits
3. Allow adaptation period
4. Use transfer learning

---

## Troubleshooting

### Issue: Training is slow

**Solution**:
- Use more parallel environments (`--num-envs 8`)
- Ensure GPU is being used (check with `nvidia-smi`)
- Reduce network size in config
- Use faster algorithm (SAC over PPO-LSTM)

### Issue: Agent doesn't learn

**Solution**:
- Check reward signal (tensorboard)
- Increase training timesteps
- Adjust reward weights
- Verify environment is working (test with random agent)

### Issue: Agent crashes frequently

**Solution**:
- Increase `crash_penalty` in config
- Add `off_track_penalty`
- Adjust `racing_line` weight
- Train longer for better convergence

---

## Next Steps

- **Hyperparameter Tuning**: Experiment with learning rates and network architectures
- **Custom Rewards**: Design reward functions for specific objectives
- **Real Data Integration**: Use real F1 telemetry for validation
- **Sim-to-Real**: Transfer learned policies to simulation platforms like Assetto Corsa

---

## Support & Resources

- **Documentation**: See `/docs` directory
- **Examples**: See `/configs` for configuration examples
- **Research Papers**: Check `PLATFORM_RECOMMENDATIONS.md` for references

## Performance Benchmarks

### Expected Lap Times (Silverstone)

| Agent | Training Steps | Lap Time | Comment |
|-------|---------------|----------|---------|
| Random | 0 | N/A | Usually crashes |
| SAC | 100k | ~110s | Learning basics |
| SAC | 500k | ~95s | Competitive |
| SAC | 1M | ~88-92s | Near optimal |
| PPO-LSTM | 2M | ~87-90s | With strategy |
| Expert (Real F1) | - | ~87s | 2020 record |

*Results may vary based on hardware, hyperparameters, and configuration*
