# 🏎️ F1 Racing Simulation - QUICK START GUIDE

Get up and running in **5 minutes**!

## 🚀 Fastest Way to Run (3 Steps)

### Step 1: Install Dependencies

```bash
pip install -e .
```

### Step 2: Edit Configuration

Open `config.yaml` and customize your simulation:

```yaml
circuit: silverstone        # Change to: monaco, spa

car_setups:
  - name: High_Downforce
    tire_compound: SOFT     # Change to: C1-C5, MEDIUM, HARD
    front_wing_angle: 15
    # ... edit other parameters
```

### Step 3: Run!

```bash
python RUN_ME.py
```

**That's it!** The system will:
- ✅ Train RL agents to find optimal driving
- ✅ Test all your car setups
- ✅ Generate lap time comparisons
- ✅ Create visualizations and reports

Results saved to: `results/`

---

## 📊 What You Get

After running, check these files:

```
results/
├── visualizations/
│   └── setup_comparison.png      # Bar chart of lap times
├── reports/
│   └── report.txt                 # Detailed comparison report
├── telemetry/
│   └── *_telemetry.json           # Full lap data
└── models/
    └── *.zip                       # Trained RL models
```

---

## 🎯 Common Use Cases

### Test a Single Setup Quickly

Edit `config.yaml`:

```yaml
total_timesteps: 100000    # Fast training (~15 min)
car_setups:
  - name: My_Setup          # Just one setup
    tire_compound: SOFT
    # ...
```

### Compare High vs Low Downforce

```yaml
car_setups:
  - name: High_Downforce
    front_wing_angle: 15
    rear_wing_angle: 12
    tire_compound: SOFT

  - name: Low_Drag
    front_wing_angle: 8
    rear_wing_angle: 6
    tire_compound: SOFT
```

### Test Different Tire Compounds

```yaml
car_setups:
  - name: Soft_Tires
    tire_compound: SOFT

  - name: Medium_Tires
    tire_compound: MEDIUM

  - name: Hard_Tires
    tire_compound: HARD
```

### Qualifying Setup (Light Fuel)

```yaml
car_setups:
  - name: Quali_Setup
    tire_compound: SOFT
    fuel_load: 30           # Minimum fuel
```

---

## 🎛️ Configuration Options Explained

### Circuit Selection

```yaml
circuit: silverstone
```

**Available circuits:**
- `silverstone` - British GP
- `monaco` - Monaco GP
- `spa` - Belgian GP

### Advanced Features (v1.2.0)

```yaml
use_advanced_tire_model: true   # Pacejka MF 6.2 (120+ parameters)
use_advanced_aero: true          # CFD-based aerodynamics
domain_randomization: moderate   # Sim-to-real robustness
```

**Options:**
- `use_advanced_tire_model`: `true` or `false`
- `use_advanced_aero`: `true` or `false`
- `domain_randomization`: `null`, `light`, `moderate`, `heavy`

### Training Settings

```yaml
train_rl_agent: true
algorithm: sac
total_timesteps: 500000
```

**Training time estimates:**
- 100,000 steps: ~15-30 min (quick test)
- 500,000 steps: ~2-3 hours (good quality)
- 1,000,000 steps: ~4-6 hours (best quality)

**Algorithms:**
- `sac`: Soft Actor-Critic (recommended)
- `ppo`: PPO with LSTM

### Car Setup Parameters

```yaml
tire_compound: SOFT          # Tire choice
front_wing_angle: 15         # More angle = more downforce
rear_wing_angle: 12          # More angle = more downforce
ride_height_front: 25        # mm (lower = more downforce, risk porpoising)
ride_height_rear: 35         # mm
fuel_load: 110               # kg
```

**Tire compounds:**
- `C1` or `HARD`: Hardest, longest lasting
- `C2`:
- `C3` or `MEDIUM`: Balanced
- `C4`:
- `C5` or `SOFT`: Softest, fastest but degrades quickly
- `INTER`: Intermediate (wet)
- `WET`: Full wet

---

## 📈 Advanced Usage

### 1. Download Real F1 Data

```bash
# Install FastF1
pip install fastf1

# Download Hamilton's Silverstone qualifying lap
python scripts/download_real_data.py \
  --year 2023 \
  --circuit Silverstone \
  --session Q \
  --driver HAM
```

### 2. Validate Against Real Data

Edit `config.yaml`:

```yaml
validate_against_real_data: true
real_data_path: data/real_f1/silverstone_HAM_2023_q.csv
real_lap_time: 85.093
```

Run:
```bash
python RUN_ME.py
```

### 3. Use Saved Models (Skip Training)

After first run, you have trained models in `results/models/`.

Edit `config.yaml`:
```yaml
train_rl_agent: false    # Don't train, just evaluate
```

Run:
```bash
python RUN_ME.py
```

### 4. Import GPS Track Data

```python
from src.tracks.track_geometry_advanced import AdvancedTrackGeometry
from src.utils.data_import import GPSTrackImporter

# Import from GPX file
gps_points, elevations = GPSTrackImporter.from_gpx('my_track.gpx')

# Create track
track = AdvancedTrackGeometry(name="Custom Track")
track.from_gps_data(gps_points, elevations)

# Optimize racing line
track.optimize_racing_line(method='minimum_time')

# Visualize
from src.utils.visualization import TrackVisualizer
viz = TrackVisualizer(track)
viz.plot_track_layout("my_track_layout.png")
```

---

## 🔍 Understanding Results

### Lap Time Comparison

Results are ranked by lap time:

```
🥇 #1  High_Downforce      85.234s
🥈 #2  Balanced            85.789s
🥉 #3  Low_Drag            86.123s
```

### What the Numbers Mean

**Lap Time Differences:**
- 0.1s difference: Noticeable improvement
- 0.5s difference: Significant improvement
- 1.0s difference: Major improvement

**Real F1 Context:**
- Pole position margin: ~0.1-0.3s
- Q1 to Q3: ~1-2s
- Front to back: ~3-5s

### Reading Visualizations

**setup_comparison.png:**
- Gold bar = best setup
- Lower = faster lap time
- Look for significant gaps (>0.5s)

**speed_comparison.png:**
- Shows speed traces through the lap
- Higher line = faster through that section
- Identifies where setup is faster/slower

---

## 🛠️ Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'xxx'
```

**Fix:**
```bash
pip install -e .
```

### Training Takes Too Long

**Fix:** Reduce timesteps in `config.yaml`:
```yaml
total_timesteps: 100000  # Quick test
```

### Out of Memory

**Fix:** Close other applications or reduce:
```yaml
num_evaluation_laps: 3  # Reduce from 5
```

### Model Not Found

```
Model not found: results/models/xxx.zip
```

**Fix:** Set in `config.yaml`:
```yaml
train_rl_agent: true  # Train new model
```

---

## 📚 Next Steps

### Learn More

- **Full documentation:** See `docs/ADVANCED_FEATURES.md`
- **API reference:** See `README.md`
- **Example scripts:** See `scripts/`

### Advanced Topics

1. **Custom Algorithms:** See `src/algorithms/`
2. **Custom Tracks:** See `src/tracks/track_geometry_advanced.py`
3. **Validation Tools:** See `src/utils/validation.py`
4. **Visualization:** See `src/utils/visualization.py`

### Get Help

- **GitHub Issues:** Report bugs or ask questions
- **Documentation:** Check `docs/` directory
- **Examples:** See `examples/` (coming soon)

---

## 🎓 For Racing Engineers

This system replaces on-track testing with AI-driven simulation:

**Traditional Process:**
1. Design car setup
2. Send to track
3. Driver tests (expensive, time-consuming)
4. Analyze telemetry
5. Repeat

**With This System:**
1. Edit `config.yaml` (car setup)
2. Run `python RUN_ME.py`
3. Get optimal lap time in hours, not days
4. Compare unlimited setups digitally
5. Validate against real data

**Benefits:**
- ✅ Test 10+ setups in parallel
- ✅ No driver required
- ✅ No track time costs
- ✅ Instant feedback
- ✅ Unlimited iterations
- ✅ Validation against real F1 data

---

## 📝 Example Workflow

### Complete Racing Engineer Workflow

```bash
# 1. Download real reference data
pip install fastf1
python scripts/download_real_data.py --circuit Silverstone --driver HAM

# 2. Edit config.yaml - set your car setups

# 3. Run simulation
python RUN_ME.py

# 4. Check results
cat results/reports/report.txt
open results/visualizations/setup_comparison.png

# 5. Iterate - modify config.yaml and run again
```

### Testing for a Race Weekend

**Friday (Practice):**
```yaml
# Test 5 different setups
total_timesteps: 200000  # Medium training
car_setups:
  - High_Downforce
  - Balanced
  - Low_Drag
  # ...
```

**Saturday (Qualifying):**
```yaml
# Optimize best setup from Friday
total_timesteps: 1000000  # Long training
car_setups:
  - Optimal_Setup_Quali  # Light fuel, soft tires
```

**Sunday (Race):**
```yaml
# Race setup with full fuel
car_setups:
  - Race_Setup  # Heavy fuel, medium tires
```

---

## 🏁 Summary

**To run the simulation:**

```bash
# 1. Install
pip install -e .

# 2. Configure
nano config.yaml

# 3. Run
python RUN_ME.py

# 4. Check results
ls results/
```

**That's all you need!** 🚀

The system handles:
- RL agent training
- Optimal driving discovery
- Lap time optimization
- Setup comparison
- Visualization generation

**Happy racing!** 🏎️💨

---

**Version:** 1.2.0
**Last Updated:** 2024-11-17
**License:** MIT
