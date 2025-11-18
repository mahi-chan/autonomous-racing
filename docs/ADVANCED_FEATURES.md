# Advanced Features - F1 Racing RL System v1.2.0

This document describes the advanced features introduced in version 1.2.0, bringing the simulation closer to F1-grade realism.

## Table of Contents

1. [Advanced Tire Model](#advanced-tire-model)
2. [CFD-Based Aerodynamics](#cfd-based-aerodynamics)
3. [High-Precision Track Geometry](#high-precision-track-geometry)
4. [Real Data Import & Validation](#real-data-import--validation)
5. [Visualization Tools](#visualization-tools)
6. [Usage Examples](#usage-examples)

---

## Advanced Tire Model

**File:** `src/physics/tire_model_advanced.py`

### Overview

The advanced tire model implements the **Pacejka Magic Formula 6.2** with **120+ parameters**, providing F1-grade tire physics simulation.

### Key Features

#### 1. Extended Pacejka Magic Formula 6.2
- **Longitudinal Force (Fx):** 25+ parameters for pure longitudinal slip
- **Lateral Force (Fy):** 30+ parameters for pure lateral slip
- **Combined Slip:** Advanced interaction between Fx and Fy
- **Aligning Moment (Mz):** Self-aligning torque with pneumatic trail
- **Overturning Moment (Mx):** Camber-induced rolling resistance

#### 2. Temperature Effects
```python
# Temperature influences on grip
- Optimal temperature: 80-110°C
- Grip multiplier: 0.7 at 50°C → 1.0 at 95°C → 0.85 at 130°C
- Thermal degradation accelerates above 110°C
```

#### 3. Pressure Effects
```python
# Pressure influences on tire behavior
- Nominal pressure: 19.5-23 psi (F1 regulations)
- Under-inflation: Increased contact patch, higher degradation
- Over-inflation: Reduced contact patch, faster warm-up
```

#### 4. Camber Effects
- Optimal camber: -2.5° to -3.5° (F1 typical)
- Camber thrust in corners
- Temperature distribution across tire width

#### 5. Transient Behavior
- **Relaxation length:** Tire force buildup delay
- **Dynamic response:** First-order lag model
- Realistic slip transition

#### 6. Degradation Model
```python
# Multi-factor degradation
- Thermal degradation: High temps accelerate wear
- Mechanical degradation: Sliding wear
- Cliff behavior: Rapid performance drop at end of life
- Compound-specific rates (C1-C5, INTER, WET)
```

### Usage Example

```python
from src.physics.tire_model_advanced import AdvancedTireModel

# Create tire
tire = AdvancedTireModel(compound="SOFT")

# Calculate forces
Fx, Fy, Mz, Mx = tire.calculate_forces(
    Fz=5000.0,          # Vertical load (N)
    kappa=0.08,         # Longitudinal slip ratio
    alpha=np.deg2rad(3),# Slip angle (rad)
    gamma=np.deg2rad(-3),# Camber angle (rad)
    V=80.0,             # Velocity (m/s)
    dt=0.01             # Time step
)

# Update temperature and degradation
tire.update_state(Fx, Fy, sliding_velocity=2.0, dt=0.01)
```

### Performance Characteristics

| Compound | Grip Peak | Degradation Rate | Optimal Temp |
|----------|-----------|------------------|--------------|
| C1 (Hard)| 0.85      | 0.02%/lap       | 90-100°C     |
| C3 (Medium)| 0.95   | 0.08%/lap       | 85-95°C      |
| C5 (Soft)| 1.00      | 0.15%/lap       | 80-90°C      |
| INTER    | 0.70      | 0.05%/lap       | 60-80°C      |
| WET      | 0.55      | 0.03%/lap       | 50-70°C      |

---

## CFD-Based Aerodynamics

**File:** `src/physics/aerodynamics_advanced.py`

### Overview

Multi-dimensional lookup tables based on CFD simulation data, capturing complex ground effect behavior.

### Key Features

#### 1. 4D Aerodynamic Maps
```python
# Interpolation dimensions:
- Ride height (10-80mm)
- Rake angle (-0.5° to 1.5°)
- Yaw angle (-10° to 10°)
- Speed (50-350 km/h)
```

#### 2. Ground Effect Simulation
- **Venturi tunnels:** High sensitivity to ride height
- **Floor suction:** Peak downforce at 20-30mm ride height
- **Ride height cliff:** Sudden downforce loss below 15mm
- **Porpoising detection:** Oscillation at low ride height + high speed

#### 3. DRS (Drag Reduction System)
```python
# DRS effects:
- Drag reduction: -15% to -25%
- Downforce loss: -10% to -15%
- Rear balance shift
- Speed-dependent effectiveness
```

#### 4. Aero Balance
- Front/rear downforce distribution
- Pitch moment sensitivity
- Balance shift with ride height and rake

#### 5. Yaw Sensitivity
- Side force generation
- Asymmetric downforce in cornering
- Yaw moment for stability

#### 6. Wake Effects
```python
# Following car effects:
- Downforce loss: 20-40% at 1 car length
- Drag reduction: 10-20% (slipstream)
- Balance shift (more understeer)
```

### Usage Example

```python
from src.physics.aerodynamics_advanced import AdvancedAeroModel

# Create aero model
aero = AdvancedAeroModel()

# Calculate forces
downforce, drag, side_force, pitch_moment, roll_moment, yaw_moment = aero.calculate_forces(
    speed=250.0 / 3.6,           # m/s
    ride_height_front=0.025,     # 25mm
    ride_height_rear=0.035,      # 35mm (10mm rake)
    yaw_angle=np.deg2rad(2.0),   # 2° yaw
    drs_active=False,
    pitch_angle=np.deg2rad(0.3),
    roll_angle=0.0
)

# Check for porpoising
is_porpoising = aero.detect_porpoising(speed, ride_height_front)

# Calculate wake effects
downforce_loss, drag_reduction = aero.calculate_wake_effect(
    distance_to_car_ahead=10.0  # 10 meters
)
```

### Aero Map Characteristics

| Ride Height | Downforce | Drag  | Balance | Notes              |
|-------------|-----------|-------|---------|-------------------|
| 10mm        | Very High | High  | 48% F   | Unstable, risk of porpoising |
| 25mm        | High      | Medium| 50% F   | Optimal performance|
| 40mm        | Medium    | Medium| 52% F   | Stable setup       |
| 60mm        | Low       | Low   | 55% F   | High-speed setup   |

---

## High-Precision Track Geometry

**File:** `src/tracks/track_geometry_advanced.py`

### Overview

Spline-based track representation with GPS import capability, matching laser scan precision.

### Key Features

#### 1. Cubic Spline Interpolation
```python
# Smooth geometry representation:
- Position splines (x, y, z)
- High-resolution sampling (0.5m intervals)
- Continuous curvature calculation
- Smooth derivatives for vehicle dynamics
```

#### 2. GPS Data Import
```python
# Supported formats:
- GPX (GPS Exchange Format)
- CSV (lat, lon, elevation, width, banking)
- Direct coordinate arrays
```

#### 3. Track Properties
- **Elevation:** Full 3D elevation profile
- **Banking:** Corner banking angles
- **Width:** Variable track width
- **Grip variation:** Track evolution, rubber buildup
- **Temperature:** Surface temperature variations

#### 4. Racing Line Optimization
```python
# Three methods:
1. Geometric: Maximum radius through corners
2. Minimum Curvature: Smoothest path
3. Minimum Time: Physics-based optimal path
```

#### 5. Speed Profile Calculation
```python
# Constraints:
- Maximum lateral G (5.0g for F1)
- Maximum longitudinal acceleration (±4.0g)
- Forward/backward passes for continuity
```

### Usage Example

```python
from src.tracks.track_geometry_advanced import AdvancedTrackGeometry

# Create track from GPS data
track = AdvancedTrackGeometry(name="Silverstone")

# Import GPS points
gps_points = np.array([
    [52.0719, -1.0175],  # lat, lon
    # ... more points
])
elevations = np.array([150.0, 151.2, ...])  # meters

track.from_gps_data(
    gps_points=gps_points,
    elevations=elevations,
    smooth_factor=0.0001
)

# Optimize racing line
track.optimize_racing_line(method='minimum_curvature')

# Calculate optimal speeds
track.calculate_optimal_speeds(max_lateral_g=5.0)

# Query track info
nearest_seg, deviation, distance = track.find_nearest_racing_line_point((x, y))
segment = track.get_segment_at_distance(distance)

print(f"Curvature: {segment.curvature:.4f} 1/m")
print(f"Optimal speed: {segment.optimal_speed:.1f} km/h")
print(f"Elevation: {segment.elevation:.1f} m")
print(f"Banking: {np.rad2deg(segment.banking):.1f}°")
```

### Track Statistics Example (Silverstone)

```
Total Length: 5,891 m
Elevation Change: 15.3 m
Corners: 18
Longest Straight: 770 m (Hangar Straight)
Highest Speed: 320 km/h
Lowest Speed: 95 km/h (Copse)
```

---

## Real Data Import & Validation

**File:** `src/utils/data_import.py`, `src/utils/validation.py`

### Data Import Capabilities

#### 1. Telemetry Import
```python
from src.utils.data_import import TelemetryImporter

# From CSV with auto-detection
telemetry = TelemetryImporter.from_csv('lap_data.csv')

# From F1 game export
telemetry = TelemetryImporter.from_f1_game('f1_2024_lap.csv')

# Access data
print(f"Max speed: {np.max(telemetry.speed)} km/h")
print(f"Lap time: {telemetry.time[-1]:.3f} s")
```

#### 2. GPS Track Import
```python
from src.utils.data_import import GPSTrackImporter

# From GPX file
gps_points, elevations = GPSTrackImporter.from_gpx('silverstone.gpx')

# From CSV
gps_points, elevations = GPSTrackImporter.from_csv('track_coords.csv')
```

#### 3. Setup Import
```python
from src.utils.data_import import SetupImporter

# From JSON
setup = SetupImporter.from_json('car_setup.json')

# From F1 game setup file
setup = SetupImporter.from_f1_game_setup('setup.xml')
```

### Validation Tools

#### 1. Telemetry Validation
```python
from src.utils.validation import TelemetryValidator

validator = TelemetryValidator(track_name="Silverstone")
validator.load_real_data(real_telemetry, real_lap_time)
validator.load_simulation_data(sim_telemetry, sim_lap_time)

# Validate
metrics = validator.validate()

print(f"Lap time error: {metrics.lap_time_error_percent:.2f}%")
print(f"Speed correlation: {metrics.speed_correlation:.4f}")
print(f"Overall score: {metrics.overall_score:.1f}/100")

# Generate visualizations
validator.generate_comparison_plots(output_dir="validation_plots")
validator.generate_report(metrics, output_path="report.txt")
```

#### 2. Tire Degradation Validation
```python
from src.utils.validation import TireDegradationValidator

validator = TireDegradationValidator()

# Add real F1 data
validator.add_real_stint(
    laps=[1, 2, 3, 4, 5, 6, 7, 8],
    lap_times=[88.5, 88.7, 89.0, 89.3, 89.7, 90.2, 90.7, 91.3],
    compound="SOFT"
)

# Add simulation data
validator.add_sim_stint(
    laps=[1, 2, 3, 4, 5, 6, 7, 8],
    lap_times=[88.6, 88.8, 89.1, 89.4, 89.8, 90.3, 90.8, 91.4],
    compound="SOFT"
)

# Validate
results = validator.validate_degradation_rate()
print(f"Degradation rate error: {results['error_percent']:.2f}%")

# Plot
validator.plot_degradation_comparison()
```

### Validation Metrics

#### Score Interpretation

| Score   | Assessment | Lap Time Error | Speed Correlation |
|---------|-----------|----------------|-------------------|
| 90-100  | EXCELLENT | < 0.5%         | > 0.95            |
| 80-89   | GOOD      | < 1.0%         | > 0.90            |
| 70-79   | ACCEPTABLE| < 2.0%         | > 0.85            |
| < 70    | NEEDS WORK| > 2.0%         | < 0.85            |

---

## Visualization Tools

**File:** `src/utils/visualization.py`

### Available Visualizations

#### 1. Tire Model Visualizations
```python
from src.utils.visualization import TireModelVisualizer
from src.physics.tire_model_advanced import AdvancedTireModel

tire = AdvancedTireModel(compound="SOFT")
viz = TireModelVisualizer(tire)

# Friction circle (combined slip limits)
viz.plot_friction_circle("tire_friction_circle.png")

# Load sensitivity curves
viz.plot_load_sensitivity("tire_load_sensitivity.png")

# Classic slip curves (Fx vs κ, Fy vs α)
viz.plot_slip_curves("tire_slip_curves.png")
```

#### 2. Aerodynamics Visualizations
```python
from src.utils.visualization import AeroVisualizer
from src.physics.aerodynamics_advanced import AdvancedAeroModel

aero = AdvancedAeroModel()
viz = AeroVisualizer(aero)

# Downforce and drag maps (Cl, Cd vs ride height & rake)
viz.plot_downforce_map("aero_downforce_map.png")

# Aero balance vs ride height
viz.plot_aero_balance("aero_balance.png")
```

#### 3. Track Visualizations
```python
from src.utils.visualization import TrackVisualizer

viz = TrackVisualizer(track)

# 2D track layout with racing line
viz.plot_track_layout("track_layout.png")

# Elevation, curvature, and speed profiles
viz.plot_track_profile("track_profile.png")
```

#### 4. Training Progress
```python
from src.utils.visualization import TrainingVisualizer

TrainingVisualizer.plot_training_curves(
    episode_rewards=rewards,
    episode_lengths=lengths,
    lap_times=lap_times,
    output_path="training_progress.png"
)
```

---

## Usage Examples

### Complete Workflow: GPS Track → Validation

```python
# 1. Import track from GPS
from src.tracks.track_geometry_advanced import AdvancedTrackGeometry
from src.utils.data_import import GPSTrackImporter

gps_points, elevations = GPSTrackImporter.from_gpx('data/silverstone.gpx')

track = AdvancedTrackGeometry(name="Silverstone")
track.from_gps_data(gps_points, elevations)
track.optimize_racing_line(method='minimum_time')
track.calculate_optimal_speeds(max_lateral_g=5.0)

# 2. Create environment with advanced physics
from src.envs.f1_racing_env import F1RacingEnv

env = F1RacingEnv(
    circuit_name='silverstone',
    use_advanced_tire_model=True,
    use_advanced_aero=True
)

# 3. Train agent
from src.algorithms.sac_adaptive import train_sac

model = train_sac(
    env=env,
    total_timesteps=1_000_000,
    save_path='models/sac_silverstone_advanced'
)

# 4. Evaluate and extract telemetry
obs, _ = env.reset()
sim_telemetry_data = []

for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    sim_telemetry_data.append({
        'time': info['time'],
        'distance': info['distance'],
        'speed': info['speed'],
        'throttle': action[0] * 100,
        'brake': action[1] * 100,
        # ... more data
    })

    if terminated or truncated:
        break

# 5. Import real F1 data
from src.utils.data_import import TelemetryImporter

real_telemetry = TelemetryImporter.from_csv('data/hamilton_silverstone_2023.csv')

# 6. Validate
from src.utils.validation import TelemetryValidator

validator = TelemetryValidator(track_name="Silverstone")
validator.load_real_data(real_telemetry, real_lap_time=85.093)
validator.load_simulation_data(sim_telemetry, sim_lap_time=sim_lap_time)

metrics = validator.validate()
validator.generate_comparison_plots(output_dir="validation_silverstone")
validator.generate_report(metrics)

# 7. Visualize advanced models
from src.utils.visualization import (
    TireModelVisualizer, AeroVisualizer, TrackVisualizer
)

# Tire behavior
tire_viz = TireModelVisualizer(env.car.tire_fl)
tire_viz.plot_friction_circle("output/tire_friction.png")

# Aero maps
aero_viz = AeroVisualizer(env.car.aero_model)
aero_viz.plot_downforce_map("output/aero_map.png")

# Track analysis
track_viz = TrackVisualizer(track)
track_viz.plot_track_layout("output/silverstone_layout.png")
track_viz.plot_track_profile("output/silverstone_profile.png")
```

---

## Performance Comparison: Standard vs Advanced Models

### Tire Model

| Metric | Standard | Advanced | Improvement |
|--------|----------|----------|-------------|
| Parameters | 15 | 120+ | 8x |
| Temperature effects | Simplified | Full thermal model | ✓ |
| Pressure effects | None | Full model | ✓ |
| Combined slip | Simplified | MF 6.2 | ✓ |
| Transient response | None | Relaxation length | ✓ |
| Degradation | Linear | Multi-factor cliff | ✓ |

### Aerodynamics

| Metric | Standard | Advanced | Improvement |
|--------|----------|----------|-------------|
| Map dimensions | 1D (speed) | 4D (RH, rake, yaw, speed) | 4x |
| Ground effect | Simplified | CFD-based | ✓ |
| DRS modeling | Basic | Speed-dependent | ✓ |
| Wake effects | None | Full model | ✓ |
| Porpoising | None | Detection + simulation | ✓ |

### Track Geometry

| Metric | Standard | Advanced | Improvement |
|--------|----------|----------|-------------|
| Representation | Linear/circular | Cubic splines | ✓ |
| Resolution | ~50 segments | 0.5m sampling | 100x |
| Elevation | Simplified | Full 3D profile | ✓ |
| GPS import | None | GPX/CSV support | ✓ |
| Racing line | Basic | 3 optimization methods | ✓ |

---

## Integration with Existing System

The advanced features are designed to integrate seamlessly:

```python
# Standard usage (backward compatible)
from src.envs.f1_racing_env import F1RacingEnv
env = F1RacingEnv(circuit_name='silverstone')

# Advanced usage (opt-in)
env = F1RacingEnv(
    circuit_name='silverstone',
    use_advanced_tire_model=True,      # Enable advanced tires
    use_advanced_aero=True,            # Enable CFD aero
    use_high_precision_track=True,     # Enable spline track
    domain_randomization='moderate'    # Enable DR for robustness
)
```

---

## Computational Cost

### Performance Impact

| Feature | CPU Impact | Memory Impact | Recommended Use |
|---------|-----------|---------------|-----------------|
| Advanced Tire | +15% | +10 MB | Always (realistic) |
| Advanced Aero | +25% | +50 MB | Training & validation |
| Spline Track | +5% | +20 MB | Always (better quality) |
| Domain Randomization | +10% | +5 MB | Training only |

### Optimization Tips

1. **Pre-compute aero maps:** Load once, reuse across episodes
2. **Cache track queries:** Use distance-based lookups
3. **Batch tire calculations:** Vectorize when possible
4. **Reduce sampling:** Use 1m track resolution for prototyping

---

## References & Resources

### Tire Modeling
- Pacejka, H. B. (2012). *Tire and Vehicle Dynamics*. 3rd Edition.
- MF-Tyre/MF-Swift 6.2 Documentation

### Aerodynamics
- Katz, J. (1995). *Race Car Aerodynamics*
- F1 Technical Regulations 2024 - Aerodynamic Testing

### Track Data
- GPS Exchange Format (GPX) 1.1 Specification
- FIA Circuit Homologation Standards

---

## Future Enhancements

Planned for v1.3.0:
- [ ] Real-time aero CFD (reduced-order models)
- [ ] Machine-learned tire model from F1 data
- [ ] Multi-car race simulations with slipstream
- [ ] Weather effects (rain, wind, temperature)
- [ ] Pit stop strategy optimization
- [ ] Driver-in-the-loop integration

---

## Support

For questions or issues with advanced features:
- GitHub Issues: https://github.com/yourusername/autonomous-racing/issues
- Documentation: See `docs/` directory
- Examples: See `examples/advanced/` directory

---

**Version:** 1.2.0
**Last Updated:** 2024-11-17
**License:** MIT
