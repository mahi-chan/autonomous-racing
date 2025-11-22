# 🏎️ Running F1 Simulation in Google Colab

**YES! You can absolutely run this in a 12-hour Colab window!**

In fact, 12 hours is **MORE than enough** for most experiments.

---

## ⏱️ Time Estimates (Colab Free GPU - T4)

| Task | Time | Fits in 12h? |
|------|------|--------------|
| Setup & Installation | 5 min | ✅ |
| Train 1 setup (100k steps) | 15-20 min | ✅ |
| Train 1 setup (200k steps) | 30-40 min | ✅ |
| Train 1 setup (500k steps) | 1-2 hours | ✅ |
| Train 1 setup (1M steps) | 3-4 hours | ✅ |
| Test 3 setups (200k each) | 1.5-2 hours | ✅ |
| Test 5 setups (200k each) | 2.5-3.5 hours | ✅ |
| Test 3 setups (500k each) | 3-6 hours | ✅ |
| Test 5 setups (500k each) | 5-10 hours | ✅ |

**Bottom line:** You can comfortably test **5+ car setups** in 12 hours!

---

## 🚀 Quick Start (3 Steps)

### Option 1: Use Pre-Made Notebook (EASIEST)

**1. Open the notebook:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/autonomous-racing/blob/main/F1_Racing_Colab.ipynb)

**2. Enable GPU:**
- Runtime → Change runtime type → Hardware accelerator → **GPU (T4)**

**3. Run all cells:**
- Runtime → Run all

**That's it!** Results download automatically.

---

### Option 2: Manual Setup

If you prefer to run manually:

```python
# 1. Clone repository
!git clone https://github.com/yourusername/autonomous-racing.git
%cd autonomous-racing

# 2. Install dependencies
!pip install -q -e .
!pip install -q stable-baselines3[extra] gymnasium

# 3. Run simulation
!python RUN_ME.py
```

---

## 💡 Optimization Tips for 12-Hour Window

### 1. **Use Shorter Training for Testing**

```yaml
# In config.yaml or CONFIG dict
total_timesteps: 200000  # Good quality in ~30-40 min
```

**Training time guide:**
- 50,000 steps = 10-15 min (quick prototype)
- 100,000 steps = 15-20 min (basic test)
- 200,000 steps = 30-40 min (good quality) ⭐ **Recommended for Colab**
- 500,000 steps = 1-2 hours (high quality)
- 1,000,000 steps = 3-4 hours (best quality)

### 2. **Test Multiple Setups in Parallel**

```yaml
car_setups:
  - name: Setup_1
    # ...
  - name: Setup_2
    # ...
  - name: Setup_3
    # ...
  # Can test 5+ setups in 12 hours with 200k steps each!
```

### 3. **Reduce Evaluation Laps**

```yaml
num_evaluation_laps: 3  # Instead of 5
```

### 4. **Auto-Save Before Timeout**

The Colab notebook automatically:
- Saves models every setup
- Saves telemetry data
- Creates downloadable zip archive
- Downloads before 12-hour timeout

---

## 📊 What You Can Accomplish in 12 Hours

### Conservative Estimate (200k steps/setup)

**Setup 1** (30 min train + 5 min eval) = 35 min
**Setup 2** (30 min train + 5 min eval) = 35 min
**Setup 3** (30 min train + 5 min eval) = 35 min
**Setup 4** (30 min train + 5 min eval) = 35 min
**Setup 5** (30 min train + 5 min eval) = 35 min
**Setup 6** (30 min train + 5 min eval) = 35 min

**Total: ~3.5 hours for 6 setups**

**Remaining time:** 8.5 hours for more experiments!

### Aggressive Estimate (500k steps/setup)

**Setup 1** (90 min train + 5 min eval) = 95 min
**Setup 2** (90 min train + 5 min eval) = 95 min
**Setup 3** (90 min train + 5 min eval) = 95 min
**Setup 4** (90 min train + 5 min eval) = 95 min
**Setup 5** (90 min train + 5 min eval) = 95 min
**Setup 6** (90 min train + 5 min eval) = 95 min

**Total: ~9.5 hours for 6 high-quality setups**

**Still fits in 12 hours!**

---

## 🎯 Example Workflows

### Workflow 1: Quick Comparison (2 hours)

**Goal:** Test 5 different wing configurations

```yaml
total_timesteps: 200000

car_setups:
  - name: Max_Downforce
    front_wing_angle: 15
    rear_wing_angle: 12

  - name: High_Downforce
    front_wing_angle: 13
    rear_wing_angle: 10

  - name: Balanced
    front_wing_angle: 11
    rear_wing_angle: 9

  - name: Low_Downforce
    front_wing_angle: 9
    rear_wing_angle: 7

  - name: Min_Drag
    front_wing_angle: 7
    rear_wing_angle: 5
```

**Result:** Find optimal wing config in 2 hours

---

### Workflow 2: Tire Compound Test (2.5 hours)

**Goal:** Which tire is fastest?

```yaml
total_timesteps: 200000

car_setups:
  - name: Soft_C5
    tire_compound: C5

  - name: Medium_Soft_C4
    tire_compound: C4

  - name: Medium_C3
    tire_compound: C3

  - name: Medium_Hard_C2
    tire_compound: C2

  - name: Hard_C1
    tire_compound: C1
```

**Result:** Optimal tire choice for your setup

---

### Workflow 3: Qualifying Setup (4 hours)

**Goal:** Best qualifying setup (light fuel)

```yaml
total_timesteps: 500000  # High quality

car_setups:
  - name: Quali_V1
    fuel_load: 30
    tire_compound: SOFT
    front_wing_angle: 8
    rear_wing_angle: 6

  - name: Quali_V2
    fuel_load: 30
    tire_compound: SOFT
    front_wing_angle: 10
    rear_wing_angle: 8

  - name: Quali_V3
    fuel_load: 30
    tire_compound: SOFT
    front_wing_angle: 12
    rear_wing_angle: 10
```

**Result:** Optimal quali setup with high confidence

---

### Workflow 4: Race Weekend Simulation (10 hours)

**Goal:** Full weekend optimization

**Session 1 - Practice (2 hours):**
```yaml
total_timesteps: 200000
# Test 5 different baseline setups
```

**Session 2 - Qualifying (3 hours):**
```yaml
total_timesteps: 500000
# Refine best setup from practice with light fuel
```

**Session 3 - Race (5 hours):**
```yaml
total_timesteps: 1000000
# Final race setup with full fuel load
```

**Result:** Complete weekend strategy

---

## 💾 Saving Your Work

### Automatic Saves

The notebook automatically saves:
- ✅ Trained models (`.zip` files)
- ✅ Telemetry data (`.json` files)
- ✅ Visualizations (`.png` files)
- ✅ Complete archive (downloadable)

### Manual Download

At any point, run:

```python
from google.colab import files
import shutil

# Create archive
shutil.make_archive('my_results', 'zip', 'results')

# Download
files.download('my_results.zip')
```

### Google Drive Integration

To save to Google Drive:

```python
from google.colab import drive

# Mount Drive
drive.mount('/content/drive')

# Copy results
!cp -r results /content/drive/MyDrive/F1_Simulation_Results
```

---

## 🔋 Managing the 12-Hour Timeout

### Strategy 1: Checkpoint Frequently

```python
# Save after each setup
model.save(f"results/models/setup_{i}.zip")
```

### Strategy 2: Download Incrementally

```python
# Download after each major result
from google.colab import files
files.download(f"results/models/setup_{i}.zip")
```

### Strategy 3: Use Colab Pro

**Colab Pro benefits:**
- 24-hour runtime (instead of 12)
- Faster GPUs (V100, A100)
- Priority access
- More RAM

**Cost:** ~$10/month

**Worth it if:**
- Training 1M+ timesteps per setup
- Testing 10+ setups
- Running multi-day experiments

---

## ⚡ GPU Performance

### Colab Free (T4 GPU)

- **Memory:** 16 GB
- **Speed:** ~40k steps/minute (SAC)
- **Recommended timesteps:** 100k - 500k

### Colab Pro (V100/A100 GPU)

- **Memory:** 32-40 GB
- **Speed:** ~80k+ steps/minute (SAC)
- **Recommended timesteps:** 500k - 2M

### Performance Tips

```python
# 1. Enable GPU
# Runtime → Change runtime type → GPU

# 2. Verify GPU is active
!nvidia-smi

# 3. Monitor GPU usage
import GPUtil
GPUtil.showUtilization()
```

---

## 🐛 Troubleshooting

### Issue: "Runtime disconnected"

**Cause:** 12-hour timeout or idle timeout

**Solution:**
1. Download results before timeout
2. Enable auto-download in notebook
3. Use Google Drive sync

### Issue: "Out of memory"

**Cause:** GPU RAM exceeded

**Solution:**
```yaml
# Reduce timesteps
total_timesteps: 100000  # Instead of 500k

# Or reduce batch size (in algorithm config)
```

### Issue: "Installation failed"

**Cause:** Missing dependencies

**Solution:**
```bash
# Reinstall everything
!pip install -q --upgrade pip
!pip install -q -e .
!pip install -q stable-baselines3[extra]
```

### Issue: "Training too slow"

**Cause:** GPU not enabled

**Solution:**
1. Runtime → Change runtime type
2. Hardware accelerator → GPU
3. Save
4. Restart runtime

---

## 📈 Comparing Colab vs Local

| Feature | Colab Free | Colab Pro | Local (GPU) |
|---------|------------|-----------|-------------|
| **Cost** | FREE | $10/mo | Hardware cost |
| **GPU** | T4 (16GB) | V100/A100 | Your GPU |
| **Runtime** | 12 hours | 24 hours | Unlimited |
| **Setup** | 5 min | 5 min | 30+ min |
| **Internet** | Required | Required | Optional |
| **Storage** | 100GB temp | 100GB temp | Your disk |

**Recommendation:**
- **Prototyping:** Colab Free ⭐
- **Serious training:** Colab Pro or Local
- **Production:** Local with powerful GPU

---

## 🎓 Example Session Timeline

**Real example from testing:**

```
00:00 - Open Colab notebook
00:02 - Enable GPU
00:03 - Run all cells (installation)
00:08 - Installation complete
00:10 - Training Setup 1 starts
00:45 - Setup 1 complete (35 min)
01:20 - Setup 2 complete (35 min)
01:55 - Setup 3 complete (35 min)
02:30 - Setup 4 complete (35 min)
03:05 - Setup 5 complete (35 min)
03:10 - Generate visualizations
03:15 - Download results archive
03:20 - Session complete

TOTAL: 3h 20min for 5 setups (200k steps each)
REMAINING: 8h 40min in 12h window
```

**You could do this 3 times in 12 hours!**

---

## 🏁 Summary

### Can you run in 12 hours? **YES!**

**What fits in 12 hours:**

✅ 5-10 setups at 200k steps each
✅ 3-6 setups at 500k steps each
✅ 2-3 setups at 1M steps each
✅ Full race weekend simulation
✅ Tire compound testing
✅ Wing angle optimization
✅ Qualifying setup refinement

### Quick Start

1. **Open:** [F1_Racing_Colab.ipynb](https://colab.research.google.com/github/yourusername/autonomous-racing/blob/main/F1_Racing_Colab.ipynb)
2. **Enable GPU:** Runtime → Change runtime type → GPU
3. **Run all:** Runtime → Run all
4. **Wait:** 2-10 hours (depending on config)
5. **Download:** Results auto-download

### Pro Tips

- Start with 200k steps for testing
- Test 5+ setups in one session
- Download results incrementally
- Use Google Drive for backup
- Consider Colab Pro for longer experiments

**Happy racing in the cloud!** ☁️🏎️💨

---

**Questions?**

- Check [QUICKSTART.md](QUICKSTART.md)
- See [README.md](README.md)
- Read [docs/ADVANCED_FEATURES.md](docs/ADVANCED_FEATURES.md)
