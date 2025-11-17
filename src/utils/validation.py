"""
Validation Tools for Comparing Simulation vs Real F1 Data

Provides comprehensive comparison and analysis tools:
- Lap time validation
- Speed trace comparison
- Tire degradation comparison
- Statistical metrics
- Visualization generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import interpolate, stats
from pathlib import Path


@dataclass
class ValidationMetrics:
    """Container for validation results."""

    # Lap time metrics
    lap_time_error: float  # seconds
    lap_time_error_percent: float  # percentage

    # Speed metrics
    mean_speed_error: float  # km/h
    max_speed_error: float  # km/h
    speed_correlation: float  # Pearson correlation
    rmse_speed: float  # km/h

    # Sector time errors
    sector_1_error: float  # seconds
    sector_2_error: float  # seconds
    sector_3_error: float  # seconds

    # Statistical metrics
    overall_score: float  # 0-100, higher is better
    confidence_level: float  # 0-1


class TelemetryValidator:
    """
    Validates simulation telemetry against real F1 data.

    Provides detailed comparison and scoring.
    """

    def __init__(self, track_name: str = ""):
        self.track_name = track_name
        self.real_data = None
        self.sim_data = None

    def load_real_data(self, telemetry_data, lap_time: float):
        """Load real F1 telemetry data."""
        self.real_data = {
            'telemetry': telemetry_data,
            'lap_time': lap_time
        }

    def load_simulation_data(self, telemetry_data, lap_time: float):
        """Load simulation telemetry data."""
        self.sim_data = {
            'telemetry': telemetry_data,
            'lap_time': lap_time
        }

    def validate(self) -> ValidationMetrics:
        """
        Compare simulation vs real data.

        Returns:
            ValidationMetrics with detailed comparison results
        """
        if self.real_data is None or self.sim_data is None:
            raise ValueError("Must load both real and simulation data first")

        real_telem = self.real_data['telemetry']
        sim_telem = self.sim_data['telemetry']

        # Align data by distance (interpolate to same distance points)
        real_aligned, sim_aligned = self._align_telemetry(real_telem, sim_telem)

        # Lap time comparison
        lap_time_error = abs(self.sim_data['lap_time'] - self.real_data['lap_time'])
        lap_time_error_percent = (lap_time_error / self.real_data['lap_time']) * 100

        # Speed comparison
        speed_errors = sim_aligned['speed'] - real_aligned['speed']
        mean_speed_error = np.mean(np.abs(speed_errors))
        max_speed_error = np.max(np.abs(speed_errors))
        speed_correlation = np.corrcoef(real_aligned['speed'], sim_aligned['speed'])[0, 1]
        rmse_speed = np.sqrt(np.mean(speed_errors**2))

        # Sector times (approximate by dividing track into thirds)
        sector_times_real = self._calculate_sector_times(real_telem)
        sector_times_sim = self._calculate_sector_times(sim_telem)

        sector_1_error = abs(sector_times_sim[0] - sector_times_real[0])
        sector_2_error = abs(sector_times_sim[1] - sector_times_real[1])
        sector_3_error = abs(sector_times_sim[2] - sector_times_real[2])

        # Calculate overall score (0-100)
        overall_score = self._calculate_overall_score(
            lap_time_error_percent,
            speed_correlation,
            rmse_speed
        )

        # Confidence level based on data quality
        confidence_level = self._calculate_confidence(real_aligned, sim_aligned)

        return ValidationMetrics(
            lap_time_error=lap_time_error,
            lap_time_error_percent=lap_time_error_percent,
            mean_speed_error=mean_speed_error,
            max_speed_error=max_speed_error,
            speed_correlation=speed_correlation,
            rmse_speed=rmse_speed,
            sector_1_error=sector_1_error,
            sector_2_error=sector_2_error,
            sector_3_error=sector_3_error,
            overall_score=overall_score,
            confidence_level=confidence_level
        )

    def _align_telemetry(self, real_telem, sim_telem) -> Tuple[Dict, Dict]:
        """Align telemetry by distance for fair comparison."""
        # Create common distance array
        min_dist = max(np.min(real_telem.distance), np.min(sim_telem.distance))
        max_dist = min(np.max(real_telem.distance), np.max(sim_telem.distance))

        # Sample at 10m intervals
        common_distance = np.arange(min_dist, max_dist, 10.0)

        # Interpolate real data
        real_speed_interp = interpolate.interp1d(
            real_telem.distance, real_telem.speed, kind='linear'
        )
        real_throttle_interp = interpolate.interp1d(
            real_telem.distance, real_telem.throttle, kind='linear'
        )
        real_brake_interp = interpolate.interp1d(
            real_telem.distance, real_telem.brake, kind='linear'
        )

        # Interpolate sim data
        sim_speed_interp = interpolate.interp1d(
            sim_telem.distance, sim_telem.speed, kind='linear'
        )
        sim_throttle_interp = interpolate.interp1d(
            sim_telem.distance, sim_telem.throttle, kind='linear'
        )
        sim_brake_interp = interpolate.interp1d(
            sim_telem.distance, sim_telem.brake, kind='linear'
        )

        real_aligned = {
            'distance': common_distance,
            'speed': real_speed_interp(common_distance),
            'throttle': real_throttle_interp(common_distance),
            'brake': real_brake_interp(common_distance),
        }

        sim_aligned = {
            'distance': common_distance,
            'speed': sim_speed_interp(common_distance),
            'throttle': sim_throttle_interp(common_distance),
            'brake': sim_brake_interp(common_distance),
        }

        return real_aligned, sim_aligned

    def _calculate_sector_times(self, telemetry) -> List[float]:
        """Calculate sector times (divide track into thirds)."""
        total_distance = np.max(telemetry.distance)
        sector_boundaries = [0, total_distance/3, 2*total_distance/3, total_distance]

        sector_times = []
        for i in range(3):
            sector_start = sector_boundaries[i]
            sector_end = sector_boundaries[i+1]

            # Find time range for this sector
            start_idx = np.argmin(np.abs(telemetry.distance - sector_start))
            end_idx = np.argmin(np.abs(telemetry.distance - sector_end))

            sector_time = telemetry.time[end_idx] - telemetry.time[start_idx]
            sector_times.append(sector_time)

        return sector_times

    def _calculate_overall_score(
        self,
        lap_time_error_percent: float,
        speed_correlation: float,
        rmse_speed: float
    ) -> float:
        """
        Calculate overall validation score (0-100).

        Higher is better. Weighted combination of metrics.
        """
        # Lap time score (50% weight)
        # Perfect = 0% error, Acceptable = <1% error
        lap_time_score = max(0, 100 - lap_time_error_percent * 100)

        # Speed correlation score (30% weight)
        # Perfect = 1.0 correlation
        speed_corr_score = speed_correlation * 100

        # RMSE score (20% weight)
        # Perfect = 0 km/h, Acceptable = <5 km/h
        rmse_score = max(0, 100 - rmse_speed * 5)

        overall = (
            0.5 * lap_time_score +
            0.3 * speed_corr_score +
            0.2 * rmse_score
        )

        return np.clip(overall, 0, 100)

    def _calculate_confidence(self, real_aligned: Dict, sim_aligned: Dict) -> float:
        """Calculate confidence level based on data quality."""
        # More data points = higher confidence
        n_points = len(real_aligned['distance'])
        point_confidence = min(1.0, n_points / 500)  # 500+ points ideal

        # Consistent sampling = higher confidence
        dist_diff = np.diff(real_aligned['distance'])
        sampling_consistency = 1.0 - np.std(dist_diff) / np.mean(dist_diff)

        # Combined confidence
        confidence = 0.7 * point_confidence + 0.3 * sampling_consistency

        return np.clip(confidence, 0, 1)

    def generate_comparison_plots(self, output_dir: str = "validation_plots"):
        """
        Generate comprehensive comparison plots.

        Creates:
        - Speed trace comparison
        - Throttle/brake comparison
        - Lap time delta plot
        - Statistical summary
        """
        if self.real_data is None or self.sim_data is None:
            raise ValueError("Must load both real and simulation data first")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        real_telem = self.real_data['telemetry']
        sim_telem = self.sim_data['telemetry']

        # Align data
        real_aligned, sim_aligned = self._align_telemetry(real_telem, sim_telem)

        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))

        distance = real_aligned['distance'] / 1000  # Convert to km

        # Plot 1: Speed comparison
        axes[0].plot(distance, real_aligned['speed'], 'b-', label='Real F1', linewidth=2)
        axes[0].plot(distance, sim_aligned['speed'], 'r--', label='Simulation', linewidth=1.5)
        axes[0].set_ylabel('Speed (km/h)', fontsize=11)
        axes[0].set_title(f'Speed Trace Comparison - {self.track_name}', fontsize=13, fontweight='bold')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Speed delta
        speed_delta = sim_aligned['speed'] - real_aligned['speed']
        axes[1].fill_between(distance, 0, speed_delta, where=(speed_delta >= 0),
                             color='green', alpha=0.5, label='Sim faster')
        axes[1].fill_between(distance, 0, speed_delta, where=(speed_delta < 0),
                             color='red', alpha=0.5, label='Sim slower')
        axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1].set_ylabel('Speed Δ (km/h)', fontsize=11)
        axes[1].set_title('Speed Delta (Simulation - Real)', fontsize=12)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Throttle comparison
        axes[2].plot(distance, real_aligned['throttle'], 'b-', label='Real F1', linewidth=2, alpha=0.7)
        axes[2].plot(distance, sim_aligned['throttle'], 'r--', label='Simulation', linewidth=1.5, alpha=0.7)
        axes[2].set_ylabel('Throttle (%)', fontsize=11)
        axes[2].set_title('Throttle Application', fontsize=12)
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Brake comparison
        axes[3].plot(distance, real_aligned['brake'], 'b-', label='Real F1', linewidth=2, alpha=0.7)
        axes[3].plot(distance, sim_aligned['brake'], 'r--', label='Simulation', linewidth=1.5, alpha=0.7)
        axes[3].set_ylabel('Brake (%)', fontsize=11)
        axes[3].set_xlabel('Distance (km)', fontsize=11)
        axes[3].set_title('Brake Application', fontsize=12)
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'telemetry_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path / 'telemetry_comparison.png'}")
        plt.close()

        # Generate statistical summary plot
        self._plot_statistical_summary(real_aligned, sim_aligned, output_path)

    def _plot_statistical_summary(self, real_aligned: Dict, sim_aligned: Dict, output_path: Path):
        """Create statistical summary visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Speed distribution comparison
        axes[0, 0].hist(real_aligned['speed'], bins=30, alpha=0.6, label='Real F1', color='blue', density=True)
        axes[0, 0].hist(sim_aligned['speed'], bins=30, alpha=0.6, label='Simulation', color='red', density=True)
        axes[0, 0].set_xlabel('Speed (km/h)')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Speed Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Speed correlation scatter plot
        axes[0, 1].scatter(real_aligned['speed'], sim_aligned['speed'], alpha=0.3, s=10)
        min_speed = min(np.min(real_aligned['speed']), np.min(sim_aligned['speed']))
        max_speed = max(np.max(real_aligned['speed']), np.max(sim_aligned['speed']))
        axes[0, 1].plot([min_speed, max_speed], [min_speed, max_speed], 'r--', label='Perfect correlation')

        # Add correlation coefficient
        corr = np.corrcoef(real_aligned['speed'], sim_aligned['speed'])[0, 1]
        axes[0, 1].text(0.05, 0.95, f'r = {corr:.4f}', transform=axes[0, 1].transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        axes[0, 1].set_xlabel('Real F1 Speed (km/h)')
        axes[0, 1].set_ylabel('Simulation Speed (km/h)')
        axes[0, 1].set_title('Speed Correlation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Throttle usage comparison
        throttle_bins = np.linspace(0, 100, 21)
        real_throttle_hist, _ = np.histogram(real_aligned['throttle'], bins=throttle_bins)
        sim_throttle_hist, _ = np.histogram(sim_aligned['throttle'], bins=throttle_bins)

        x_pos = np.arange(len(real_throttle_hist))
        width = 0.35
        axes[1, 0].bar(x_pos - width/2, real_throttle_hist, width, label='Real F1', alpha=0.7)
        axes[1, 0].bar(x_pos + width/2, sim_throttle_hist, width, label='Simulation', alpha=0.7)
        axes[1, 0].set_xlabel('Throttle (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Throttle Usage Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Brake usage comparison
        brake_bins = np.linspace(0, 100, 21)
        real_brake_hist, _ = np.histogram(real_aligned['brake'], bins=brake_bins)
        sim_brake_hist, _ = np.histogram(sim_aligned['brake'], bins=brake_bins)

        axes[1, 1].bar(x_pos - width/2, real_brake_hist, width, label='Real F1', alpha=0.7)
        axes[1, 1].bar(x_pos + width/2, sim_brake_hist, width, label='Simulation', alpha=0.7)
        axes[1, 1].set_xlabel('Brake (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Brake Usage Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'statistical_summary.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path / 'statistical_summary.png'}")
        plt.close()

    def generate_report(self, metrics: ValidationMetrics, output_path: str = "validation_report.txt"):
        """Generate detailed text report."""
        report = []
        report.append("=" * 70)
        report.append("SIMULATION VALIDATION REPORT")
        report.append("=" * 70)
        report.append(f"\nTrack: {self.track_name}")
        report.append(f"Real F1 Lap Time: {self.real_data['lap_time']:.3f}s")
        report.append(f"Simulation Lap Time: {self.sim_data['lap_time']:.3f}s")
        report.append("\n" + "-" * 70)
        report.append("LAP TIME ANALYSIS")
        report.append("-" * 70)
        report.append(f"Lap Time Error: {metrics.lap_time_error:.3f}s ({metrics.lap_time_error_percent:.2f}%)")

        if metrics.lap_time_error_percent < 0.5:
            assessment = "EXCELLENT - Within 0.5%"
        elif metrics.lap_time_error_percent < 1.0:
            assessment = "GOOD - Within 1%"
        elif metrics.lap_time_error_percent < 2.0:
            assessment = "ACCEPTABLE - Within 2%"
        else:
            assessment = "NEEDS IMPROVEMENT - Over 2% error"
        report.append(f"Assessment: {assessment}")

        report.append("\n" + "-" * 70)
        report.append("SECTOR TIME ANALYSIS")
        report.append("-" * 70)
        report.append(f"Sector 1 Error: {metrics.sector_1_error:.3f}s")
        report.append(f"Sector 2 Error: {metrics.sector_2_error:.3f}s")
        report.append(f"Sector 3 Error: {metrics.sector_3_error:.3f}s")

        report.append("\n" + "-" * 70)
        report.append("SPEED ANALYSIS")
        report.append("-" * 70)
        report.append(f"Mean Speed Error: {metrics.mean_speed_error:.2f} km/h")
        report.append(f"Max Speed Error: {metrics.max_speed_error:.2f} km/h")
        report.append(f"Speed Correlation: {metrics.speed_correlation:.4f}")
        report.append(f"RMSE Speed: {metrics.rmse_speed:.2f} km/h")

        if metrics.speed_correlation > 0.95:
            speed_assessment = "EXCELLENT - Very high correlation"
        elif metrics.speed_correlation > 0.90:
            speed_assessment = "GOOD - High correlation"
        elif metrics.speed_correlation > 0.85:
            speed_assessment = "ACCEPTABLE - Moderate correlation"
        else:
            speed_assessment = "NEEDS IMPROVEMENT - Low correlation"
        report.append(f"Assessment: {speed_assessment}")

        report.append("\n" + "-" * 70)
        report.append("OVERALL VALIDATION SCORE")
        report.append("-" * 70)
        report.append(f"Overall Score: {metrics.overall_score:.1f}/100")
        report.append(f"Confidence Level: {metrics.confidence_level:.2f}")

        if metrics.overall_score >= 90:
            overall_assessment = "EXCELLENT - Simulation closely matches real F1 data"
        elif metrics.overall_score >= 80:
            overall_assessment = "GOOD - Simulation adequately represents F1 behavior"
        elif metrics.overall_score >= 70:
            overall_assessment = "ACCEPTABLE - Simulation captures key characteristics"
        else:
            overall_assessment = "NEEDS IMPROVEMENT - Significant deviations from real data"
        report.append(f"\nOverall Assessment: {overall_assessment}")

        report.append("\n" + "=" * 70)

        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))

        print(f"\nValidation report saved to: {output_path}")

        # Also print to console
        print('\n'.join(report))


class TireDegradationValidator:
    """Validate tire degradation model against real F1 data."""

    def __init__(self):
        self.real_deg_data = []
        self.sim_deg_data = []

    def add_real_stint(self, laps: List[float], lap_times: List[float], compound: str):
        """Add real F1 tire stint data."""
        self.real_deg_data.append({
            'laps': np.array(laps),
            'lap_times': np.array(lap_times),
            'compound': compound
        })

    def add_sim_stint(self, laps: List[float], lap_times: List[float], compound: str):
        """Add simulation tire stint data."""
        self.sim_deg_data.append({
            'laps': np.array(laps),
            'lap_times': np.array(lap_times),
            'compound': compound
        })

    def validate_degradation_rate(self) -> Dict[str, float]:
        """Calculate degradation rate comparison."""
        if not self.real_deg_data or not self.sim_deg_data:
            raise ValueError("Must add both real and simulation stint data")

        # Calculate degradation slopes (lap time increase per lap)
        real_deg_rate = self._calculate_deg_rate(self.real_deg_data[0])
        sim_deg_rate = self._calculate_deg_rate(self.sim_deg_data[0])

        error = abs(sim_deg_rate - real_deg_rate)
        error_percent = (error / real_deg_rate) * 100 if real_deg_rate > 0 else 0

        return {
            'real_deg_rate': real_deg_rate,
            'sim_deg_rate': sim_deg_rate,
            'error': error,
            'error_percent': error_percent
        }

    def _calculate_deg_rate(self, stint_data: Dict) -> float:
        """Calculate degradation rate (seconds per lap)."""
        laps = stint_data['laps']
        lap_times = stint_data['lap_times']

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(laps, lap_times)

        return slope  # seconds per lap

    def plot_degradation_comparison(self, output_path: str = "tire_degradation.png"):
        """Plot tire degradation comparison."""
        if not self.real_deg_data or not self.sim_deg_data:
            return

        plt.figure(figsize=(10, 6))

        # Plot real data
        for stint in self.real_deg_data:
            plt.plot(stint['laps'], stint['lap_times'], 'bo-',
                    label=f"Real F1 ({stint['compound']})", linewidth=2, markersize=6)

        # Plot sim data
        for stint in self.sim_deg_data:
            plt.plot(stint['laps'], stint['lap_times'], 'r^--',
                    label=f"Simulation ({stint['compound']})", linewidth=1.5, markersize=5)

        plt.xlabel('Lap Number', fontsize=11)
        plt.ylabel('Lap Time (s)', fontsize=11)
        plt.title('Tire Degradation: Real vs Simulation', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


# === EXAMPLE USAGE ===

def example_validate_telemetry():
    """Example: Validate simulation against real F1 telemetry."""
    from src.utils.data_import import TelemetryImporter

    # Import real F1 data
    real_telemetry = TelemetryImporter.from_csv('data/real_f1_lap.csv')
    real_lap_time = real_telemetry.time[-1]

    # Import simulation data
    sim_telemetry = TelemetryImporter.from_csv('data/sim_lap.csv')
    sim_lap_time = sim_telemetry.time[-1]

    # Create validator
    validator = TelemetryValidator(track_name="Silverstone")
    validator.load_real_data(real_telemetry, real_lap_time)
    validator.load_simulation_data(sim_telemetry, sim_lap_time)

    # Validate
    metrics = validator.validate()

    # Generate visualizations
    validator.generate_comparison_plots(output_dir="validation_output")

    # Generate report
    validator.generate_report(metrics, output_path="validation_output/report.txt")

    print(f"\nValidation Score: {metrics.overall_score:.1f}/100")
    print(f"Lap Time Error: {metrics.lap_time_error_percent:.2f}%")
    print(f"Speed Correlation: {metrics.speed_correlation:.4f}")


def example_validate_tire_degradation():
    """Example: Validate tire degradation model."""
    validator = TireDegradationValidator()

    # Real F1 data (example from a race)
    real_laps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    real_times = [88.5, 88.7, 88.9, 89.1, 89.4, 89.7, 90.1, 90.5, 91.0, 91.6]
    validator.add_real_stint(real_laps, real_times, compound="SOFT")

    # Simulation data
    sim_laps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sim_times = [88.6, 88.8, 89.0, 89.3, 89.5, 89.9, 90.2, 90.6, 91.1, 91.7]
    validator.add_sim_stint(sim_laps, sim_times, compound="SOFT")

    # Validate
    results = validator.validate_degradation_rate()

    print(f"Real degradation rate: {results['real_deg_rate']:.4f} s/lap")
    print(f"Sim degradation rate: {results['sim_deg_rate']:.4f} s/lap")
    print(f"Error: {results['error_percent']:.2f}%")

    # Plot
    validator.plot_degradation_comparison(output_path="tire_degradation_comparison.png")


if __name__ == "__main__":
    print("Validation Tools for F1 Simulation")
    print("=" * 50)
    print("\nExamples:")
    print("1. example_validate_telemetry()")
    print("2. example_validate_tire_degradation()")
