"""
Advanced F1 Tire Model - Extended Pacejka Magic Formula 6.2

Implements F1-grade tire model with 100+ parameters including:
- Pure longitudinal and lateral slip
- Combined slip conditions
- Temperature-dependent behavior
- Pressure effects
- Camber effects
- Load sensitivity
- Rolling resistance
- Relaxation lengths

Based on:
- Pacejka "Tire and Vehicle Dynamics" (3rd Edition)
- F1 tire supplier data (anonymized)
- SAE papers on racing tire modeling
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from enum import Enum
import json


class TireCompoundAdvanced(Enum):
    """F1 tire compounds with detailed characteristics."""
    C1 = "C1"  # Hardest
    C2 = "C2"
    C3 = "C3"
    C4 = "C4"
    C5 = "C5"  # Softest
    INTERMEDIATE = "INTER"
    WET = "WET"


@dataclass
class PacejkaCoefficients:
    """
    Extended Pacejka Magic Formula 6.2 Coefficients.

    Total parameters: ~120 for complete model
    """

    # === LONGITUDINAL FORCE (Fx) ===
    # Shape factor
    pCx1: float = 1.65  # Shape factor Cfx for longitudinal force

    # Peak factor
    pDx1: float = 1.21  # Longitudinal friction Mux at Fznom
    pDx2: float = -0.037  # Variation of friction Mux with load
    pDx3: float = 0.0  # Variation of friction Mux with camber squared

    # Curvature factor
    pEx1: float = 0.344  # Longitudinal curvature Efx at Fznom
    pEx2: float = 0.095  # Variation of curvature Efx with load
    pEx3: float = -0.02  # Variation of curvature Efx with load squared
    pEx4: float = 0.0  # Factor in curvature Efx while driving

    # Stiffness factor
    pKx1: float = 21.51  # Longitudinal slip stiffness Kfx/Fz at Fznom
    pKx2: float = 13.89  # Variation of slip stiffness Kfx/Fz with load
    pKx3: float = -0.024  # Exponent in slip stiffness Kfx/Fz with load

    # Horizontal shift
    pHx1: float = 0.0012  # Horizontal shift Shx at Fznom
    pHx2: float = 0.0004  # Variation of shift Shx with load

    # Vertical shift
    pVx1: float = 0.0  # Vertical shift Svx/Fz at Fznom
    pVx2: float = 0.0  # Variation of shift Svx/Fz with load

    # === LATERAL FORCE (Fy) ===
    # Shape factor
    pCy1: float = 1.30  # Shape factor Cfy for lateral force

    # Peak factor
    pDy1: float = -0.990  # Lateral friction Muy
    pDy2: float = 0.145  # Variation of friction Muy with load
    pDy3: float = -1.2  # Variation of friction Muy with camber squared

    # Curvature factor
    pEy1: float = -0.65  # Lateral curvature Efy at Fznom
    pEy2: float = -0.24  # Variation of curvature Efy with load
    pEy3: float = 0.14  # Zero order camber dependency of curvature Efy
    pEy4: float = -4.0  # Variation of curvature Efy with camber
    pEy5: float = 0.0  # Camber curvature Efc

    # Stiffness factor
    pKy1: float = -15.324  # Maximum value of stiffness Kfy/Fznom
    pKy2: float = 1.715  # Load at which Kfy reaches maximum value
    pKy3: float = 0.365  # Variation of Kfy/Fznom with camber
    pKy4: float = 2.0  # Curvature of stiffness Kfy
    pKy5: float = 0.0  # Peak stiffness variation with camber squared
    pKy6: float = -1.12  # Fy camber stiffness factor
    pKy7: float = -0.24  # Vertical load dependency of camber stiffness

    # Horizontal shift
    pHy1: float = 0.0026  # Horizontal shift Shy at Fznom
    pHy2: float = 0.0014  # Variation of shift Shy with load

    # Vertical shift
    pVy1: float = 0.045  # Vertical shift in Svy/Fz at Fznom
    pVy2: float = -0.024  # Variation of shift Svy/Fz with load
    pVy3: float = -0.532  # Variation of shift Svy/Fz with camber
    pVy4: float = -0.279  # Variation of shift Svy/Fz with camber and load

    # === COMBINED SLIP (Fx at slip angle) ===
    rBx1: float = 12.35  # Slope factor for combined slip Fx reduction
    rBx2: float = -10.77  # Variation of slope Fx reduction with kappa
    rBx3: float = 0.0  # Influence of camber on stiffness for Fx combined

    rCx1: float = 1.092  # Shape factor for combined slip Fx reduction

    rEx1: float = 0.0  # Curvature factor of combined Fx
    rEx2: float = 0.0  # Curvature factor of combined Fx with load

    rHx1: float = 0.0085  # Shift factor for combined slip Fx reduction

    # === COMBINED SLIP (Fy at longitudinal slip) ===
    rBy1: float = 10.622  # Slope factor for combined Fy reduction
    rBy2: float = 7.82  # Variation of slope Fy reduction with alpha
    rBy3: float = 0.0027  # Shift term for alpha in slope Fy reduction
    rBy4: float = 0.0  # Influence of camber on stiffness of Fy combined

    rCy1: float = 1.081  # Shape factor for combined Fy reduction

    rEy1: float = 0.0  # Curvature factor of combined Fy
    rEy2: float = 0.0  # Curvature factor of combined Fy with load

    rHy1: float = 0.0089  # Shift factor for combined Fy reduction
    rHy2: float = 0.0  # Shift factor for combined Fy reduction with load

    rVy1: float = 0.0  # Kappa induced side force Svyk/Muy*Fz at Fznom
    rVy2: float = 0.0  # Variation of Svyk/Muy*Fz with load
    rVy3: float = 0.0  # Variation of Svyk/Muy*Fz with camber
    rVy4: float = 0.0  # Variation of Svyk/Muy*Fz with alpha
    rVy5: float = 1.9  # Variation of Svyk/Muy*Fz with kappa
    rVy6: float = 0.0  # Variation of Svyk/Muy*Fz with atan(kappa)

    # === ALIGNING TORQUE (Mz) ===
    qBz1: float = 12.0  # Trail slope factor for trail Bpt at Fznom
    qBz2: float = -1.1  # Variation of slope Bpt with load
    qBz3: float = -0.5  # Variation of slope Bpt with load squared
    qBz4: float = 0.0  # Variation of slope Bpt with camber
    qBz5: float = -0.4  # Variation of slope Bpt with absolute camber
    qBz9: float = 18.0  # Slope factor Br of residual torque Mzr
    qBz10: float = 0.0  # Slope factor Br of residual torque Mzr with load

    qCz1: float = 1.2  # Shape factor Cpt for pneumatic trail

    qDz1: float = 0.093  # Peak trail Dpt = Dpt*(Fz/Fznom*R0)
    qDz2: float = -0.009  # Variation of peak Dpt with load
    qDz3: float = -0.057  # Variation of peak Dpt with camber
    qDz4: float = 0.0  # Variation of peak Dpt with camber squared
    qDz6: float = 0.0011  # Peak residual torque Dmr
    qDz7: float = -0.002  # Variation of peak factor Dmr with load
    qDz8: float = -0.27  # Variation of peak factor Dmr with camber
    qDz9: float = 0.0  # Variation of peak factor Dmr with camber and load
    qDz10: float = 0.0  # Variation of peak factor Dmr with camber squared
    qDz11: float = 0.0  # Variation of Dmr with camber squared and load

    qEz1: float = -1.609  # Trail curvature Ept at Fznom
    qEz2: float = 0.062  # Variation of curvature Ept with load
    qEz3: float = 0.0  # Variation of curvature Ept with load squared
    qEz4: float = 0.18  # Variation of curvature Ept with sign of Alpha-t
    qEz5: float = -0.96  # Variation of Ept with camber and sign Alpha-t

    qHz1: float = 0.0014  # Trail horizontal shift Sht at Fznom
    qHz2: float = 0.0025  # Variation of shift Sht with load
    qHz3: float = 0.24  # Variation of shift Sht with camber
    qHz4: float = -0.21  # Variation of shift Sht with camber and load

    # === OVERTURNING MOMENT (Mx) ===
    qSx1: float = 0.0  # Lateral force induced overturning moment
    qSx2: float = 0.0  # Camber induced overturning moment
    qSx3: float = 0.0  # Fy induced overturning moment
    qSx4: float = 4.0  # Mixed load lateral force and camber on Mx
    qSx5: float = 1.0  # Load effect on Mx with lateral force and camber
    qSx6: float = 10.0  # B-factor of load with Mx
    qSx7: float = 0.0  # Camber with load on Mx
    qSx8: float = 0.0  # Lateral force with load on Mx
    qSx9: float = 0.4  # B-factor of lateral force with load on Mx
    qSx10: float = 0.0  # Vertical force with camber on Mx
    qSx11: float = 0.0  # B-factor of vertical force with camber on Mx
    qSx12: float = 0.0  # Camber squared induced overturning moment
    qSx13: float = 0.0  # Lateral force induced overturning moment
    qSx14: float = 0.0  # Lateral force induced overturning moment with camber

    # === ROLLING RESISTANCE ===
    qSy1: float = 0.01  # Rolling resistance torque coefficient
    qSy2: float = 0.0  # Rolling resistance torque depending on Fx
    qSy3: float = 0.0  # Rolling resistance torque depending on speed
    qSy4: float = 0.0  # Rolling resistance torque depending on speed^4
    qSy5: float = 0.0  # Rolling resistance torque depending on camber squared
    qSy6: float = 0.0  # Rolling resistance torque depending on load and camber squared
    qSy7: float = 0.85  # Rolling resistance torque depending on load
    qSy8: float = -0.4  # Rolling resistance torque depending on pressure

    # === TEMPERATURE EFFECTS ===
    # Temperature dependency of peak friction
    lambda_mu_T: float = -0.0015  # Per degree C

    # Reference temperature
    T_ref: float = 90.0  # °C

    # === PRESSURE EFFECTS ===
    # Pressure dependency
    lambda_Fz_p: float = 1.0  # Normalized load variation with pressure
    lambda_mu_p: float = 0.02  # Peak friction variation with pressure (per PSI from nominal)
    p_ref: float = 22.0  # Reference pressure (PSI)

    # === RELAXATION LENGTHS ===
    # For transient tire behavior
    sigma_kappa: float = 0.5  # Longitudinal relaxation length (m)
    sigma_alpha: float = 0.3  # Lateral relaxation length (m)

    # === INFLATION PRESSURE ===
    Fz_nominal: float = 4000.0  # Nominal vertical load (N)

    # === TIRE DIMENSIONS ===
    R0: float = 0.36  # Unloaded tire radius (m)
    width: float = 0.405  # Tire width (m)

    # === LOAD SENSITIVITY ===
    # How forces scale with load (critical for F1)
    Fz_ref: float = 4000.0  # Reference load
    mu_decrease_rate: float = 0.00005  # Friction decrease per N of load

    @classmethod
    def for_compound(cls, compound: TireCompoundAdvanced) -> 'PacejkaCoefficients':
        """Get coefficients for specific F1 compound."""
        # Base coefficients (C3 medium)
        base = cls()

        if compound == TireCompoundAdvanced.C1:  # Hardest
            base.pDx1 = 1.15  # Lower peak friction
            base.pDy1 = -0.95
            base.pKx1 = 19.0  # Lower stiffness
            base.pKy1 = -13.0
            base.lambda_mu_T = -0.001  # Less temp sensitive

        elif compound == TireCompoundAdvanced.C2:
            base.pDx1 = 1.18
            base.pDy1 = -0.97
            base.pKx1 = 20.0
            base.pKy1 = -14.0
            base.lambda_mu_T = -0.0012

        elif compound == TireCompoundAdvanced.C3:  # Medium (default)
            pass

        elif compound == TireCompoundAdvanced.C4:
            base.pDx1 = 1.24
            base.pDy1 = -1.01
            base.pKx1 = 22.5
            base.pKy1 = -16.0
            base.lambda_mu_T = -0.0018

        elif compound == TireCompoundAdvanced.C5:  # Softest
            base.pDx1 = 1.30
            base.pDy1 = -1.05
            base.pKx1 = 24.0
            base.pKy1 = -17.5
            base.lambda_mu_T = -0.002  # Most temp sensitive

        elif compound == TireCompoundAdvanced.INTERMEDIATE:
            base.pDx1 = 1.10  # Lower grip in wet
            base.pDy1 = -0.90
            base.pKx1 = 17.0
            base.pKy1 = -12.0

        elif compound == TireCompoundAdvanced.WET:
            base.pDx1 = 0.95  # Much lower grip
            base.pDy1 = -0.80
            base.pKx1 = 15.0
            base.pKy1 = -10.0

        return base


class AdvancedTireModel:
    """
    F1-Grade Tire Model using Extended Pacejka Magic Formula 6.2.

    Features:
    - 100+ parameters per compound
    - Combined slip (longitudinal + lateral)
    - Temperature effects
    - Pressure effects
    - Camber effects
    - Load sensitivity
    - Transient behavior (relaxation)
    """

    def __init__(
        self,
        compound: TireCompoundAdvanced = TireCompoundAdvanced.C3,
        position: str = "FL"  # FL, FR, RL, RR
    ):
        self.compound = compound
        self.position = position
        self.coeffs = PacejkaCoefficients.for_compound(compound)

        # State variables
        self.temperature = 80.0  # °C
        self.pressure = self.coeffs.p_ref  # PSI
        self.wear = 0.0  # 0-1
        self.distance = 0.0  # km

        # Transient states (for relaxation)
        self.kappa_transient = 0.0  # Longitudinal slip
        self.alpha_transient = 0.0  # Lateral slip (rad)

        # Load history (for thermal model)
        self.load_history = []

    def calculate_forces(
        self,
        Fz: float,  # Vertical load (N)
        kappa: float,  # Longitudinal slip ratio
        alpha: float,  # Slip angle (rad)
        gamma: float,  # Camber angle (rad)
        V: float,  # Forward velocity (m/s)
        dt: float = 0.01  # Timestep
    ) -> Dict[str, float]:
        """
        Calculate tire forces using full Pacejka model.

        Args:
            Fz: Vertical load (N)
            kappa: Longitudinal slip ratio
            alpha: Slip angle (radians)
            gamma: Camber angle (radians)
            V: Forward velocity (m/s)
            dt: Timestep for relaxation

        Returns:
            Dictionary with Fx, Fy, Mz, Mx
        """
        # Update transient states (relaxation)
        self._update_transient_slip(kappa, alpha, V, dt)

        # Use transient slips for force calculation
        kappa_eff = self.kappa_transient
        alpha_eff = self.alpha_transient

        # Normalized load
        dfz = (Fz - self.coeffs.Fz_nominal) / self.coeffs.Fz_nominal

        # Temperature effect on friction
        dT = self.temperature - self.coeffs.T_ref
        lambda_mu = 1.0 + self.coeffs.lambda_mu_T * dT

        # Pressure effect
        dp = self.pressure - self.coeffs.p_ref
        lambda_mu *= (1.0 + self.coeffs.lambda_mu_p * dp)

        # Wear effect (simple degradation)
        wear_factor = 1.0 - 0.3 * self.wear  # Up to 30% loss
        lambda_mu *= wear_factor

        # === PURE LONGITUDINAL FORCE (Fx0) ===
        Fx0 = self._pure_longitudinal_force(kappa_eff, Fz, dfz, gamma, lambda_mu)

        # === PURE LATERAL FORCE (Fy0) ===
        Fy0 = self._pure_lateral_force(alpha_eff, Fz, dfz, gamma, lambda_mu)

        # === COMBINED SLIP EFFECTS ===
        Fx = self._combined_fx(Fx0, Fy0, kappa_eff, alpha_eff, Fz, dfz, gamma)
        Fy = self._combined_fy(Fx0, Fy0, kappa_eff, alpha_eff, Fz, dfz, gamma)

        # === ALIGNING TORQUE (Mz) ===
        Mz = self._aligning_torque(alpha_eff, Fz, dfz, gamma, Fy)

        # === OVERTURNING MOMENT (Mx) ===
        Mx = self._overturning_moment(Fy, Fz, gamma)

        # === ROLLING RESISTANCE ===
        My = self._rolling_resistance(Fz, V)

        return {
            'Fx': Fx,  # Longitudinal force
            'Fy': Fy,  # Lateral force
            'Fz': Fz,  # Vertical force (input)
            'Mx': Mx,  # Overturning moment
            'My': My,  # Rolling resistance moment
            'Mz': Mz,  # Aligning torque (self-aligning moment)
            'kappa': kappa_eff,  # Effective longitudinal slip
            'alpha': alpha_eff,  # Effective slip angle
        }

    def _pure_longitudinal_force(
        self,
        kappa: float,
        Fz: float,
        dfz: float,
        gamma: float,
        lambda_mu: float
    ) -> float:
        """Calculate pure longitudinal force Fx0."""
        c = self.coeffs

        # Friction coefficient
        mu_x = (c.pDx1 + c.pDx2 * dfz) * (1 + c.pDx3 * gamma**2) * lambda_mu

        # Peak longitudinal force
        Dx = mu_x * Fz

        # Longitudinal stiffness
        Kxk = Fz * (c.pKx1 + c.pKx2 * dfz) * np.exp(c.pKx3 * dfz)

        # Shape factor
        Cx = c.pCx1

        # Curvature factor
        Ex = (c.pEx1 + c.pEx2 * dfz + c.pEx3 * dfz**2) * (1 - c.pEx4 * np.sign(kappa))

        # Horizontal shift
        SHx = c.pHx1 + c.pHx2 * dfz

        # Composite slip
        kappa_x = kappa + SHx

        # Slip stiffness factor
        Bx = Kxk / (Cx * Dx + 1e-6)

        # Magic Formula
        Fx0 = Dx * np.sin(Cx * np.arctan(Bx * kappa_x - Ex * (Bx * kappa_x - np.arctan(Bx * kappa_x))))

        # Vertical shift
        SVx = Fz * (c.pVx1 + c.pVx2 * dfz)

        Fx0 = Fx0 + SVx

        return Fx0

    def _pure_lateral_force(
        self,
        alpha: float,
        Fz: float,
        dfz: float,
        gamma: float,
        lambda_mu: float
    ) -> float:
        """Calculate pure lateral force Fy0."""
        c = self.coeffs

        # Friction coefficient
        mu_y = (c.pDy1 + c.pDy2 * dfz) * (1 + c.pDy3 * gamma**2) * lambda_mu

        # Peak lateral force
        Dy = mu_y * Fz

        # Cornering stiffness
        Kya = c.pKy1 * c.Fz_nominal * np.sin(c.pKy4 * np.arctan(Fz / (c.pKy2 * c.Fz_nominal)))
        Kya = Kya * (1 + c.pKy5 * gamma**2) * (1 + c.pKy3 * gamma)

        # Shape factor
        Cy = c.pCy1

        # Curvature factor
        Ey = (c.pEy1 + c.pEy2 * dfz) * (1 + c.pEy5 * gamma**2 - (c.pEy3 + c.pEy4 * gamma) * np.sign(alpha))

        # Horizontal shift
        SHy = (c.pHy1 + c.pHy2 * dfz) + c.pKy6 * gamma

        # Composite slip angle
        alpha_y = alpha + SHy

        # Slip stiffness factor
        By = Kya / (Cy * Dy + 1e-6)

        # Magic Formula
        Fy0 = Dy * np.sin(Cy * np.arctan(By * alpha_y - Ey * (By * alpha_y - np.arctan(By * alpha_y))))

        # Vertical shift
        SVy = Fz * ((c.pVy1 + c.pVy2 * dfz) + (c.pVy3 + c.pVy4 * dfz) * gamma)

        Fy0 = Fy0 + SVy

        return Fy0

    def _combined_fx(
        self,
        Fx0: float,
        Fy0: float,
        kappa: float,
        alpha: float,
        Fz: float,
        dfz: float,
        gamma: float
    ) -> float:
        """Calculate combined longitudinal force (Fx with lateral slip)."""
        c = self.coeffs

        # Friction ellipse approach
        # Fx = Fx0 * Gxa

        # Weighting function
        Bxa = (c.rBx1 + c.rBx3 * gamma**2) * np.cos(np.arctan(c.rBx2 * kappa))
        Cxa = c.rCx1
        Exa = c.rEx1 + c.rEx2 * dfz
        SHxa = c.rHx1

        alpha_s = alpha + SHxa

        # Combined slip reduction factor
        Gxa0 = np.cos(Cxa * np.arctan(Bxa * alpha_s - Exa * (Bxa * alpha_s - np.arctan(Bxa * alpha_s))))

        Fx = Gxa0 * Fx0

        return Fx

    def _combined_fy(
        self,
        Fx0: float,
        Fy0: float,
        kappa: float,
        alpha: float,
        Fz: float,
        dfz: float,
        gamma: float
    ) -> float:
        """Calculate combined lateral force (Fy with longitudinal slip)."""
        c = self.coeffs

        # Weighting function
        Byk = (c.rBy1 + c.rBy4 * gamma**2) * np.cos(np.arctan(c.rBy2 * (alpha - c.rBy3)))
        Cyk = c.rCy1
        Eyk = c.rEy1 + c.rEy2 * dfz
        SHyk = c.rHy1 + c.rHy2 * dfz
        DVyk = Fz * (c.rVy1 + c.rVy2 * dfz + c.rVy3 * gamma) * np.cos(np.arctan(c.rVy4 * alpha))
        SVyk = DVyk * np.sin(c.rVy5 * np.arctan(c.rVy6 * kappa))

        kappa_s = kappa + SHyk

        # Combined slip reduction factor
        Gyk0 = np.cos(Cyk * np.arctan(Byk * kappa_s - Eyk * (Byk * kappa_s - np.arctan(Byk * kappa_s))))

        Fy = Gyk0 * Fy0 + SVyk

        return Fy

    def _aligning_torque(
        self,
        alpha: float,
        Fz: float,
        dfz: float,
        gamma: float,
        Fy: float
    ) -> float:
        """Calculate aligning torque (self-aligning moment) Mz."""
        c = self.coeffs

        # Pneumatic trail
        SHt = c.qHz1 + c.qHz2 * dfz + (c.qHz3 + c.qHz4 * dfz) * gamma
        alpha_t = alpha + SHt

        Bt = (c.qBz1 + c.qBz2 * dfz + c.qBz3 * dfz**2) * (1 + c.qBz4 * gamma + c.qBz5 * abs(gamma))
        Ct = c.qCz1
        Dt = Fz * (c.qDz1 + c.qDz2 * dfz) * (1 + c.qDz3 * gamma + c.qDz4 * gamma**2) * c.R0
        Et = (c.qEz1 + c.qEz2 * dfz + c.qEz3 * dfz**2) * (1 + (c.qEz4 + c.qEz5 * gamma) * np.sign(alpha_t))

        # Pneumatic trail
        t = Dt * np.cos(Ct * np.arctan(Bt * alpha_t - Et * (Bt * alpha_t - np.arctan(Bt * alpha_t))))

        # Residual torque
        Br = c.qBz9 + c.qBz10 * dfz
        Dr = Fz * (c.qDz6 + c.qDz7 * dfz) * (1 + c.qDz8 * gamma + c.qDz9 * gamma * dfz) * c.R0

        Mzr = Dr * np.cos(np.arctan(Br * alpha_t))

        # Total aligning torque
        Mz = -t * Fy + Mzr

        return Mz

    def _overturning_moment(self, Fy: float, Fz: float, gamma: float) -> float:
        """Calculate overturning moment Mx."""
        c = self.coeffs

        # Simplified model
        Mx = Fy * c.R0 * gamma * 0.5  # Rough approximation

        return Mx

    def _rolling_resistance(self, Fz: float, V: float) -> float:
        """Calculate rolling resistance moment My."""
        c = self.coeffs

        My = c.R0 * Fz * (c.qSy1 + c.qSy3 * abs(V) + c.qSy4 * V**4) * (c.qSy7 + c.qSy8 * (self.pressure - c.p_ref))

        return My

    def _update_transient_slip(self, kappa: float, alpha: float, V: float, dt: float):
        """Update transient slip states using relaxation lengths."""
        # Relaxation for longitudinal slip
        if V > 0.1:
            self.kappa_transient += (kappa - self.kappa_transient) * (V * dt / self.coeffs.sigma_kappa)
        else:
            self.kappa_transient = kappa

        # Relaxation for lateral slip
        if V > 0.1:
            self.alpha_transient += (alpha - self.alpha_transient) * (V * dt / self.coeffs.sigma_alpha)
        else:
            self.alpha_transient = alpha

    def update_temperature(self, energy_dissipated: float, ambient_temp: float, dt: float):
        """Update tire temperature based on energy dissipation."""
        # Simplified thermal model
        # Heat capacity of tire (rough estimate)
        C_tire = 1000.0  # J/K

        # Heat generation
        Q_gen = energy_dissipated

        # Cooling (convection + radiation)
        h = 50.0  # W/m²K (combined heat transfer coefficient)
        A = 0.5  # m² (tire surface area)
        Q_cool = h * A * (self.temperature - ambient_temp)

        # Temperature change
        dT = (Q_gen - Q_cool) * dt / C_tire

        self.temperature += dT
        self.temperature = np.clip(self.temperature, ambient_temp, 150.0)

    def update_wear(self, sliding_work: float, dt: float):
        """Update tire wear based on sliding work."""
        # Wear rate depends on energy dissipated and compound
        wear_coefficient = {
            TireCompoundAdvanced.C1: 0.00001,
            TireCompoundAdvanced.C2: 0.00002,
            TireCompoundAdvanced.C3: 0.00003,
            TireCompoundAdvanced.C4: 0.00005,
            TireCompoundAdvanced.C5: 0.00008,
        }.get(self.compound, 0.00003)

        # Accelerated wear at high temperatures
        if self.temperature > 120.0:
            wear_coefficient *= 2.0

        dwear = wear_coefficient * sliding_work * dt
        self.wear = min(1.0, self.wear + dwear)

    def get_state(self) -> Dict:
        """Get current tire state."""
        return {
            'compound': self.compound.value,
            'position': self.position,
            'temperature': self.temperature,
            'pressure': self.pressure,
            'wear': self.wear,
            'distance_km': self.distance,
            'kappa_transient': self.kappa_transient,
            'alpha_transient': self.alpha_transient,
        }

    def export_coefficients(self, filepath: str):
        """Export Pacejka coefficients to JSON."""
        import dataclasses
        with open(filepath, 'w') as f:
            json.dump(dataclasses.asdict(self.coeffs), f, indent=2)


# === VEHICLE-LEVEL TIRE SYSTEM ===

class TireSet:
    """Manages all 4 tires of the vehicle."""

    def __init__(self, compound: TireCompoundAdvanced = TireCompoundAdvanced.C3):
        self.FL = AdvancedTireModel(compound, "FL")
        self.FR = AdvancedTireModel(compound, "FR")
        self.RL = AdvancedTireModel(compound, "RL")
        self.RR = AdvancedTireModel(compound, "RR")

        self.tires = [self.FL, self.FR, self.RL, self.RR]

    def calculate_all_forces(
        self,
        loads: np.ndarray,  # [FL, FR, RL, RR] vertical loads
        slips: np.ndarray,  # [FL, FR, RL, RR] longitudinal slips
        alphas: np.ndarray,  # [FL, FR, RL, RR] slip angles
        gammas: np.ndarray,  # [FL, FR, RL, RR] camber angles
        V: float,
        dt: float = 0.01
    ) -> Dict[str, np.ndarray]:
        """Calculate forces for all 4 tires."""
        Fx_all = np.zeros(4)
        Fy_all = np.zeros(4)
        Mz_all = np.zeros(4)

        for i, tire in enumerate(self.tires):
            forces = tire.calculate_forces(
                Fz=loads[i],
                kappa=slips[i],
                alpha=alphas[i],
                gamma=gammas[i],
                V=V,
                dt=dt
            )

            Fx_all[i] = forces['Fx']
            Fy_all[i] = forces['Fy']
            Mz_all[i] = forces['Mz']

        return {
            'Fx': Fx_all,
            'Fy': Fy_all,
            'Mz': Mz_all,
        }

    def get_total_forces(self, individual_forces: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Sum forces from all tires."""
        return {
            'Fx_total': np.sum(individual_forces['Fx']),
            'Fy_total': np.sum(individual_forces['Fy']),
            'Mz_total': np.sum(individual_forces['Mz']),
        }
