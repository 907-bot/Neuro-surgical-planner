"""
src/simulation/snn_physiology.py
Spiking Neural Network layer for real-time intraoperative physiology simulation.
Encodes physiological signals as spike trains; predicts real-time patient state.

Uses SpikingJelly (preferred) or a pure NumPy LIF simulation as fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import neuron, encoding, layer, functional
    SJ_AVAILABLE = True
except ImportError:
    SJ_AVAILABLE = False
    logger.warning("SpikingJelly not available — using NumPy LIF simulation")


# ─── Physiological Signal Definitions ────────────────────────────────────────
PHYSIO_CHANNELS = [
    "blood_pressure_systolic",   # mmHg (80–180)
    "blood_pressure_diastolic",  # mmHg (50–120)
    "heart_rate",                # bpm  (40–150)
    "spo2",                      # %    (70–100)
    "intracranial_pressure",     # mmHg (0–40)
    "cerebral_blood_flow",       # ml/100g/min (20–80)
    "eto2",                      # end-tidal O2 (%)
    "etco2",                     # end-tidal CO2 mmHg
]

N_CHANNELS = len(PHYSIO_CHANNELS)

# Normal ranges (min, max) for normalization
NORMAL_RANGES = {
    "blood_pressure_systolic":  (80.0,  180.0),
    "blood_pressure_diastolic": (50.0,  120.0),
    "heart_rate":               (40.0,  150.0),
    "spo2":                     (70.0,  100.0),
    "intracranial_pressure":    (0.0,   40.0),
    "cerebral_blood_flow":      (20.0,  80.0),
    "eto2":                     (15.0,  50.0),
    "etco2":                    (20.0,  60.0),
}


# ─── Data Structures ──────────────────────────────────────────────────────────
@dataclass
class IntraoperativeState:
    """Snapshot of real-time patient physiology during surgery."""
    timestamp_ms:       float
    vitals:             Dict[str, float]
    alert_level:        str           # NORMAL | WARNING | CRITICAL
    predicted_outcome:  float         # 0–1 recovery score
    spike_rates:        Dict[str, float] = field(default_factory=dict)
    alerts:             List[str]     = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "timestamp_ms":      self.timestamp_ms,
            "vitals":            {k: round(v, 2) for k, v in self.vitals.items()},
            "alert_level":       self.alert_level,
            "predicted_outcome": round(self.predicted_outcome, 4),
            "alerts":            self.alerts,
        }


# ─── NumPy LIF Neuron (fallback) ──────────────────────────────────────────────
class LIFNeuron:
    """
    Leaky Integrate-and-Fire neuron — pure NumPy.
    v[t] = beta * v[t-1] + I[t]
    spike when v >= threshold; reset to 0
    """

    def __init__(
        self,
        n_neurons:  int,
        beta:       float = 0.9,    # membrane time constant
        threshold:  float = 1.0,
        reset:      float = 0.0,
    ):
        self.beta      = beta
        self.threshold = threshold
        self.reset_val = reset
        self.v         = np.zeros(n_neurons, dtype=np.float32)

    def forward(self, current: np.ndarray) -> np.ndarray:
        self.v = self.beta * self.v + current
        spikes = (self.v >= self.threshold).astype(np.float32)
        self.v[spikes > 0] = self.reset_val
        return spikes

    def reset(self):
        self.v[:] = 0.0


class RateEncoder:
    """Encode normalized signal (0–1) as Poisson spike rate."""

    def __init__(self, max_rate: float = 100.0, dt: float = 0.001):
        self.max_rate = max_rate
        self.dt = dt

    def encode(self, x: np.ndarray) -> np.ndarray:
        """x: (N,) normalized 0–1 → spike probability per timestep."""
        probs = np.clip(x, 0, 1) * self.max_rate * self.dt
        return (np.random.rand(*x.shape) < probs).astype(np.float32)


# ─── SNN Physiology Model ─────────────────────────────────────────────────────
class SNNPhysiologyModel:
    """
    Spatio-temporal SNN that monitors intraoperative physiology.

    Architecture:
        Input layer  — rate-encoded physiology channels
        LIF layer 1  — temporal integration (T timesteps)
        LIF layer 2  — pattern detection
        Readout      — regression head → outcome prediction

    Can use SpikingJelly (GPU-accelerated) or pure NumPy fallback.
    """

    T_STEPS = 20          # timesteps per inference window
    HIDDEN  = 64          # LIF neurons per hidden layer

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.encoder = RateEncoder(max_rate=100.0, dt=0.001)

        if SJ_AVAILABLE:
            self.model = self._build_sj_model()
            logger.info("SNN: SpikingJelly model initialized")
        else:
            self._lif1 = LIFNeuron(self.HIDDEN, beta=0.9)
            self._lif2 = LIFNeuron(self.HIDDEN // 2, beta=0.8)
            self._w1   = np.random.randn(N_CHANNELS, self.HIDDEN).astype(np.float32) * 0.1
            self._w2   = np.random.randn(self.HIDDEN, self.HIDDEN // 2).astype(np.float32) * 0.1
            self._wout = np.random.randn(self.HIDDEN // 2, 1).astype(np.float32) * 0.1
            logger.info("SNN: NumPy LIF fallback initialized")

    def _build_sj_model(self):
        """Build SpikingJelly sequential SNN."""
        import torch.nn as nn
        from spikingjelly.activation_based import neuron, layer, functional

        model = nn.Sequential(
            layer.Linear(N_CHANNELS, self.HIDDEN),
            neuron.LIFNode(tau=2.0, detach_reset=True),
            layer.Linear(self.HIDDEN, self.HIDDEN // 2),
            neuron.LIFNode(tau=2.0, detach_reset=True),
            layer.Linear(self.HIDDEN // 2, 1),
        )
        functional.set_step_mode(model, step_mode="m")
        return model

    def normalize_vitals(self, vitals: Dict[str, float]) -> np.ndarray:
        """Normalize raw vitals to [0, 1] for SNN encoding."""
        vec = np.zeros(N_CHANNELS, dtype=np.float32)
        for i, ch in enumerate(PHYSIO_CHANNELS):
            val = vitals.get(ch, 0.0)
            lo, hi = NORMAL_RANGES[ch]
            vec[i] = float(np.clip((val - lo) / (hi - lo), 0, 1))
        return vec

    def predict(self, vitals_sequence: List[Dict[str, float]]) -> Tuple[float, Dict[str, float]]:
        """
        Run SNN inference on a sequence of vital sign snapshots.

        Args:
            vitals_sequence: list of T dicts with physiology values

        Returns:
            (predicted_outcome, spike_rates_per_channel)
        """
        T = min(len(vitals_sequence), self.T_STEPS)
        x_seq = np.stack([
            self.normalize_vitals(v) for v in vitals_sequence[:T]
        ])  # (T, N_CHANNELS)

        if SJ_AVAILABLE:
            return self._predict_sj(x_seq)
        else:
            return self._predict_numpy(x_seq)

    def _predict_numpy(self, x_seq: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """NumPy LIF simulation."""
        self._lif1.reset()
        self._lif2.reset()
        output_acc = []
        spike_acc  = np.zeros(N_CHANNELS)

        for t in range(len(x_seq)):
            spikes_in = self.encoder.encode(x_seq[t])
            spike_acc += spikes_in

            h1 = self._lif1.forward(spikes_in @ self._w1)
            h2 = self._lif2.forward(h1 @ self._w2)
            out = float(np.clip((h2 @ self._wout).flatten()[0], 0, 1))
            output_acc.append(out)

        mean_out = float(np.mean(output_acc))
        spike_rates = {
            ch: float(spike_acc[i] / len(x_seq))
            for i, ch in enumerate(PHYSIO_CHANNELS)
        }
        return mean_out, spike_rates

    def _predict_sj(self, x_seq: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """SpikingJelly simulation."""
        import torch
        from spikingjelly.activation_based import functional

        functional.reset_net(self.model)
        x_t = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(1)  # (T, 1, N)

        with torch.no_grad():
            out = self.model(x_t)  # (T, 1, 1)

        mean_out = float(out.mean().item())
        mean_out = float(np.clip(mean_out, 0, 1))

        spike_rates = {ch: float(x_seq[:, i].mean()) for i, ch in enumerate(PHYSIO_CHANNELS)}
        return mean_out, spike_rates


# ─── Intraoperative Monitor ───────────────────────────────────────────────────
class IntraoperativeMonitor:
    """
    Real-time physiology monitor using the SNN model.
    Raises alerts when physiological parameters deviate from safe ranges.
    """

    ALERT_THRESHOLDS = {
        "blood_pressure_systolic":  (90.0,  160.0),   # (low_alarm, high_alarm)
        "heart_rate":               (50.0,  120.0),
        "spo2":                     (92.0,  100.0),    # only low alarm
        "intracranial_pressure":    (0.0,   20.0),     # only high alarm
        "cerebral_blood_flow":      (30.0,  100.0),
    }

    def __init__(self):
        self.snn = SNNPhysiologyModel()
        self.history: List[Dict[str, float]] = []
        self.window_size = SNNPhysiologyModel.T_STEPS

    def update(self, vitals: Dict[str, float], timestamp_ms: float = 0.0) -> IntraoperativeState:
        """
        Process a new vital signs reading.

        Args:
            vitals: dict of {channel: value}
            timestamp_ms: recording timestamp

        Returns:
            IntraoperativeState with alerts and SNN-predicted outcome
        """
        self.history.append(vitals)
        window = self.history[-self.window_size:]

        predicted_outcome, spike_rates = self.snn.predict(window)

        alerts = self._check_alerts(vitals)
        alert_level = (
            "CRITICAL" if len(alerts) >= 2 else
            "WARNING"  if len(alerts) >= 1 else "NORMAL"
        )

        return IntraoperativeState(
            timestamp_ms=timestamp_ms,
            vitals=vitals,
            alert_level=alert_level,
            predicted_outcome=predicted_outcome,
            spike_rates=spike_rates,
            alerts=alerts,
        )

    def _check_alerts(self, vitals: Dict[str, float]) -> List[str]:
        alerts = []
        for channel, (lo, hi) in self.ALERT_THRESHOLDS.items():
            val = vitals.get(channel)
            if val is None:
                continue
            if val < lo:
                alerts.append(f"LOW {channel}: {val:.1f} < {lo}")
            elif val > hi:
                alerts.append(f"HIGH {channel}: {val:.1f} > {hi}")
        return alerts

    def simulate_surgery(
        self,
        duration_ms: int = 10_000,
        dt_ms: int = 100,
        tumor_removal_at_ms: Optional[int] = 5_000,
    ) -> List[IntraoperativeState]:
        """
        Simulate physiological response during surgery.
        Useful for testing / demo without real sensor data.
        """
        states = []
        rng = np.random.RandomState(42)

        for t in range(0, duration_ms, dt_ms):
            progress = t / duration_ms

            # Simulate tumor removal effect
            if tumor_removal_at_ms and t >= tumor_removal_at_ms:
                cbf_boost = 10.0 * (t - tumor_removal_at_ms) / duration_ms
                icp_drop  = 5.0  * (t - tumor_removal_at_ms) / duration_ms
            else:
                cbf_boost = 0.0
                icp_drop  = 0.0

            vitals = {
                "blood_pressure_systolic":  120 + rng.normal(0, 5),
                "blood_pressure_diastolic":  80 + rng.normal(0, 3),
                "heart_rate":                72 + rng.normal(0, 4),
                "spo2":                      98 + rng.normal(0, 0.5),
                "intracranial_pressure":     15 - icp_drop + rng.normal(0, 1),
                "cerebral_blood_flow":       45 + cbf_boost + rng.normal(0, 3),
                "eto2":                      30 + rng.normal(0, 1),
                "etco2":                     35 + rng.normal(0, 1),
            }

            state = self.update(vitals, timestamp_ms=float(t))
            states.append(state)

        logger.info(f"Simulated {len(states)} timesteps | "
                    f"Final outcome: {states[-1].predicted_outcome:.1%}")
        return states
