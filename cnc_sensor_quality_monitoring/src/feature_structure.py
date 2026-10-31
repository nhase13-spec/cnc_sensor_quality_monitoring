"""Physical and statistical grouping of the sensor feature names (`<channel>__<statistic>`).

Three orthogonal groupings, defined from the signal names ONLY (never from labels or results):

* subsystem        - S1 spindle, X1/Y1/Z1 axes, M1 machine/program channel
* signal_type      - what the channel measures (current, power, voltage, position, velocity, acceleration, ...)
* statistic_family - what kind of summary the feature is (location, spread, shape, dynamics, spectral)

A feature belongs to exactly one group within each grouping (checked by `validate_partition`).
"""

from __future__ import annotations

SIGNAL_TYPE_BY_SUFFIX = (
    ("CurrentFeedback", "current"),
    ("OutputCurrent", "current"),
    ("OutputPower", "power"),
    ("OutputVoltage", "voltage"),
    ("DCBusVoltage", "voltage"),
    ("ActualPosition", "position"),
    ("CommandPosition", "position"),
    ("ActualVelocity", "velocity"),
    ("CommandVelocity", "velocity"),
    ("ActualAcceleration", "acceleration"),
    ("CommandAcceleration", "acceleration"),
    ("SystemInertia", "inertia"),
    ("FEEDRATE", "feedrate_signal"),
)

STAT_FAMILY = {
    "mean": "location", "median": "location", "min": "location", "max": "location",
    "std": "spread", "iqr": "spread", "ptp": "spread", "rms": "spread",
    "skew": "shape", "kurtosis": "shape", "crest": "shape",
    "diff_mean": "dynamics", "diff_std": "dynamics", "trend_slope": "dynamics", "autocorr_lag1": "dynamics",
    "spectral_centroid_hz": "spectral", "spectral_entropy": "spectral", "dominant_frequency_hz": "spectral",
}


def parse_feature(name: str) -> tuple[str, str]:
    channel, _, stat = name.partition("__")
    if not stat:
        raise ValueError(f"Feature '{name}' is not of the form <channel>__<statistic>.")
    return channel, stat


def subsystem_of(channel: str) -> str:
    return channel.split("_", 1)[0]


def signal_type_of(channel: str) -> str:
    for suffix, label in SIGNAL_TYPE_BY_SUFFIX:
        if channel.endswith(suffix):
            return label
    raise ValueError(f"Channel '{channel}' has no signal-type mapping; extend SIGNAL_TYPE_BY_SUFFIX explicitly.")


def groupings(cols: list[str]) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {"subsystem": {}, "signal_type": {}, "statistic_family": {}}
    for c in cols:
        channel, stat = parse_feature(c)
        if stat not in STAT_FAMILY:
            raise ValueError(f"Statistic '{stat}' of '{c}' has no family; extend STAT_FAMILY explicitly.")
        out["subsystem"].setdefault(subsystem_of(channel), []).append(c)
        out["signal_type"].setdefault(signal_type_of(channel), []).append(c)
        out["statistic_family"].setdefault(STAT_FAMILY[stat], []).append(c)
    for scheme, groups in out.items():
        validate_partition(cols, groups, scheme)
    return out


def validate_partition(cols: list[str], groups: dict[str, list[str]], label: str = "") -> None:
    flat = [c for g in groups.values() for c in g]
    if len(flat) != len(set(flat)) or set(flat) != set(cols):
        raise ValueError(f"Grouping '{label}' is not a partition of the feature set.")
