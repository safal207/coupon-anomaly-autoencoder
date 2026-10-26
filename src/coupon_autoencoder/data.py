"""Deterministic synthetic employee-day aggregates; labels are metadata only."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURES = (
    "tickets_printed", "served_customers", "reprints", "cancelled_tickets",
    "median_print_gap_seconds", "short_gap_share", "after_hours_share",
    "repeat_customer_share", "same_terminal_reuse_share", "unique_customers",
    "staff_session_minutes", "manual_override_count",
)
LOG_COLUMNS = (0, 1, 2, 3, 4, 9, 10, 11)
RATIO_COLUMNS = (5, 6, 7, 8)
SCENARIOS = ("bulk_printing", "repeat_use", "after_hours", "subtle_mismatch")


@dataclass
class Dataset:
    x: np.ndarray
    labels: np.ndarray
    day: np.ndarray
    employee: np.ndarray
    scenario: np.ndarray
    split: np.ndarray


def validate_features(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or x.shape[0] == 0:
        raise ValueError(f"Expected nonempty matrix with {len(FEATURES)} features")
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Features must be finite and nonnegative")
    if (x[:, RATIO_COLUMNS] > 1).any():
        raise ValueError("Share features must be between zero and one")
    return x


def log_features(x):
    z = validate_features(x).copy()
    z[:, LOG_COLUMNS] = np.log1p(z[:, LOG_COLUMNS])
    return z


def generate(seed=42, employees=60, days=120):
    if employees < 2 or days < 20:
        raise ValueError("Use at least two employees and twenty days")
    rng = np.random.default_rng(seed)
    n = employees * days
    day = np.repeat(np.arange(days), employees)
    employee = np.tile(np.arange(employees), days)
    employee_load = rng.lognormal(0, 0.25, employees)
    demand = rng.lognormal(3.4, 0.25, n) * employee_load[employee]
    demand *= 1 + 0.12 * np.sin(2 * np.pi * day / 7)
    busy = rng.random(n) < 0.04
    demand[busy] *= 1.8
    tickets = rng.poisson(demand) + 5
    served = rng.binomial(tickets, 0.88)
    unique = rng.binomial(served, 0.96)
    reprints = rng.binomial(tickets, 0.025)
    cancelled = rng.binomial(tickets, 0.06)
    session = np.clip(rng.normal(410, 22, n), 300, 500)
    gap = np.clip(session * 60 / tickets * rng.lognormal(0, 0.1, n), 1, None)
    x = np.column_stack([
        tickets, served, reprints, cancelled, gap,
        rng.beta(1.1, 40, n), rng.beta(1, 55, n),
        1 - unique / np.maximum(served, 1), rng.beta(1.5, 30, n),
        unique, session, rng.poisson(0.008 * tickets),
    ]).astype(float)
    train_end, validation_end = days // 2, days * 3 // 4
    split = np.where(day < train_end, "train", np.where(day < validation_end, "validation", "test"))
    labels = np.zeros(n, dtype=int)
    scenario = np.where(busy, "legitimate_busy", "normal").astype("<U24")
    candidates = np.flatnonzero(split == "test")
    anomalous = rng.choice(candidates, size=max(4, len(candidates) // 20), replace=False)
    labels[anomalous] = 1
    for k, row in enumerate(anomalous):
        case = SCENARIOS[k % len(SCENARIOS)]
        scenario[row] = case
        if case == "bulk_printing":
            x[row, 0] += rng.integers(45, 100)
            x[row, 4] *= 0.12
            x[row, 5] = rng.uniform(0.5, 0.85)
        elif case == "repeat_use":
            x[row, 2] += rng.integers(12, 25)
            x[row, 7] = rng.uniform(0.45, 0.8)
            x[row, 8] = rng.uniform(0.4, 0.7)
            x[row, 9] = np.floor(x[row, 1] * (1 - x[row, 7]))
        elif case == "after_hours":
            x[row, 6] = rng.uniform(0.45, 0.8)
            x[row, 11] += rng.integers(8, 15)
        else:
            x[row, 0] = np.ceil(x[row, 0] * 1.18)
            x[row, 1] = np.floor(x[row, 1] * 0.7)
            x[row, 9] = min(x[row, 9], x[row, 1])
            x[row, 7] = 1 - x[row, 9] / max(x[row, 1], 1)
            x[row, 5] += 0.07
            x[row, 8] += 0.08
    validate_features(x)
    return Dataset(x, labels, day, employee, scenario, split)


def write_dataset(data, path):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["day", "employee_id", "split", "label", "scenario", *FEATURES])
        for i, values in enumerate(data.x):
            writer.writerow([int(data.day[i]), f"E{data.employee[i]:04d}", data.split[i],
                             int(data.labels[i]), data.scenario[i], *values])


def read_features(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not set(FEATURES).issubset(reader.fieldnames):
            raise ValueError("CSV is missing required feature columns")
        rows = list(reader)
    return validate_features([[float(r[k]) for k in FEATURES] for r in rows])
