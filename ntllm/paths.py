"""Repo paths. Everything is relative to the repo root so scripts work from anywhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
SPLITS = DATA / "splits"
RUNS = ROOT / "runs"
CHECKPOINTS = ROOT / "checkpoints"
LOGS = ROOT / "logs"
