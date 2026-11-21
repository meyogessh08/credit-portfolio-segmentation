"""Paths and colours shared by the notebooks, the dashboard and the tests."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
APP_DATA_DIR = ROOT / "app_data"
FIGURE_DIR = ROOT / "docs" / "figures"

SEGMENT_COLOURS = [
    "#2F6F8F",
    "#E4572E",
    "#00895C",
    "#F2A541",
    "#7B5EA7",
    "#8C8C8C",
    "#C44569",
    "#3FA7D6",
]
ACCENT = "#00895C"
WARNING = "#E4572E"
