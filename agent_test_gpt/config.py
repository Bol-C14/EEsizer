"""Shared configuration constants."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
RESOURCE_DIR = PACKAGE_ROOT / "resources"

OUTPUT_DIR = str(PROJECT_ROOT / "output" / "90nm")
RESULT_HISTORY_FILE = str(PROJECT_ROOT / "output" / "90nm" / "result_history.txt")
CSV_FILE = str(PROJECT_ROOT / "output" / "90nm" / "g2_o3.csv")
NETLIST_OUTPUT_PATH = str(PROJECT_ROOT / "output" / "90nm" / "netlist_cs_o3" / "a1.cir")
PLOT_PDF_PATH = str(PROJECT_ROOT / "output" / "railtorail_subplots_4x2_g1.pdf")
RUN_OUTPUT_ROOT = str(PROJECT_ROOT / "output" / "runs")

OP_TXT_PATH = str(PROJECT_ROOT / "output" / "op.txt")
VGS_FILTERED_PATH = str(PROJECT_ROOT / "output" / "vgscheck.txt")
VGS_CSV_PATH = str(PROJECT_ROOT / "output" / "vgscheck.csv")
VGS_OUTPUT_PATH = str(PROJECT_ROOT / "output" / "vgscheck_output.txt")

MAX_ITERATIONS = 25
TOLERANCE = 0.05
