import os
import tempfile

# Must be set before any matplotlib import to avoid font cache lock flakiness
# (e.g. shared home directories / parallel pytest workers).
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-"))

