import os
import pytest

from agent_test_gpt import reporting


def test_plot_optimization_history_missing_file():
    with pytest.raises(FileNotFoundError):
        reporting.plot_optimization_history("nonexistent.csv", "out.pdf")
