from pathlib import Path

from eesizer_core.agents.reporting import OptimizationReporter
from eesizer_core.context import ArtifactKind, ContextManager


def test_variant_comparison_reporting(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    reporter = OptimizationReporter(artifacts)
    variants = [
        ("baseline", {"gain_db": 30.0, "power_mw": 5.0}),
        ("improved", {"gain_db": 50.0, "power_mw": 3.0}),
    ]
    with ContextManager("run", tmp_path, "test") as ctx:
        csv_path, json_path = reporter.write_variant_comparison(
            ctx, variants, scoring_fn=lambda m: m["gain_db"] / m["power_mw"]
        )
        assert csv_path.exists()
        assert json_path.exists()
        opt_artifacts = list(ctx.list_artifacts(kind=ArtifactKind.OPTIMIZATION))
        names = [record.name for record in opt_artifacts]
        assert "variant_comparison_csv" in names
        assert "variant_comparison_json" in names
