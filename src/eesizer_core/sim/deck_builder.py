from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..contracts.artifacts import SimPlan
from ..contracts.enums import SimKind
from ..contracts.errors import ValidationError
from ..contracts.operators import Operator, OperatorResult
from ..contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_json, stable_hash_str
from ..domain.spice.sanitize_rules import has_control_block
from .artifacts import SpiceDeck


def _format_value(v: Any) -> str:
    if isinstance(v, (int, float)):
        return f"{v:g}"
    return str(v)


def _normalize_output_nodes(nodes: Iterable[str] | None) -> tuple[str, ...]:
    if nodes is None:
        return ("out",)
    out: list[str] = []
    for n in nodes:
        if not isinstance(n, str) or not n.strip():
            raise ValidationError("output_nodes must be non-empty strings")
        out.append(n.strip())
    if not out:
        raise ValidationError("output_nodes cannot be empty")
    return tuple(out)


def _inject_control_block(netlist_text: str, control_lines: Sequence[str]) -> str:
    lines = netlist_text.splitlines()
    lower_lines = [ln.strip().lower() for ln in lines]

    if ".end" in lower_lines:
        end_idx = len(lower_lines) - 1 - lower_lines[::-1].index(".end")
        prefix = lines[:end_idx]
        suffix = lines[end_idx:]
    else:
        prefix = lines
        suffix = [".end"]

    new_lines = list(prefix)
    if new_lines and new_lines[-1].strip():
        new_lines.append("")  # spacer for readability
    new_lines.extend(control_lines)
    if suffix and suffix[0].strip():
        new_lines.append("")  # spacer before .end if needed
    new_lines.extend(suffix)
    return "\n".join(new_lines) + "\n"


class DeckBuildOperator(Operator):
    """Build a SpiceDeck by injecting a .control block for AC analysis."""

    name = "deck_build"
    version = "0.1.0"

    def __init__(
        self,
        default_points_per_decade: int = 10,
        default_start_hz: float = 1.0,
        default_stop_hz: float = 1e6,
    ) -> None:
        self.default_points_per_decade = default_points_per_decade
        self.default_start_hz = default_start_hz
        self.default_stop_hz = default_stop_hz

    def _build_ac_control(self, params: Mapping[str, Any]) -> tuple[list[str], tuple[str, ...]]:
        p_per_dec = params.get("points_per_decade", self.default_points_per_decade)
        start = params.get("start_hz", self.default_start_hz)
        stop = params.get("stop_hz", self.default_stop_hz)
        output_nodes = _normalize_output_nodes(params.get("output_nodes"))

        try:
            p_per_dec_int = int(p_per_dec)
        except (TypeError, ValueError) as exc:
            raise ValidationError("points_per_decade must be an integer") from exc
        if p_per_dec_int <= 0:
            raise ValidationError("points_per_decade must be positive")

        control_lines = [
            ".control",
            "set filetype=ascii",
            f"ac dec {p_per_dec_int} {_format_value(start)} {_format_value(stop)}",
            f"wrdata ac.csv frequency {' '.join(self._ac_columns(output_nodes))}",
            "quit",
            ".endc",
        ]
        return control_lines, output_nodes

    @staticmethod
    def _ac_columns(nodes: tuple[str, ...]) -> list[str]:
        cols: list[str] = []
        for node in nodes:
            cols.append(f"vdb({node})")
            cols.append(f"vp({node})")
        return cols

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        netlist_text = inputs.get("netlist_text")
        if not isinstance(netlist_text, str):
            raise ValidationError("netlist_text must be provided as a string")
        if has_control_block(netlist_text):
            raise ValidationError("netlist_text must not contain .control/.endc blocks")

        sim_plan = inputs.get("sim_plan")
        if not isinstance(sim_plan, SimPlan):
            raise ValidationError("sim_plan must be provided as a SimPlan")

        ac_req = next((s for s in sim_plan.sims if s.kind == SimKind.ac), None)
        if ac_req is None:
            raise ValidationError("sim_plan must include an AC SimRequest")

        control_lines, output_nodes = self._build_ac_control(ac_req.params)
        deck_text = _inject_control_block(netlist_text, control_lines)

        deck = SpiceDeck(
            text=deck_text,
            kind=SimKind.ac,
            expected_outputs={"ac_csv": "ac.csv"},
        )

        provenance = Provenance(operator=self.name, version=self.version)
        provenance.inputs["netlist_text"] = ArtifactFingerprint(sha256=stable_hash_str(netlist_text))
        provenance.inputs["sim_plan"] = ArtifactFingerprint(
            sha256=stable_hash_json(
                {"sims": [{"kind": s.kind.value, "params": s.params} for s in sim_plan.sims]}
            )
        )
        provenance.outputs["deck"] = ArtifactFingerprint(
            sha256=stable_hash_json({"kind": deck.kind.value, "expected_outputs": dict(deck.expected_outputs)})
        )
        provenance.outputs["deck_text"] = ArtifactFingerprint(sha256=stable_hash_str(deck.text))
        provenance.outputs["output_nodes"] = ArtifactFingerprint(sha256=stable_hash_json(output_nodes))
        provenance.finish()

        return OperatorResult(outputs={"deck": deck, "output_nodes": output_nodes}, provenance=provenance)
