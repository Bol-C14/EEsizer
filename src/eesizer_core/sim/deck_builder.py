from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..contracts.artifacts import SimPlan, SimRequest
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
    """Build a SpiceDeck by injecting a .control block for ac/dc/tran analyses."""

    name = "deck_build"
    version = "0.1.0"

    def __init__(
        self,
        default_points_per_decade: int = 10,
        default_start_hz: float = 1.0,
        default_stop_hz: float = 1e6,
        default_dc_source: str = "V1",
        default_dc_start: float = 0.0,
        default_dc_stop: float = 1.0,
        default_dc_step: float = 0.1,
        default_dc_sweep_node: str = "in",
        default_tran_step: float = 1e-6,
        default_tran_stop: float = 1e-3,
    ) -> None:
        self.default_points_per_decade = default_points_per_decade
        self.default_start_hz = default_start_hz
        self.default_stop_hz = default_stop_hz
        self.default_dc_source = default_dc_source
        self.default_dc_start = default_dc_start
        self.default_dc_stop = default_dc_stop
        self.default_dc_step = default_dc_step
        self.default_dc_sweep_node = default_dc_sweep_node
        self.default_tran_step = default_tran_step
        self.default_tran_stop = default_tran_stop

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
        return control_lines, output_nodes, {"ac_csv": "ac.csv"}

    @staticmethod
    def _ac_columns(nodes: tuple[str, ...]) -> list[str]:
        cols: list[str] = []
        for node in nodes:
            cols.append(f"vdb({node})")
            cols.append(f"vp({node})")
        return cols

    def _build_dc_control(self, params: Mapping[str, Any]) -> tuple[list[str], tuple[str, ...], dict[str, str]]:
        source = params.get("sweep_source", self.default_dc_source)
        sweep_node = params.get("sweep_node", self.default_dc_sweep_node)
        start = params.get("start", self.default_dc_start)
        stop = params.get("stop", self.default_dc_stop)
        step = params.get("step", self.default_dc_step)
        output_nodes = _normalize_output_nodes(params.get("output_nodes"))

        if not isinstance(source, str) or not source:
            raise ValidationError("sweep_source must be a non-empty string")

        control_lines = [
            ".control",
            "set filetype=ascii",
            f"dc {source} {_format_value(start)} {_format_value(stop)} {_format_value(step)}",
            f"wrdata dc.csv v({sweep_node}) {' '.join(f'v({n})' for n in output_nodes)}",
            "quit",
            ".endc",
        ]
        return control_lines, output_nodes, {"dc_csv": "dc.csv"}

    def _build_tran_control(self, params: Mapping[str, Any]) -> tuple[list[str], tuple[str, ...], dict[str, str]]:
        step = params.get("step", self.default_tran_step)
        stop = params.get("stop", self.default_tran_stop)
        output_nodes = _normalize_output_nodes(params.get("output_nodes"))

        control_lines = [
            ".control",
            "set filetype=ascii",
            f"tran {_format_value(step)} {_format_value(stop)}",
            f"wrdata tran.csv time {' '.join(f'v({n})' for n in output_nodes)}",
            "quit",
            ".endc",
        ]
        return control_lines, output_nodes, {"tran_csv": "tran.csv"}

    @staticmethod
    def _select_sim(sim_plan: SimPlan, sim_kind: SimKind | None) -> tuple[SimRequest, SimKind]:
        if sim_kind is not None:
            for s in sim_plan.sims:
                if s.kind == sim_kind:
                    return s, sim_kind
            raise ValidationError(f"sim_plan does not include requested SimKind '{sim_kind.value}'")

        if not sim_plan.sims:
            raise ValidationError("sim_plan must include at least one SimRequest")
        req = sim_plan.sims[0]
        return req, req.kind

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        netlist_text = inputs.get("netlist_text")
        if not isinstance(netlist_text, str):
            raise ValidationError("netlist_text must be provided as a string")
        if has_control_block(netlist_text):
            raise ValidationError("netlist_text must not contain .control/.endc blocks")

        sim_plan = inputs.get("sim_plan")
        if not isinstance(sim_plan, SimPlan):
            raise ValidationError("sim_plan must be provided as a SimPlan")

        sim_kind = inputs.get("sim_kind")
        if sim_kind is not None and not isinstance(sim_kind, SimKind):
            raise ValidationError("sim_kind must be a SimKind if provided")

        sim_req, resolved_kind = self._select_sim(sim_plan, sim_kind)

        if resolved_kind == SimKind.ac:
            control_lines, output_nodes, expected_outputs = self._build_ac_control(sim_req.params)
        elif resolved_kind == SimKind.dc:
            control_lines, output_nodes, expected_outputs = self._build_dc_control(sim_req.params)
        elif resolved_kind == SimKind.tran:
            control_lines, output_nodes, expected_outputs = self._build_tran_control(sim_req.params)
        else:
            raise ValidationError(f"Unsupported SimKind '{resolved_kind.value}'")

        deck_text = _inject_control_block(netlist_text, control_lines)

        deck = SpiceDeck(
            text=deck_text,
            kind=resolved_kind,
            expected_outputs=expected_outputs,
        )

        provenance = Provenance(operator=self.name, version=self.version)
        provenance.inputs["netlist_text"] = ArtifactFingerprint(sha256=stable_hash_str(netlist_text))
        provenance.inputs["sim_plan"] = ArtifactFingerprint(
            sha256=stable_hash_json(
                {"sims": [{"kind": s.kind.value, "params": s.params} for s in sim_plan.sims]}
            )
        )
        provenance.inputs["sim_kind"] = ArtifactFingerprint(sha256=stable_hash_str(resolved_kind.value))
        provenance.outputs["deck"] = ArtifactFingerprint(
            sha256=stable_hash_json({"kind": deck.kind.value, "expected_outputs": dict(deck.expected_outputs)})
        )
        provenance.outputs["deck_text"] = ArtifactFingerprint(sha256=stable_hash_str(deck.text))
        provenance.outputs["output_nodes"] = ArtifactFingerprint(sha256=stable_hash_json(output_nodes))
        provenance.finish()

        return OperatorResult(outputs={"deck": deck, "output_nodes": output_nodes}, provenance=provenance)
