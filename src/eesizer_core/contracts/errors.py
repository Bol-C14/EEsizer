from __future__ import annotations


class EEsizerError(Exception):
    """Base error for eesizer_core."""


class ContractError(EEsizerError):
    """Raised when a contract (artifact/operator/policy) is violated."""


class ValidationError(ContractError):
    """Raised when an input artifact fails validation."""


class OperatorError(EEsizerError):
    """Raised when an operator fails to execute correctly."""


class PolicyError(EEsizerError):
    """Raised when a policy cannot produce a valid decision."""


class SimulationError(OperatorError):
    """Raised when a simulator run fails."""


class MetricError(OperatorError):
    """Raised when metric computation fails due to missing/invalid data."""
