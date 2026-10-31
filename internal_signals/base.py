"""Provider-neutral signal result; values retain provider-specific semantics."""
from dataclasses import asdict, dataclass
from typing import Optional, Protocol


@dataclass
class InternalSignals:
    premise_uncertainty: Optional[float]
    consequence_uncertainty: Optional[float]
    local_uncertainty: Optional[float]
    global_uncertainty: Optional[float]
    missing_span: bool
    token_count: int
    extraction_latency: float
    backend: str
    dtype: str
    model_revision: str
    status: str
    errors: list
    details: dict

    def to_dict(self):
        return asdict(self)


class InternalSignalProvider(Protocol):
    def extract(self, record: dict) -> InternalSignals:
        """Observe only the generation record, not future environment transitions."""
        ...
