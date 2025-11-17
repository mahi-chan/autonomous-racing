"""Racing circuit models."""

from src.tracks.circuit import Circuit, TrackSegment, TrackType
from src.tracks.silverstone import Silverstone
from src.tracks.monaco import Monaco
from src.tracks.spa import Spa

__all__ = ["Circuit", "TrackSegment", "TrackType", "Silverstone", "Monaco", "Spa"]


# Circuit registry
CIRCUITS = {
    'silverstone': Silverstone,
    'monaco': Monaco,
    'spa': Spa,
}


def get_circuit(name: str) -> Circuit:
    """
    Get circuit by name.

    Args:
        name: Circuit name (lowercase)

    Returns:
        Circuit instance
    """
    if name.lower() not in CIRCUITS:
        raise ValueError(f"Unknown circuit: {name}. Available: {list(CIRCUITS.keys())}")

    return CIRCUITS[name.lower()]()
