"""Abstract base class every tool must implement."""

from abc import ABC, abstractmethod


class BaseTool(ABC):
    """
    Every tool (calculator, weather, search, wikipedia, ...) inherits from
    this class so the agent's router can treat them uniformly.
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, **kwargs):
        """Execute the tool and return a JSON-serialisable result."""
        raise NotImplementedError
