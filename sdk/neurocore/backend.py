"""Abstract backend interface for chip or simulator execution."""

from abc import ABC, abstractmethod


class Backend(ABC):
    """Abstract interface that Chip and Simulator both implement."""

    @abstractmethod
    def deploy(self, network_or_compiled):
        """Compile (if needed) and load a network onto the target."""

    @abstractmethod
    def inject(self, target, current):
        """Set external stimulus current for specified neurons."""

    @abstractmethod
    def run(self, timesteps):
        """Execute timesteps and return a RunResult."""

    @abstractmethod
    def set_learning(self, learn=False, graded=False, dendritic=False,
                     async_mode=False, three_factor=False, noise=False):
        """Configure hardware feature flags."""

    @abstractmethod
    def reward(self, value):
        """Apply reward signal for 3-factor learning (P13c)."""

    @abstractmethod
    def status(self):
        """Query backend state."""

    @abstractmethod
    def close(self):
        """Release resources."""
