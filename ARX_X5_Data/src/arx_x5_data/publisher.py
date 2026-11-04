"""Optional integration point for an already-running teleop process."""

import json
import socket
import time
from pathlib import Path


class TelemetryPublisher:
    """Best-effort transport only: no hardware access, retries or controller lifecycle."""

    def __init__(self, path: Path, hz: float = 50):
        if not 1 <= hz <= 200:
            raise ValueError("Telemetry rate must be in 1..200 Hz")
        self.path, self.period = str(path), 1 / hz
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._next = 0.0

    def publish(self, arms: dict) -> bool:
        """Call after SDK submissions; each arm uses the documented legacy fields."""
        now = time.monotonic()
        if now < self._next:
            return False
        self._next = now + self.period
        try:
            packet = {"at_monotonic_s": now, "at_s": time.time(), "arms": arms}
            self._socket.sendto(
                json.dumps(packet, allow_nan=False, separators=(",", ":")).encode(), self.path
            )
            return True
        except (OSError, ValueError, TypeError):
            return False

    def close(self) -> None:
        self._socket.close()
