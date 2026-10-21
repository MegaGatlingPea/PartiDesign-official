"""Native AlphaFold 3 reward configuration marker."""

from __future__ import annotations

from typing import Any


class AF3Reward:
  """Configuration marker; inference is deferred until model offload."""

  def __init__(self, **config: Any) -> None:
    self.config = dict(config)
