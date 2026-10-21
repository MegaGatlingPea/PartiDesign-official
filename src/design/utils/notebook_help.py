"""Immediate client-side help without a second native browser tooltip."""
from pathlib import Path
import anywidget
import traitlets


class HelpTooltip(anywidget.AnyWidget):
    _esm = Path(__file__).with_name("notebook_help.js")
    help_text = traitlets.Unicode("").tag(sync=True)
    description = traitlets.Unicode("").tag(sync=True)

    def __init__(self, *, tooltip="", **kwargs):
        super().__init__(help_text=tooltip, tooltip="", **kwargs)
