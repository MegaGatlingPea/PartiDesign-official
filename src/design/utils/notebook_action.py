"""Buttons that apply a single browser-state snapshot before their action."""
from __future__ import annotations

from pathlib import Path

import anywidget
import ipywidgets as W
import traitlets


class SyncedActionButton(anywidget.AnyWidget):
    _esm = Path(__file__).with_name("notebook_action.js")
    description = traitlets.Unicode("").tag(sync=True)
    button_style = traitlets.Unicode("").tag(sync=True)
    disabled = traitlets.Bool(False).tag(sync=True)
    bindings = traitlets.Dict(value_trait=traitlets.Instance(W.Widget)).tag(
        sync=True, **W.widget_serialization)
    request = traitlets.Dict().tag(sync=True)
    ack = traitlets.Unicode("").tag(sync=True)
    error = traitlets.Unicode("").tag(sync=True)

    def __init__(self, *, bindings=None, on_error=None, **kwargs):
        self._handlers = []
        self._seen_requests = set()
        self._on_error = on_error
        super().__init__(bindings=dict(bindings or {}), **kwargs)
        self.add_class("pd-synced-action")
        self.observe(self._requested, names="request")

    def on_click(self, callback, remove=False):
        if remove:
            if callback in self._handlers:
                self._handlers.remove(callback)
        elif callback not in self._handlers:
            self._handlers.append(callback)

    def _report_error(self, exc):
        self.error = str(exc)
        if self._on_error is not None:
            self._on_error(exc)

    def click(self):
        """Programmatic clicks use current Python state, like ipywidgets.Button."""
        self.error = ""
        try:
            for callback in tuple(self._handlers):
                callback(self)
        except Exception as exc:
            self._report_error(exc)

    def _validated_snapshot(self, values):
        if not isinstance(values, dict) or set(values) != set(self.bindings):
            raise ValueError("The controls changed. Review the current settings and try again.")
        updates = []
        for key, control in self.bindings.items():
            item = values[key]
            if not isinstance(item, dict) or item.get("model_id") != control.model_id:
                raise ValueError("The task controls changed. Review the current settings and try again.")
            if item.get("kind") == "index":
                labels = list(getattr(control, "_options_labels", ()))
                if item.get("labels") != labels or "index" not in control.traits():
                    raise ValueError("The available choices changed. Select the current option again.")
                value = item.get("value")
                if isinstance(control, W.SelectMultiple):
                    if (not isinstance(value, (list, tuple))
                            or any(type(index) is not int or not 0 <= index < len(labels) for index in value)
                            or len(value) != len(set(value))):
                        raise ValueError("Choose valid, unique options.")
                    value = tuple(value)
                elif value is not None and (type(value) is not int or not 0 <= value < len(labels)):
                    raise ValueError("Choose an available option.")
                field = "index"
            elif item.get("kind") == "value" and "value" in control.traits():
                field = "value"
                value = control.traits()[field]._validate(control, item.get("value"))
            else:
                raise ValueError("This control cannot be used for this action.")
            updates.append((control, field, value))
        return updates

    def _requested(self, change):
        request = change["new"]
        request_id = request.get("id")
        if not isinstance(request_id, str) or not request_id:
            return
        if request_id in self._seen_requests:
            self.ack = request_id
            return
        self._seen_requests.add(request_id)
        self.error = ""
        try:
            for control, field, value in self._validated_snapshot(request.get("values")):
                setattr(control, field, value)
            for callback in tuple(self._handlers):
                callback(self)
        except Exception as exc:
            self._report_error(exc)
        finally:
            self.ack = request_id
