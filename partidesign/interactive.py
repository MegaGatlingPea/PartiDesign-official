"""Keyboard-controlled terminal selection; immutable catalog reads only."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Mapping, Sequence


def can_prompt() -> bool:
    """Never wait for input in pipes, scripts, or a dumb terminal."""
    return (os.environ.get("TERM") != "dumb"
            and bool(getattr(sys.stdin, "isatty", lambda: False)())
            and bool(getattr(sys.stdout, "isatty", lambda: False)()))


def _literal(value: object) -> str:
    return "".join(c if c.isprintable() else " " for c in str(value))


def select_downloads(options: Sequence[Mapping[str, object]], root: Path) -> list[str] | None:
    """Return selected catalog names, or None after an explicit cancel/EOF."""
    from prompt_toolkit.application import Application, get_app
    from prompt_toolkit.filters import has_focus
    from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
    from prompt_toolkit.key_binding.defaults import load_key_bindings
    from prompt_toolkit.layout import HSplit, Layout
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.output import ColorDepth
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import Button, CheckboxList, Dialog, Label

    records = {str(item["target"]): item for item in options}
    names = list(records)
    enabled = {name for name, item in records.items() if item.get("selectable")}
    message = ["Nothing is selected. Existing files are kept unless --force was given."]

    class ModelChoices(CheckboxList):
        def _handle_enter(self):
            name = self.values[self._selected_index][0]
            if name not in enabled:
                message[0] = str(records[name].get("selection_detail") or "This resource is unavailable.")
                get_app().invalidate()
                return
            super()._handle_enter()
            message[0] = f"Selected {len(self.current_values)} model(s). Enter to continue; Esc to cancel."

    values = []
    for name, item in records.items():
        ready = item.get("status") == "installed"
        source = {"git_repo": "with git repo", "download_url": "download URL",
                  "unconfigured": "local only" if ready else "local files required"}.get(str(item.get("source_kind")), "local files required")
        if name not in enabled:
            fragments = [("class:disabled", f"{name}  [manual]  {source}")]
        else:
            fragments = [("class:model", name), ("class:muted", "  "),
                         ("class:ready" if ready else "class:pending", "[ready]" if ready else "[download]"),
                         ("class:muted", "  " + source)]
        values.append((name, fragments))
    choices = ModelChoices(values, default_values=[])
    # Space toggles; Enter confirms instead of toggling the focused row.
    choices.control.key_bindings.remove("enter")
    if enabled:
        choices._selected_index = next(i for i, name in enumerate(names) if name in enabled)

    def accept():
        if not choices.current_values:
            message[0] = "Select at least one model with Space, or press Esc to cancel."
            get_app().invalidate()
            return
        chosen = set(choices.current_values)
        get_app().exit(result=[name for name in names if name in chosen and name in enabled])

    def cancel():
        get_app().exit(result=None)

    def detail():
        item = records[names[choices._selected_index]]
        reason = str(item.get("selection_detail") or "")
        if reason:
            return [("class:muted", _literal(reason))]
        paths = item.get("paths") or []
        if paths:
            path = Path(str(paths[0]))
            try:
                location = str(path.relative_to(root))
            except ValueError:
                location = str(path)
            return [("class:muted", "Files: " + _literal(location))]
        return []

    bindings = KeyBindings()
    @bindings.add("enter", filter=has_focus(choices.control))
    def _continue(event):
        accept()
    @bindings.add("escape", eager=True)
    @bindings.add("c-d", eager=True)
    def _cancel(event):
        cancel()
    @bindings.add("c-c", eager=True)
    def _interrupt(event):
        event.app.exit(exception=KeyboardInterrupt())
    @bindings.add("a", filter=has_focus(choices.control))
    def _select_all(event):
        choices.current_values = [name for name in names if name in enabled]
        message[0] = f"Selected {len(choices.current_values)} model(s). Enter to continue."
    @bindings.add("n", filter=has_focus(choices.control))
    def _select_none(event):
        choices.current_values = []
        message[0] = "Selection cleared. Space selects a model; Esc cancels."

    body = HSplit([
        Label(text=[("class:muted", "Model root: " + _literal(root))], dont_extend_height=True),
        Label(text="Up/Down move · Space select · Enter continue · Esc cancel", dont_extend_height=True),
        choices,
        Label(text=detail, dont_extend_height=True),
        Label(text=lambda: [("class:pending", _literal(message[0]))], dont_extend_height=True),
    ], padding=1)
    dialog = Dialog(title="PartiDesign / Model downloads", body=body,
                    buttons=[Button(text="Continue", handler=accept), Button(text="Cancel", handler=cancel)],
                    width=Dimension(preferred=100, max=110), with_background=True)
    style = Style.from_dict({
        "dialog": "bg:#101827 #e5e7eb", "dialog frame.label": "bold #67e8f9",
        "dialog frame.border": "#22d3ee", "dialog.body": "bg:#101827 #e5e7eb",
        "dialog shadow": "bg:#080c14", "button": "bg:#1f2937 #d1d5db",
        "button.focused": "bg:#0284c7 #ffffff bold", "model": "#67e8f9 bold",
        "ready": "#86efac", "pending": "#fcd34d", "muted": "#9ca3af",
        "disabled": "#6b7280", "checkbox": "#e5e7eb",
        "checkbox-selected": "bg:#1e3a5f", "checkbox-checked": "#86efac bold",
    })
    app = Application(layout=Layout(dialog), full_screen=True, mouse_support=True,
                      key_bindings=merge_key_bindings([load_key_bindings(), bindings]), style=style,
                      color_depth=ColorDepth.MONOCHROME if "NO_COLOR" in os.environ else None)
    try:
        return app.run()
    except EOFError:
        return None
