"""Compact keyboard-controlled download selection; catalog reads only."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Mapping, Sequence


def can_prompt() -> bool:
    return (os.environ.get("TERM") != "dumb"
            and bool(getattr(sys.stdin, "isatty", lambda: False)())
            and bool(getattr(sys.stdout, "isatty", lambda: False)()))


def _literal(value: object) -> str:
    return "".join(c if c.isprintable() else " " for c in str(value))


def _size_text(item: Mapping[str, object]) -> str:
    from .resource_info import format_bytes
    actual = item.get("local_bytes")
    if actual is not None and (int(actual) > 0 or int(item.get("present_count") or 0) > 0):
        return format_bytes(actual)
    expected = item.get("expected_bytes")
    if expected is not None:
        return format_bytes(expected)
    hint = item.get("size_hint_bytes")
    prefix = "~" if hint is not None and str(item.get("size_hint_label", "")).startswith("approx") else ""
    return prefix + format_bytes(hint)


def _choice_fragments(name: str, item: Mapping[str, object], *, enabled: bool,
                      unicode: bool = True) -> list[tuple[str, str]]:
    if item.get("status") == "installed":
        marker, style = ("✓" if unicode else "+"), "class:ready"
    elif item.get("integrity") in {"partial", "invalid"}:
        marker, style = "!", "class:pending"
    else:
        marker, style = " ", "class:muted"
    return [(style, marker + " "),
            ("class:model" if enabled else "class:disabled", _literal(name).ljust(21)),
            ("class:muted", "  " + _size_text(item))]


def _status_text(item: Mapping[str, object]) -> str:
    integrity = str(item.get("integrity") or "")
    if item.get("status") == "installed":
        return "Verified" if integrity == "checksum_verified" else "Available"
    invalid = [f for f in item.get("files") or [] if f.get("integrity") == "invalid"]
    if invalid:
        issue = invalid[0]
        detail = str(issue.get("detail") or "").lower()
        filename = Path(str(issue.get("relative") or issue.get("path") or "file")).name
        if "sha256" in detail:
            return "Checksum mismatch: " + filename
        if "size mismatch" in detail:
            return "Size mismatch: " + filename
        if "empty" in detail:
            return "Empty file: " + filename
        if "symlink" in detail or "escapes" in detail:
            return "Unsafe path: " + filename
        if "read" in detail or "permission" in detail:
            return "Cannot read: " + filename
        return "Invalid file: " + filename
    if integrity == "partial":
        return "Incomplete: " + str(len(item.get("missing") or [])) + " missing"
    if item.get("source_kind") == "git_repo":
        return "Restore files from Git"
    if not item.get("selectable", True):
        return "Local files required"
    return "Not downloaded"


def _detail_text(name: str, item: Mapping[str, object], root: Path) -> str:
    from .resource_info import target_metadata
    short = str(item.get("short_description") or target_metadata(name)["short_description"])
    count = item.get("file_count", len(item.get("paths") or []))
    present = item.get("present_count", count if item.get("status") == "installed" else 0)
    return "\n".join(_literal(line) for line in [
        "Usage: " + short,
        "Size: " + _size_text(item) + " · Files: " + str(present) + "/" + str(count),
        "Status: " + _status_text(item),
    ])


def select_downloads(options: Sequence[Mapping[str, object]], root: Path) -> list[str] | None:
    """Installed ticks indicate local assets; checkboxes select downloads."""
    if not options:
        return None
    from prompt_toolkit.application import Application, get_app
    from prompt_toolkit.filters import has_focus
    from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
    from prompt_toolkit.key_binding.defaults import load_key_bindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.output import ColorDepth
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import Button, CheckboxList, Dialog, Label
    from ._mascot import PROMPT_ACCENT, supports_unicode

    records = {str(item["target"]): item for item in options}
    names = list(records)
    enabled = {name for name, item in records.items() if item.get("selectable")}
    error = [None, ""]

    class ModelChoices(CheckboxList):
        def _handle_enter(self):
            name = self.values[self._selected_index][0]
            if name not in enabled:
                error[:] = [name, _status_text(records[name])]
                get_app().invalidate()
                return
            error[:] = [None, ""]
            super()._handle_enter()

    unicode = supports_unicode(sys.stdout)
    values = [(name, _choice_fragments(name, item, enabled=name in enabled, unicode=unicode))
              for name, item in records.items()]
    choices = ModelChoices(values, default_values=[])
    choices.control.key_bindings.remove("enter")
    choices.window.height = Dimension(min=min(3, len(values)), preferred=len(values), max=len(values))
    if enabled:
        choices._selected_index = next(i for i, name in enumerate(names) if name in enabled)

    def accept():
        if choices.current_values:
            selected = set(choices.current_values)
            get_app().exit(result=[name for name in names if name in selected and name in enabled])

    def cancel():
        get_app().exit(result=None)

    def detail():
        name = names[choices._selected_index]
        return [("class:muted", _detail_text(name, records[name], root))]

    def status():
        name = names[choices._selected_index]
        value = error[1] if error[0] == name else "Selected: " + str(len(choices.current_values))
        return [("class:pending", _literal(value))]

    bindings = KeyBindings()
    @bindings.add("enter", filter=has_focus(choices.control))
    def _download(event):
        accept()
    @bindings.add("escape", eager=True)
    @bindings.add("c-d", eager=True)
    def _back(event):
        cancel()
    @bindings.add("c-c", eager=True)
    def _interrupt(event):
        event.app.exit(exception=KeyboardInterrupt())

    guide = "↑↓ Move · Space Select · Enter Download · Esc Back" if unicode else "Up/Down Move | Space Select | Enter Download | Esc Back"
    body = HSplit([
        Label(text=[("class:muted", "Model root: " + _literal(root))], dont_extend_height=True),
        Window(height=1), choices, Window(height=1),
        Label(text=detail, dont_extend_height=True),
        Label(text=status, dont_extend_height=True),
        Label(text=guide, dont_extend_height=True),
    ])
    colored = (os.environ.get("TERM") != "dumb" and "NO_COLOR" not in os.environ
               and bool(getattr(sys.stdout, "isatty", lambda: False)()))
    title = ([("fg:ansidefault nobold", "📦 "), ("", "PartiDesign / Model downloads")]
             if colored and unicode else "PartiDesign / Model downloads")
    dialog = Dialog(title=title, body=body,
                    buttons=[Button(text="Download", handler=accept), Button(text="Back", handler=cancel)],
                    width=Dimension(preferred=100, max=110), with_background=True)
    styles = {
        "dialog": "bg:#1c1b16 #f1ecdc", "dialog frame.label": "bold " + PROMPT_ACCENT,
        "dialog frame.border": PROMPT_ACCENT, "dialog.body": "bg:#1c1b16 #f1ecdc",
        "dialog shadow": "bg:#0c0d09", "button": "bg:#353123 #eee5cb",
        "button.focused": "bg:" + PROMPT_ACCENT + " #242013 bold",
        "model": PROMPT_ACCENT + " bold", "ready": "#86efac", "pending": PROMPT_ACCENT,
        "muted": "#b2ae9d", "disabled": "#8d8b7d", "checkbox": "#f1ecdc",
        "checkbox-selected": "bg:#413a26", "checkbox-checked": "#86efac bold",
    }
    app = Application(layout=Layout(dialog), full_screen=True, mouse_support=True,
                      key_bindings=merge_key_bindings([load_key_bindings(), bindings]),
                      style=Style.from_dict(styles),
                      color_depth=ColorDepth.MONOCHROME if "NO_COLOR" in os.environ else None)
    try:
        return app.run()
    except EOFError:
        return None
