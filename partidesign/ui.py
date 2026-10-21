"""Small terminal views for the local CLI; no model or network imports."""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import shutil
import sys


def _console(stream=None):
    stream = sys.stdout if stream is None else stream
    if not getattr(stream, "isatty", lambda: False)() or os.environ.get("TERM") == "dumb" or "NO_COLOR" in os.environ:
        return None
    try:
        from rich.console import Console
    except ImportError:
        return None
    return Console(file=stream, force_terminal=True, markup=False, highlight=False)


def _text(value, style=None):
    from rich.text import Text
    return Text(str(value), style=style)


_ACCENT = "cyan"
_ACCENT_ALT = "bright_blue"


def _status_style(status):
    value = str(status)
    return _STATUS_STYLE.get(value, _STATUS_STYLE.get(value.upper(), _ACCENT_ALT))


def status_badge(status):
    """Return a width-safe Rich status badge for a human-facing view."""
    return _text(f"[{status}]", _status_style(status))


def section_heading(title):
    """Return the shared Rich section heading used by terminal views."""
    return _text(title, f"bold {_ACCENT_ALT}")


def _append_status(value, status):
    value = _text(value, _ACCENT)
    value.append(" ")
    value.append(status_badge(status))
    return value


def _banner(console, command, root=None):
    from rich.box import ROUNDED
    from rich.panel import Panel
    from rich.text import Text
    title = Text("PartiDesign", style=f"bold {_ACCENT}")
    title.append("  /  " + str(command), style=f"bold {_ACCENT_ALT}")
    if root is not None:
        title.append("\nModel root: ", style="dim")
        title.append(str(root), style="dim")
    console.print(Panel(title, box=ROUNDED, border_style=_ACCENT, padding=(0, 1),
                        expand=True))


def _section(console, title, *, first=False):
    if not first:
        console.print()
    console.print(section_heading(title))


def _help_metavar(action):
    if action.metavar:
        return str(action.metavar)
    if action.choices is not None:
        # Choices can be useful to argparse's parser but are too wide for a
        # human-facing first column (especially for --model and --method).
        compact = {
            "targets": "TARGET",
            "model": "TARGET",
            "route_ckpt": "MEMBER",
            "method": "METHOD",
            "route": "true|false",
            "compile": "true|false",
            "overwrite": "true|false",
        }
        if action.dest in compact:
            return compact[action.dest]
    value = str(action.dest or "value").replace("_", "-")
    return value.upper()


def _help_invocation(action):
    """Build a compact invocation without expanding every choice."""
    if isinstance(action, argparse._SubParsersAction):
        return action.dest
    names = ", ".join(action.option_strings)
    if action.nargs == 0:
        return names
    metavar = _help_metavar(action)
    nargs = action.nargs
    if nargs is None:
        suffix = metavar
    elif nargs == "?":
        suffix = f"[{metavar}]"
    elif nargs == "*":
        suffix = f"[{metavar} ...]"
    elif nargs == "+":
        suffix = f"{metavar} [{metavar} ...]"
    elif nargs == argparse.REMAINDER:
        suffix = f"{metavar} ..."
    elif isinstance(nargs, int):
        suffix = " ".join(metavar for _ in range(nargs))
    else:
        suffix = metavar
    return f"{names} {suffix}".strip() if names else suffix


def _command_hint(parser):
    command = str(parser.prog).rsplit(" ", 1)[-1]
    hints = {
        "partidesign": "partidesign check --help",
        "download": "partidesign download --list",
        "design": "partidesign design TASK_FILE.yaml --show-config",
        "check": "partidesign check --all --load",
        "relax": "partidesign relax TASK_FILE.yaml --out RUN_DIR --plan",
        "score": "partidesign score RUN_DIR --show-config",
        "validate": "partidesign validate RUN_OR_TASK_PATH --plan",
        "cluster": "partidesign cluster RUN_OR_TASK_OR_SUCCESS --plan",
    }
    return hints.get(command)


def print_help(parser, file=None):
    stream = sys.stdout if file is None else file
    console = _console(stream)
    if console is None:
        print(parser.format_help(), file=stream, end="")
        return
    from rich.table import Table
    _banner(console, "CLI" if parser.prog == "partidesign" else parser.prog.rsplit(" ", 1)[-1] + " help")
    if parser.description:
        console.print(_text(parser.description))
    console.print(_text(parser.format_usage().strip(), "dim"))
    console.print()
    formatter = argparse.HelpFormatter(parser.prog, width=console.width)
    first = True
    for group in parser._action_groups:
        actions = [a for a in group._group_actions if a.help != argparse.SUPPRESS]
        if not actions:
            continue
        subcommands = any(isinstance(a, argparse._SubParsersAction) for a in actions)
        _section(console, "Commands" if subcommands else group.title.capitalize(), first=first)
        first = False
        compact = console.width < 56
        table = Table(box=None, show_edge=False, expand=True, pad_edge=False,
                      collapse_padding=True, padding=(0, 1), show_header=not compact,
                      header_style=f"bold {_ACCENT_ALT}")
        labels = [choice.dest for a in actions if isinstance(a, argparse._SubParsersAction)
                  for choice in a._get_subactions()] if subcommands else [_help_invocation(a) for a in actions]
        argument_width = min(max(16, max(map(len, labels))), max(12, console.width // 3))
        table.add_column("Command" if subcommands else "Argument", style=_ACCENT,
                         width=argument_width, overflow="fold")
        table.add_column("Description", ratio=1, overflow="fold")
        for action in actions:
            if isinstance(action, argparse._SubParsersAction):
                for choice in action._get_subactions():
                    table.add_row(_text(choice.dest, _ACCENT), _text(choice.help or ""))
            else:
                # Keep argparse's help expansion (including %(default)s), but
                # use a concise invocation in the Rich table.
                description = formatter._expand_help(action) if action.help else ""
                table.add_row(_text(_help_invocation(action), _ACCENT),
                              _text(description))
        console.print(table)
    if parser.epilog:
        console.print()
        console.print(_text(parser.epilog))
    hint = _command_hint(parser)
    if hint:
        console.print()
        console.print(_text("Example: " + hint, "dim"))


def show_error(message, file=None):
    stream = sys.stderr if file is None else file
    console = _console(stream)
    if console is None:
        print(str(message), file=stream)
        return
    from rich.box import ROUNDED
    from rich.panel import Panel
    console.print(Panel(_text(message), title="Failed", title_align="left",
                        border_style="red", box=ROUNDED, padding=(0, 1), expand=True))


def show_context(command, root, detail=None):
    console = _console()
    if console is not None:
        _banner(console, command, root)
        if detail:
            console.print(_text(detail, "dim"))


_STATUS_STYLE = {"PASS": "green", "FAIL": "bold red", "SKIP": "dim", "NOT_RUN": "yellow",
                 "RUN": "cyan", "installed": "green", "downloaded": "green",
                 "unavailable": "yellow", "failed": "bold red",
                 "pending": "yellow", "running": "cyan"}

_SOURCE_LABELS = {
    "git_repo": "with git repo",
    "download_url": "download URL",
    "unconfigured": "not configured",
}


def _source_label(record):
    """Return a stable source label for current and legacy records."""
    source_kind = record.get("source_kind")
    if source_kind in _SOURCE_LABELS:
        return _SOURCE_LABELS[source_kind]
    return "download URL" if record.get("sources") else "not configured"


def _asset_detail(record, root):
    paths = record.get("paths", [])
    detail = str(record.get("error") or "")
    if not detail and paths:
        path = Path(paths[0])
        try:
            detail = str(path.relative_to(root))
        except ValueError:
            detail = str(path)
        if len(paths) > 1:
            detail += f" (+{len(paths) - 1} files)"
    return detail or "local resource"


def _asset_table(records, root, *, listing=False, width=None):
    from rich.table import Table
    width = max(int(width or 100), 35)
    compact = width < 62
    medium = width < 94
    table = Table(box=None, show_edge=False, expand=True, pad_edge=False,
                  collapse_padding=True, padding=(0, 1),
                  show_header=not compact, header_style=f"bold {_ACCENT_ALT}")
    if compact:
        table.add_column("Model", ratio=3, overflow="fold")
        table.add_column("State / detail", ratio=5, overflow="fold")
    elif medium:
        table.add_column("Model", style=_ACCENT, ratio=2, overflow="fold")
        table.add_column("State", width=13, no_wrap=True)
        table.add_column("Location / source", ratio=4, overflow="fold")
    else:
        table.add_column("Model", style=_ACCENT, ratio=2, overflow="fold")
        table.add_column("State", width=13, no_wrap=True)
        table.add_column("Location / detail", ratio=4, overflow="fold")
        table.add_column("Download source" if listing else "Source", ratio=2, overflow="fold")
    root = Path(root)
    for record in records:
        status = str(record["status"])
        detail = _asset_detail(record, root)
        source = _source_label(record)
        if compact:
            location = _text(detail)
            location.append("\n" + source, style="dim")
            cells = [_append_status(record["target"], status), location]
        elif medium:
            location = _text(detail)
            location.append("\n" + source, style="dim")
            cells = [_text(record["target"], _ACCENT), status_badge(status), location]
        else:
            cells = [_text(record["target"], _ACCENT), status_badge(status),
                     _text(detail), _text(source, "dim")]
        table.add_row(*cells)
    return table


def show_inventory(records, root):
    console = _console()
    if console is None:
        for item in records:
            print(f"{item['target']}: {item['status']} ({item.get('error') or 'local resource'}); "
                  f"source: {_source_label(item)}")
        return
    _banner(console, "Model inventory", root)
    groups = (
        ("PartiDesign models", {"partidesign", "partiscore"}),
        ("Sequence design", {"proteinmpnn", "solublempnn", "ligandmpnn", "cyclicmpnn", "nampnn"}),
        ("Structure prediction", {"esmfold2-fast", "esmfold2-standard", "af2", "af3", "af3-la"}),
        ("Chemistry", {"chemistry"}),
    )
    known = set()
    first = True
    for title, targets in groups:
        selected = [item for item in records if item["target"] in targets]
        if selected:
            known.update(item["target"] for item in selected)
            _section(console, title, first=first)
            first = False
            console.print(_asset_table(selected, root, listing=True, width=console.width))
    remaining = [item for item in records if item["target"] not in known]
    if remaining:
        _section(console, "Other resources", first=first)
        console.print(_asset_table(remaining, root, listing=True, width=console.width))
    console.print()
    console.print(_text("Local presence is separate from strict model loading. Use check --load to verify weights.", "dim"))


class CheckDisplay:
    def __init__(self, root):
        self.root = root
        self.console = _console()
        self.rows = []
        self.running = None
        self.live = None

    def _table(self):
        from rich.table import Table
        width = max(int(self.console.width if self.console is not None else 100), 35)
        compact = width < 62
        table = Table(box=None, show_edge=False, expand=True, pad_edge=False,
                      collapse_padding=True, padding=(0, 1),
                      show_header=not compact, header_style=f"bold {_ACCENT_ALT}")
        if compact:
            table.add_column("Check / state", ratio=2, overflow="fold")
            table.add_column("Detail", ratio=3, overflow="fold")
        else:
            table.add_column("Check", style=_ACCENT, ratio=2, overflow="fold")
            table.add_column("State", width=13, no_wrap=True)
            table.add_column("Detail", ratio=5, overflow="fold")
        visible = [row for row in self.rows
                   if not (row[1] == "SKIP" and row[2] == "unused optional model")]
        optional = len(self.rows) - len(visible)
        if optional:
            visible.append(("Optional models", "SKIP", f"{optional} not selected; use --all to check them"))
        for label, status, detail in visible:
            if compact:
                table.add_row(_append_status(label, status), _text(detail))
            else:
                table.add_row(_text(label, _ACCENT), status_badge(status), _text(detail))
        if self.running is not None:
            if compact:
                table.add_row(_append_status(self.running, "RUN"), _text("checking...", "dim"))
            else:
                table.add_row(_text(self.running, _ACCENT), status_badge("RUN"),
                              _text("checking...", "dim"))
        return table

    def __enter__(self):
        if self.console is not None:
            from rich.live import Live
            _banner(self.console, "Check", self.root)
            self.live = Live(self._table(), console=self.console, refresh_per_second=4)
            self.live.start()
        return self

    def begin(self, label):
        self.running = label
        if self.live is not None:
            self.live.update(self._table(), refresh=True)

    def emit(self, label, status, detail):
        self.rows.append((str(label), str(status), str(detail)))
        self.running = None
        if self.live is None:
            print(f"{status} {label}: {detail}", flush=True)
        else:
            self.live.update(self._table(), refresh=True)

    def __exit__(self, exc_type, exc, tb):
        if self.live is not None:
            self.live.stop()
            if exc_type is None:
                counts = {s: sum(row[1] == s for row in self.rows) for s in ("PASS", "FAIL", "SKIP", "NOT_RUN")}
                summary = _text("")
                for index, status in enumerate(counts):
                    if index:
                        summary.append("  /  ")
                    summary.append(f"{counts[status]} {status.lower()}",
                                   style=_status_style(status))
                self.console.print(summary)
        return False


class DownloadDisplay:
    def __init__(self, root, targets):
        self.root = root
        self.targets = list(targets)
        self.console = _console()
        self.progress = None
        self.tasks = {}
        self.done = False

    def __enter__(self):
        if self.console is not None:
            from rich.progress import (
                BarColumn,
                DownloadColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
                TransferSpeedColumn,
            )
            from rich.table import Column
            _banner(self.console, "Download", self.root)
            backend = ("aria2c / up to 8 connections per file"
                       if shutil.which("aria2c")
                       else "Built-in HTTP / aria2c not found")
            self.console.print(_text(backend, "dim"))
            width = max(int(self.console.width), 35)
            description = Column(ratio=3, overflow="fold", no_wrap=False)
            bar = Column(ratio=2, min_width=8, no_wrap=True)
            columns = [
                SpinnerColumn(spinner_name="line", style=_ACCENT),
                TextColumn("{task.description}", markup=False, table_column=description),
            ]
            if width < 48:
                columns.append(BarColumn(bar_width=None, table_column=bar))
            elif width < 78:
                columns.extend([
                    BarColumn(bar_width=None, table_column=bar),
                    DownloadColumn(),
                    TimeElapsedColumn(),
                ])
            else:
                columns.extend([
                    BarColumn(bar_width=None, table_column=bar),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeElapsedColumn(),
                ])
            self.progress = Progress(*columns, console=self.console, expand=True)
            self.progress.start()
        return self

    def on_event(self, event):
        kind = event.get("event")
        label = str(event.get("label") or "")
        if kind == "prepare":
            message = f"Preparing {label}…"
            if self.console is not None:
                self.console.print(_text(message, "dim"))
            else:
                print(message, file=sys.stderr, flush=True)
        elif kind == "fallback":
            reason = str(event.get("reason") or "aria2c unavailable")
            message = f"Fallback to built-in HTTP for {label}: {reason}"
            if self.progress is not None and label in self.tasks:
                task_id = self.tasks.pop(label)
                self.progress.update(task_id, description=label + " / aria2c failed; retrying")
                self.progress.stop_task(task_id)
            if self.console is not None:
                self.console.print(_text(message, "yellow"))
            else:
                print(message, file=sys.stderr, flush=True)
        elif kind == "transfer_start":
            backend = str(event.get("backend") or "HTTP")
            if self.progress is not None:
                task = self.progress.add_task(label + " / " + backend, total=event.get("total"))
                self.tasks[label] = task
            else:
                print(f"Downloading {label} ({backend})", file=sys.stderr, flush=True)
        elif kind == "transfer_progress" and self.progress is not None and label in self.tasks:
            updates = {}
            if event.get("completed") is not None:
                updates["completed"] = event["completed"]
            if event.get("total") is not None:
                updates["total"] = event["total"]
            if updates:
                self.progress.update(self.tasks[label], **updates)
        elif kind == "transfer_done" and self.progress is not None and label in self.tasks:
            task_id = self.tasks.pop(label)
            task = self.progress.tasks[task_id]
            # Unknown-size transfers stay indeterminate until the actual fetch completes.
            actual = event.get("completed", task.completed)
            if task.total is None:
                if actual:
                    self.progress.update(task_id, total=actual, completed=actual)
            elif actual is not None:
                # The done event is authoritative about lifecycle, but only
                # its observed byte count is authoritative about progress.
                self.progress.update(task_id, completed=actual)
            self.progress.stop_task(task_id)

    def finish(self, records):
        self.done = True
        if self.progress is not None:
            self.progress.stop()
            self.progress = None
        if self.console is not None:
            self.console.print(_asset_table(records, self.root, width=self.console.width))
        else:
            for item in records:
                detail = f"{item['target']}: {item['status']}; source: {_source_label(item)}"
                if item.get("error"):
                    detail += f" - {item['error']}"
                print(detail)

    def __exit__(self, exc_type, exc, tb):
        if self.progress is not None:
            self.progress.stop()
            self.progress = None
        return False
