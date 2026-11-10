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


from ._mascot import ACCENT as _ACCENT

_ACCENT_ALT = _ACCENT


def _status_style(status):
    value = str(status)
    return _STATUS_STYLE.get(value, _STATUS_STYLE.get(value.upper(), _ACCENT_ALT))


def status_badge(status):
    """Return a width-safe Rich status badge for a human-facing view."""
    return _text(f"[{status}]", _status_style(status))


def section_heading(title, *, emoji=None):
    """A warm label; native emoji keep default foreground and normal weight."""
    from rich.style import Style
    from rich.text import Text
    text = Text()
    if emoji:
        text.append(emoji + " ", style=Style(color="default", bold=False))
    text.append(str(title), style=f"bold {_ACCENT_ALT}")
    return text


def _append_status(value, status):
    value = _text(value, _ACCENT)
    value.append(" ")
    value.append(status_badge(status))
    return value


def _banner(console, command, root=None, *, mascot=True, description=None):
    from rich.box import ROUNDED
    from rich.panel import Panel
    from rich.text import Text
    if not mascot:
        # Existing relax --plan text/layout is retained; only its hue changes.
        title = Text("PartiDesign", style=f"bold {_ACCENT}")
        title.append("  /  " + str(command), style=f"bold {_ACCENT_ALT}")
        if root is not None:
            title.append("\nModel root: ", style="dim")
            title.append(str(root), style="dim")
        console.print(Panel(title, box=ROUNDED, border_style=_ACCENT, padding=(0, 1), expand=True))
        return False
    from rich.console import Group
    from rich.style import Style
    from ._mascot import IMAGE_WIDTH, supports_unicode, native_image, place_native_image
    key = str(command).lower().removesuffix(" help")
    icons = {"cli": "✨", "design": "🧬", "download": "📦", "model inventory": "📦",
             "pipeline": "🚀", "screen": "🔬", "check": "🔎", "validate": "🔎", "cluster": "🧩",
             "postfilter": "🔎", "score": "✨", "msa": "🧬", "relax": "✨"}
    title = Text()
    if console.width >= 60 and supports_unicode(console.file):
        title.append(icons.get(key, "✨") + " ", style=Style(color="default", bold=False))
    title.append("PartiDesign", style=f"bold {_ACCENT}")
    title.append("  /  " + str(command), style=f"bold {_ACCENT_ALT}")
    intro = Text(str(description)) if description else Text("Model root: " + str(root), style="dim") if root is not None else Text("")
    image = native_image(console, root_help=key == "cli")
    text_width = max(1, console.width - 4 - (IMAGE_WIDTH + 2 if image else 0))
    intro_rows = len(console.render_lines(intro, console.options.update(width=text_width), pad=False))
    # Keep a wrapped introduction complete without increasing the compact frame.
    text = Group(title, intro) if intro_rows > 1 else Group(title, Text(""), intro)
    if image:
        from rich.table import Table
        content = Table.grid(padding=0, expand=True)
        content.add_column(width=IMAGE_WIDTH, no_wrap=True, vertical="middle")
        content.add_column(width=2)
        content.add_column(ratio=1, vertical="middle")
        artwork = Text("        \n        \n        ")
        content.add_row(artwork, Text(""), text)
    else:
        content = text
    panel = Panel(content, box=ROUNDED, border_style=_ACCENT, padding=(0, 1), expand=True)
    height = len(console.render_lines(panel, console.options, pad=False)) if image else None
    if image and console.height <= height:
        image = None
        panel = Panel(text, box=ROUNDED, border_style=_ACCENT, padding=(0, 1), expand=True)
    console.print(panel)
    if image:
        place_native_image(console, image, height)
    return bool(description)


def _section(console, title, *, first=False):
    if not first:
        console.print()
    from ._mascot import supports_unicode
    icon = {"Commands": "🧭", "Options": "⚙️", "Positional arguments": "🧩",
            "Machine overview": "🖥️", "PartiDesign models": "🧪",
            "Sequence design": "🧬", "Structure prediction": "🔬"}.get(title)
    heading = section_heading(title, emoji=icon if supports_unicode(console.file) else None)
    from ._heading import print_section_heading
    if not print_section_heading(console, heading, title):
        console.print(heading, justify="left")


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
            "compile": "auto|true|false",
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
        "partidesign": "partidesign check",
        "download": "partidesign download --list",
        "design": "partidesign design TASK_FILE.yaml --show-config",
        "check": "partidesign check",
        "relax": "partidesign relax TASK_FILE.yaml --out RUN_DIR --plan",
        "score": "partidesign score RUN_DIR --show-config",
        "validate": "partidesign validate RUN_OR_TASK_PATH --plan",
        "screen": "partidesign screen SCREEN.yaml --plan",
        "cluster": "partidesign cluster RUN_OR_TASK_OR_SUCCESS --plan",
    }
    return hints.get(command)


def _catalog_records():
    try:
        from .resource_info import target_catalog
    except ImportError:
        return []
    return target_catalog()


def _bytes(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return "unknown"
    try:
        from .resource_info import format_bytes
    except ImportError:
        amount = float(value)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
            if amount < 1024 or unit == "PiB":
                return f"{amount:.1f} {unit}"
            amount /= 1024
    return format_bytes(value)


def _short_use(record):
    return str(record.get("short_description") or " ".join(str(record.get("description") or "Model resource").split()[:5]))


def _resource_catalog_text():
    records = _catalog_records()
    if not records:
        return ""
    return "\nResource targets\n\n" + "\n".join(
        f"  {item['target']:<19} {_short_use(item)}" for item in records) + "\n"


def _show_resource_catalog(console):
    records = _catalog_records()
    if not records:
        return
    from rich.box import ROUNDED
    from rich.table import Table
    _section(console, "Resource targets")
    paired = console.width >= 100
    table = Table(box=ROUNDED, show_lines=False, expand=True,
                  border_style=_ACCENT, header_style=f"bold {_ACCENT}")
    for _ in range(2 if paired else 1):
        table.add_column("Model", ratio=2, justify="center", vertical="middle", overflow="fold")
        table.add_column("Usage", ratio=3, justify="center", vertical="middle", overflow="fold")
    if paired:
        half = (len(records) + 1) // 2
        for index, item in enumerate(records[:half]):
            other = records[index+half] if index+half < len(records) else None
            table.add_row(_text(item["target"], _ACCENT), _text(_short_use(item)),
                          _text(other["target"], _ACCENT) if other else _text(""),
                          _text(_short_use(other)) if other else _text(""))
    else:
        for item in records:
            table.add_row(_text(item["target"], _ACCENT), _text(_short_use(item)))
    console.print(table)


def print_help(parser, file=None):
    stream = sys.stdout if file is None else file
    console = _console(stream)
    if console is None:
        print(parser.format_help(), file=stream, end="")
        return
    from rich.table import Table
    from rich.text import Text
    summary, separator, remaining_description = str(parser.description or "").partition("\n\n")
    description_shown = _banner(
        console, "CLI" if parser.prog == "partidesign" else parser.prog.rsplit(" ", 1)[-1] + " help",
        description=summary,
    )
    if parser.description and not description_shown:
        console.print(_text(parser.description))
    elif separator and remaining_description:
        console.print(_text(remaining_description))
    console.print(_text(parser.format_usage().strip(), "dim"))
    console.print()
    formatter = parser.formatter_class(parser.prog, width=console.width)
    first = True
    for group in parser._action_groups:
        actions = [a for a in group._group_actions if a.help != argparse.SUPPRESS]
        if not actions:
            continue
        subcommands = any(isinstance(a, argparse._SubParsersAction) for a in actions)
        _section(console, "Commands" if subcommands else group.title.capitalize(), first=first)
        first = False
        from rich.box import ROUNDED
        from rich.console import Group
        compact = console.width < 56
        table = Table(box=ROUNDED, show_edge=True, show_lines=True, expand=True,
                      pad_edge=True, collapse_padding=True, padding=(0, 1),
                      header_style=f"bold {_ACCENT_ALT}", border_style=_ACCENT)
        labels = [choice.dest for a in actions if isinstance(a, argparse._SubParsersAction)
                  for choice in a._get_subactions()] if subcommands else [_help_invocation(a) for a in actions]
        if compact:
            table.add_column(Text("Command / Description" if subcommands else "Argument / Description", justify="center"),
                             justify="center", vertical="middle", overflow="fold")
        else:
            argument_width = min(max(16, max(map(len, labels))), max(12, console.width // 3))
            table.add_column(Text("Command" if subcommands else "Argument", justify="center"), width=argument_width,
                             justify="center", vertical="middle", overflow="fold")
            table.add_column(Text("Description", justify="center"), ratio=1, justify="center",
                             vertical="middle", overflow="fold")
        def add_help_row(label, description, command=False):
            left = _text(label, f"bold {_ACCENT}" if command else None)
            right = _text(description)
            table.add_row(Group(left, right)) if compact else table.add_row(left, right)
        for action in actions:
            if isinstance(action, argparse._SubParsersAction):
                for choice in action._get_subactions():
                    add_help_row(choice.dest, choice.help or "", command=True)
            else:
                description = formatter._expand_help(action) if action.help else ""
                add_help_row(_help_invocation(action), description)
        console.print(table)
    if parser.epilog:
        console.print()
        console.print(_text(parser.epilog))
    if parser.prog.rsplit(" ", 1)[-1] in {"download", "check"}:
        _show_resource_catalog(console)
    hint = _command_hint(parser)
    if hint:
        console.print()
        from ._mascot import supports_unicode
        console.print(section_heading("Try next", emoji="💡" if supports_unicode(console.file) else None))
        console.print(_text(hint))


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


def show_context(command, root, detail=None, *, mascot=True):
    console = _console()
    if console is not None:
        _banner(console, command, root, mascot=mascot)
        if detail:
            console.print(_text(detail, "dim"))


_STATUS_STYLE = {"PASS": "green", "FAIL": "bold red", "WARN": "yellow", "INFO": "white", "SKIP": "dim", "NOT_RUN": "yellow",
                 "RUN": _ACCENT, "installed": "green", "downloaded": "green",
                 "unavailable": "yellow", "failed": "bold red",
                 "pending": "yellow", "running": _ACCENT}

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


def _resource_metadata(record):
    try:
        from .resource_info import target_metadata
        metadata = dict(target_metadata(str(record["target"])))
    except (ImportError, KeyError):
        metadata = {}
    metadata.update(record)
    return metadata


def _resource_state(record):
    status = str(record.get("status", "unavailable"))
    integrity = str(record.get("integrity") or ("presence only" if status in {"installed", "downloaded"} else "not verified"))
    lines = [status, "Integrity: " + integrity]
    if record.get("file_count") is not None:
        lines.append(f"Files: {record.get('present_count', 'unknown')}/{record['file_count']}")
    lines.append("Local: " + _bytes(record.get("local_bytes")))
    if record.get("size_hint_bytes") is not None:
        lines.append("Reference: " + _bytes(record["size_hint_bytes"]))
        lines.append(str(record.get("size_hint_label") or "installed footprint; varies"))
    if record.get("integrity_detail"):
        lines.append(str(record["integrity_detail"]))
    return "\n".join(lines)


def _resource_description(record, root, *, files=False):
    lines = [str(record.get("description") or "Local model resource")]
    lines.append("Used by: " + (", ".join(map(str, record.get("used_by") or [])) or "not specified"))
    if record.get("notes"):
        lines.append("Notes: " + str(record["notes"]))
    lines.append("Source: " + _source_label(record))
    if files:
        lines.extend("  " + str(source) for source in record.get("sources") or [])
    lines.append("Location: " + _asset_detail(record, root))
    if record.get("missing"):
        lines.append("Missing: " + ", ".join(map(str, record["missing"])))
    if files:
        for item in record.get("files") or []:
            path = item.get("relative") or item.get("path") or "unknown file"
            lines.append(f"{path}: {_bytes(item.get('bytes'))}; {item.get('integrity') or 'not verified'}")
            if item.get("detail"):
                lines.append("  " + str(item["detail"]))
    return "\n".join(lines)


def _short_status(record):
    integrity = str(record.get("integrity") or "")
    status = str(record.get("status") or "unavailable")
    if integrity == "checksum_verified":
        return "✓ Verified", "green"
    if integrity == "invalid":
        return "Corrupt", "bold red"
    if integrity == "partial":
        return "Incomplete", "yellow"
    if integrity == "missing":
        return "Missing", "yellow"
    if status in {"installed", "downloaded"}:
        return "✓ Installed", "green"
    if status == "failed":
        return "Failed", "bold red"
    count, present = record.get("file_count"), record.get("present_count")
    return ("Incomplete" if present and count else "Missing"), "yellow"


def _short_size(record):
    local = record.get("local_bytes")
    if isinstance(local, (int, float)) and not isinstance(local, bool) and local > 0:
        return _bytes(local)
    expected = record.get("size_hint_bytes")
    return "~" + _bytes(expected) if expected is not None else "—"


def _asset_table(records, root, *, listing=False, width=None):
    from rich.box import ROUNDED
    from rich.table import Table
    table = Table(box=ROUNDED, show_lines=False, expand=True, padding=(0, 1),
                  header_style=f"bold {_ACCENT}", border_style=_ACCENT)
    table.add_column("Model", width=17, justify="center", vertical="middle", overflow="fold")
    table.add_column("Usage", ratio=1, justify="center", vertical="middle", overflow="fold")
    table.add_column("Size", width=10, justify="center", vertical="middle", overflow="fold")
    table.add_column("Status", width=11, justify="center", vertical="middle", overflow="fold")
    for original in records:
        item = _resource_metadata(original)
        status, style = _short_status(item)
        table.add_row(_text(item["target"], _ACCENT), _text(_short_use(item)),
                      _text(_short_size(item)), _text(status, style))
    return table


_MODEL_GROUPS = (
    ("PartiDesign models", {"partidesign", "partiscore"}),
    ("Sequence design", {"proteinmpnn", "solublempnn", "ligandmpnn", "cyclicmpnn", "nampnn"}),
    ("Structure prediction", {"esmfold2-fast", "esmfold2-standard", "af2", "mol1", "mol1-constraint", "af3", "af3-la", "boltz2"}),
)


def _short_plain_table(records, *, usage_width=None):
    items = [_resource_metadata(record) for record in records]
    usage_width = usage_width or max([len("Usage"), *(len(_short_use(item)) for item in items)])
    print(f"{'Model':<19} {'Usage':<{usage_width}} {'Size':>11}  Status")
    for item in items:
        status, _ = _short_status(item)
        print(f"{str(item['target']):<19} {_short_use(item):<{usage_width}} {_short_size(item):>11}  {status}")


def _inventory_plain(records, root, *, files=False):
    usage_width = max([len("Usage"), *(len(_short_use(_resource_metadata(item))) for item in records)])
    for title, names in _MODEL_GROUPS:
        selected = [item for item in records if item["target"] in names]
        if selected:
            print("\n" + title)
            _short_plain_table(selected, usage_width=usage_width)


def _render_inventory(console, records, root, *, listing=False):
    for index, (title, names) in enumerate(_MODEL_GROUPS):
        selected = [item for item in records if item["target"] in names]
        if selected:
            _section(console, title, first=index == 0)
            console.print(_asset_table(selected, root, width=console.width))


def show_inventory(records, root):
    console = _console()
    if console is None:
        _inventory_plain(records, root)
        return
    _banner(console, "Model inventory", root)
    _render_inventory(console, records, root)


def _meter_values(meter):
    def number(value):
        import math
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None
    total = number(meter.get("total_bytes"))
    used = number(meter.get("used_bytes"))
    available = number(meter.get("available_bytes"))
    if used is None and total is not None and available is not None and available <= total:
        used = total - available
    valid = total is not None and total > 0 and used is not None and used <= total
    return total, used, available, valid


def _meter_caption(meter):
    total, used, available, valid = _meter_values(meter)
    values = f"Used {_bytes(used)} / {_bytes(total)}; available {_bytes(available)}"
    values += f" ({used / total:.1%} used)" if valid else "; utilization unavailable"
    if meter.get("scope"):
        values += "\n" + str(meter["scope"])
    return values


def _render_system(console, system):
    from rich.box import ROUNDED
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.bar import Bar
    _section(console, "System overview", first=True)
    for section in system.get("sections") or []:
        table = Table.grid(expand=True, padding=(0, 1))
        compact = console.width < 62
        table.add_column(ratio=2, justify="center", vertical="middle", overflow="fold")
        if not compact:
            table.add_column(ratio=5, justify="center", vertical="middle", overflow="fold")
        for row in section.get("rows") or []:
            style = "yellow" if row.get("level") == "WARN" else None
            label = _text(row.get("label", ""), _ACCENT)
            value = _text(row.get("value", "unavailable"), style)
            table.add_row(Group(label, value)) if compact else table.add_row(label, value)
        console.print(Panel(table, title=_text(section.get("title", "System"), f"bold {_ACCENT}"),
                            title_align="left", border_style=_ACCENT, box=ROUNDED, padding=(0, 1)))
    for meter in system.get("meters") or []:
        total, used, _, valid = _meter_values(meter)
        content = []
        if valid:
            color = "red" if used / total >= .9 else _ACCENT
            content.append(Bar(total, begin=0, end=used, color=color, bgcolor="grey23"))
        content.append(_text(_meter_caption(meter)))
        console.print(Panel(Group(*content), title=_text(meter.get("label", "Capacity"), f"bold {_ACCENT}"),
                            title_align="left", border_style=_ACCENT, box=ROUNDED, padding=(0, 1)))


def _system_plain(system):
    for section in system.get("sections") or []:
        print("\n" + str(section.get("title", "System")))
        for row in section.get("rows") or []:
            prefix = "WARN " if row.get("level") == "WARN" else ""
            print("  " + prefix + str(row.get("label", "")) + ": " + str(row.get("value", "unavailable")))
    for meter in system.get("meters") or []:
        print("\n" + str(meter.get("label", "Capacity")) + ": " + _meter_caption(meter))


def _machine_rows(system):
    given = {str(item.get("label", "")).upper(): item for item in system.get("machine_overview") or []}
    rows = []
    for label in ("CPU", "GPU", "RAM", "Disk"):
        item = given.get(label.upper()) or (given.get("MEMORY") if label == "RAM" else None)
        if item:
            rows.append({**item, "label": label})
            continue
        value = "unavailable"
        used = total = None
        unit = "%" if label in {"CPU", "GPU"} else "bytes"
        if label in {"CPU", "GPU"}:
            for section in system.get("sections") or []:
                if label not in str(section.get("title", "")).upper():
                    continue
                preferred = [row for row in section.get("rows") or [] if any(
                    key in str(row.get("label", "")).lower() for key in ("usable", "usage", "visible", "device", "model"))]
                if preferred:
                    value = "; ".join(str(row.get("value", "")) for row in preferred[:2])
                break
        else:
            for meter in system.get("meters") or []:
                title = str(meter.get("label", "")).lower()
                matches = ("ram" in title or "memory" in title) if label == "RAM" else ("disk" in title or "filesystem" in title or "storage" in title)
                if not matches:
                    continue
                total, used, available, _ = _meter_values(meter)
                value = f"{_bytes(available)} free / {_bytes(total)} total"
                break
        rows.append({"label": label, "value": value, "used": used, "total": total, "unit": unit})
    return rows


def _capacity_bar(row, width=24):
    from rich.bar import Bar
    total, used = row.get("total"), row.get("used")
    import math
    valid = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (used, total))
    if not valid or total <= 0 or used < 0 or used > total:
        return _text("—", "dim")
    return Bar(total, begin=0, end=used, width=width,
               color="red" if used / total >= .9 else _ACCENT, bgcolor="grey23")


def _render_machine_overview(console, system):
    from rich.box import ROUNDED
    from rich.table import Table
    _section(console, "Machine overview", first=True)
    table = Table(box=ROUNDED, show_header=False, show_lines=False, expand=True,
                  border_style=_ACCENT, padding=(0, 1))
    table.add_column(width=5, justify="center", vertical="middle")
    table.add_column(ratio=1, justify="center", vertical="middle", overflow="fold")
    bar_width = 24 if console.width >= 100 else 16
    table.add_column(width=bar_width, justify="center", vertical="middle")
    for row in _machine_rows(system):
        table.add_row(_text(row["label"], _ACCENT), _text(row.get("value", "unavailable")), _capacity_bar(row, width=bar_width))
    console.print(table)


def _concise_failure(label, detail, resources):
    for record in sorted(resources or [], key=lambda item: len(str(item.get("target", ""))), reverse=True):
        target = str(record.get("target", ""))
        if not target or not (label == target or label.startswith(target + "-")):
            continue
        files = record.get("files") or []
        failed = [item for item in files if item.get("integrity") == "invalid"]
        failed += [item for item in files if item.get("integrity") == "missing"]
        if failed:
            item = failed[0]
            filename = Path(str(item.get("relative") or item.get("path") or target)).name
            reason = str(item.get("detail") or item.get("integrity") or "integrity check failed")
            lower = reason.lower()
            if "mismatch" in lower and ("sha256" in lower or "checksum" in lower):
                reason = "checksum mismatch"
            elif "size mismatch" in lower or "incorrect size" in lower:
                reason = "incorrect size"
            elif "cannot inspect" in lower or "unreadable" in lower or "permissionerror" in lower:
                reason = "unreadable"
            elif "symlink" in lower or "unsafe" in lower:
                reason = "unsafe path"
            elif item.get("integrity") == "missing":
                reason = "missing"
            return filename + ": " + " ".join(reason.split())
        return " ".join(str(record.get("error") or "resource integrity check failed").split())
    return " ".join(detail.split())


def _render_concise_check(console, system, resources, root, failures=()):
    if console is None:
        print("Machine overview")
        for row in _machine_rows(system or {}):
            print(f"{row['label']}: {row.get('value', 'unavailable')}")
        _inventory_plain(resources or [], root)
        for label, _, detail in failures:
            print("Failed: " + label + ": " + _concise_failure(label, detail, resources))
        return
    _render_machine_overview(console, system or {})
    console.print()
    _render_inventory(console, resources or [], root)
    for label, _, detail in failures:
        console.print(_text("Failed: " + label + ": " + _concise_failure(label, detail, resources), "bold red"))


class CheckDisplay:
    def __init__(self, root, *, system=None, resources=None, concise=False):
        self.root = root
        self.system = system
        self.resources = resources
        self.concise = concise
        self.console = _console()
        self.rows = []
        self.running = None
        self.live = None

    def _table(self):
        from rich.table import Table
        width = max(int(self.console.width if self.console is not None else 100), 35)
        compact = width < 62
        from rich.box import ROUNDED
        table = Table(box=ROUNDED, show_edge=True, show_lines=True, expand=True,
                      padding=(0, 1), header_style=f"bold {_ACCENT_ALT}", border_style=_ACCENT)
        if compact:
            table.add_column("Check / state", ratio=2, justify="center", vertical="middle", overflow="fold")
            table.add_column("Detail", ratio=3, justify="center", vertical="middle", overflow="fold")
        else:
            table.add_column("Check", style=_ACCENT, ratio=2, justify="center", vertical="middle", overflow="fold")
            table.add_column("State", ratio=1, justify="center", vertical="middle", overflow="fold")
            table.add_column("Detail", ratio=5, justify="center", vertical="middle", overflow="fold")
        visible = self.rows
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

    def update_resources(self, records):
        self.resources = list(records)

    def __enter__(self):
        if self.concise:
            if self.console is not None:
                from rich.live import Live
                self.live = Live(_text("Checking local resources…", "dim"), console=self.console,
                                 refresh_per_second=2, transient=True)
                self.live.start()
            return self
        if self.console is not None:
            from rich.live import Live
            _banner(self.console, "Check", self.root)
            if self.system is not None:
                _render_system(self.console, self.system)
            if self.resources is not None:
                _render_inventory(self.console, self.resources, self.root)
            _section(self.console, "Check results")
            self.live = Live(self._table(), console=self.console, refresh_per_second=4, vertical_overflow="visible")
            self.live.start()
        else:
            if self.system is not None:
                print("System overview")
                _system_plain(self.system)
            if self.resources is not None:
                print("\nAll model resources; model root: " + str(self.root))
                _inventory_plain(self.resources, self.root, files=False)
        return self

    def begin(self, label):
        self.running = label
        if self.live is not None and not self.concise:
            self.live.update(self._table(), refresh=True)

    def emit(self, label, status, detail):
        self.rows.append((str(label), str(status), str(detail)))
        self.running = None
        if self.concise:
            return
        if self.live is None:
            print(f"{status} {label}: {detail}", flush=True)
        else:
            self.live.update(self._table(), refresh=True)

    def __exit__(self, exc_type, exc, tb):
        if self.concise:
            if self.live is not None:
                self.live.stop()
            failures = [row for row in self.rows if row[1] == "FAIL"]
            _render_concise_check(self.console, self.system, self.resources, self.root, failures)
            return False
        if self.live is not None:
            self.live.stop()
            if exc_type is None:
                counts = {s: sum(row[1] == s for row in self.rows) for s in ("PASS", "FAIL", "WARN", "INFO", "SKIP", "NOT_RUN")}
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
            from .download import list_targets
            selected = [item for item in list_targets(Path(self.root)) if item["target"] in self.targets]
            if selected:
                self.console.print(_asset_table(selected, self.root, width=self.console.width))
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
            _inventory_plain(records, self.root, files=False)

    def __exit__(self, exc_type, exc, tb):
        if self.progress is not None:
            self.progress.stop()
            self.progress = None
        return False
