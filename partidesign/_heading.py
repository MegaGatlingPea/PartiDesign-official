"""Optional, size-enhanced section labels for compatible Kitty terminals."""
from __future__ import annotations

import os

# Only labels owned by the CLI may become an escape-sequence payload.
_TITLES = frozenset({
    "Commands", "Options", "Positional arguments", "Try next",
    "Machine overview", "PartiDesign models", "Sequence design", "Structure prediction",
    "Chemistry", "Other resources",
})


def _kitty_console(console) -> bool:
    from ._mascot import supports_unicode

    if console.width < 80 or console.height < 6 or not supports_unicode(console.file):
        return False
    term = os.environ.get("TERM", "").lower()
    program = os.environ.get("TERM_PROGRAM", "").lower()
    if os.environ.get("TMUX") or os.environ.get("STY") or term.startswith(("screen", "tmux")):
        return False
    return bool(os.environ.get("KITTY_WINDOW_ID")
                or term == "xterm-kitty" or program == "kitty")


def print_section_heading(console, heading, title: str, *, prefix_width: int | None = None) -> bool:
    """Print a left-aligned heading, with a modest optional size overlay.

    Return False without output when the caller should print its usual label.
    True means the ordinary label has been printed, even if the optional
    overlay could not be written. An older terminal which ignores OSC 66 keeps
    that ordinary label. Two rows are reserved before any cursor movement;
    emoji remain in the unchanged, default-color ordinary label.
    """
    if not isinstance(title, str) or title not in _TITLES or not _kitty_console(console):
        return False
    plain = getattr(heading, "plain", None)
    if not isinstance(plain, str) or not plain.endswith(title):
        return False
    prefix = plain[:-len(title)]
    if any(ord(character) < 32 or ord(character) == 127 for character in prefix):
        return False
    from rich.cells import cell_len

    measured_prefix = cell_len(prefix)
    if prefix_width is None:
        prefix_width = measured_prefix
    if (not isinstance(prefix_width, int) or isinstance(prefix_width, bool)
            or prefix_width != measured_prefix):
        return False
    chunks = tuple(title[index:index + 10] for index in range(0, len(title), 10))
    # s=2 and n/d=3/5 gives 1.2x glyphs inside a two-row block. Each
    # chunk gets enough columns without changing the terminal's global font.
    widths = tuple(max(1, (len(chunk) * 3 + 4) // 5) for chunk in chunks)
    if (any(width > 7 for width in widths)
            or prefix_width + 2 * sum(widths) >= console.width
            or cell_len(plain) >= console.width):
        return False

    console.print(heading, justify="left")
    console.print()
    stream = console.file
    saved = False
    try:
        stream.write("\x1b7")
        saved = True
        stream.write("\x1b[2A\x1b[" + str(prefix_width + 1) + "G\x1b[1;33m")
        for chunk, width in zip(chunks, widths, strict=True):
            stream.write("\x1b]66;s=2:n=3:d=5:w=" + str(width) + ";" + chunk + "\x1b\\")
    except (OSError, UnicodeError):
        # The readable heading is already present; decoration cannot remove it.
        pass
    finally:
        if saved:
            try:
                stream.write("\x1b[0m\x1b8")
                stream.flush()
            except (OSError, UnicodeError):
                pass
    return True
