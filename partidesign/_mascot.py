"""Warm CLI accents and optional native display of shipped original images; no pixel art."""
from __future__ import annotations

import os

ACCENT = "yellow"
PROMPT_ACCENT = "#edc963"
IMAGE_WIDTH = 8
IMAGE_HEIGHT = 3
TERMINAL_HEIGHT = IMAGE_HEIGHT
IMAGE_FILES = {
    "standing": "standing.png",
    "thinking": "thinking.png",
    "peace": "peace.png",
    "shy": "shy.png",
    "horse": "horse.jpg",
    "pointing": "pointing.jpg",
}
IMAGE_NAMES = tuple(IMAGE_FILES)
_selected_image = None


def supports_unicode(stream) -> bool:
    if (not getattr(stream, "isatty", lambda: False)()
            or os.environ.get("TERM") == "dumb" or "NO_COLOR" in os.environ):
        return False
    try:
        "✨🧬".encode(getattr(stream, "encoding", None) or "utf-8")
    except (UnicodeError, LookupError):
        return False
    return True


def supports_image(stream, width: int) -> bool:
    return width >= 80 and supports_unicode(stream)


def image_name(*, root_help=False):
    if root_help:
        return "standing"
    global _selected_image
    if _selected_image is None:
        import secrets
        _selected_image = secrets.choice(IMAGE_NAMES)
    return _selected_image


def native_image(console, *, root_help=False):
    """Return PNG transport only for explicitly known image terminals."""
    if console.height < 6 or not supports_image(console.file, console.width):
        return None
    term = os.environ.get("TERM", "").lower()
    program = os.environ.get("TERM_PROGRAM", "").lower()
    if os.environ.get("TMUX") or os.environ.get("STY") or term.startswith(("screen", "tmux")):
        return None
    if (os.environ.get("KITTY_WINDOW_ID") or term in {"xterm-kitty", "xterm-ghostty"}
            or program in {"kitty", "ghostty"}):
        protocol = "kitty"
    elif program in {"iterm.app", "iterm2", "wezterm"}:
        protocol = "iterm"
    else:
        return None
    from importlib.resources import files
    try:
        filename = IMAGE_FILES[image_name(root_help=root_help)]
        data = files("partidesign").joinpath("assets", "milkyfrog", filename).read_bytes()
    except (OSError, ImportError):
        return None
    if len(data) < 24:
        return None
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        if not data.startswith(b"\xff\xd8\xff"):
            return None
        # Keep bundled JPEG originals; PNG is only the native transport format.
        try:
            from io import BytesIO
            from PIL import Image
            with Image.open(BytesIO(data)) as image:
                if image.format != "JPEG":
                    return None
                payload = BytesIO()
                image.convert("RGB").save(payload, format="PNG", exif=b"")
                data = payload.getvalue()
        except (ImportError, OSError, ValueError):
            return None
    return protocol, data


def image_sequences(protocol, data):
    """No queries/replies: inline image data is local and base64 only."""
    import base64
    payload = base64.b64encode(data).decode("ascii")
    if protocol == "kitty":
        chunks = [payload[i:i + 4096] for i in range(0, len(payload), 4096)]
        result = []
        for index, chunk in enumerate(chunks):
            more = int(index < len(chunks) - 1)
            controls = "a=T,f=100,t=d,c=8,r=3,C=1,q=2," if index == 0 else "q=2,"
            result.append("\x1b_G" + controls + "m=" + str(more) + ";" + chunk + "\x1b\\")
        return "".join(result)
    name = base64.b64encode(b"milkfrog.png").decode("ascii")
    return ("\x1b]1337;File=name=" + name + ";size=" + str(len(data))
            + ";width=8;height=3;preserveAspectRatio=1;inline=1;doNotMoveCursor=1:"
            + payload + "\x07")


def place_native_image(console, plan, panel_height):
    """Overlay immediately after the panel, restoring the caller's cursor."""
    protocol, data = plan
    # Border + left padding = two columns; account for wrapped panel content.
    top = 1 + max(0, (panel_height - 2 - TERMINAL_HEIGHT) // 2)
    up = panel_height - top
    stream = console.file
    saved = success = False
    try:
        stream.write("\x1b7")
        saved = True
        stream.write("\x1b[" + str(up) + "A\x1b[3G")
        stream.write(image_sequences(protocol, data))
        success = True
    except (OSError, UnicodeError):
        pass
    finally:
        if saved:
            try:
                stream.write("\x1b8")
                stream.flush()
            except (OSError, UnicodeError):
                success = False
    return success
