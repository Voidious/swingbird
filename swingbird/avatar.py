"""Builds the "Emoji" style avatar image for the daemon's own identity.

Buzz Desktop encodes an "Emoji" avatar as a `data:image/svg+xml,...` URI --
a colored rounded-rect background with the emoji centered as an SVG `<text>`
element (see `ProfileAvatarEditor.utils.ts::emojiAvatarDataUrl` in the
buzz-cli desktop app). Matching that template means `buzz users get` reports
back exactly what was set, and Buzz Desktop's own avatar editor recognizes
it as an emoji avatar it can re-edit, rather than an opaque uploaded image.
"""

from __future__ import annotations

from urllib.parse import quote

_DATA_URL_PREFIX = "data:image/svg+xml,"
_FONT_SIZE = 258


def _escape_svg_text(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def emoji_avatar_data_url(emoji: str, color: str) -> str:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" '
        'viewBox="0 0 512 512">'
        f'<rect width="512" height="512" rx="256" fill="{color}"/>'
        '<text x="50%" y="56%" dominant-baseline="middle" text-anchor="middle" '
        f'font-size="{_FONT_SIZE}">{_escape_svg_text(emoji)}</text></svg>'
    )
    return _DATA_URL_PREFIX + quote(svg, safe="")
