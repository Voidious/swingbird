from urllib.parse import unquote

from swingbird.avatar import emoji_avatar_data_url


def test_emoji_avatar_data_url_embeds_color_and_emoji():
    url = emoji_avatar_data_url("🐦", "#3399FF")

    assert url.startswith("data:image/svg+xml,")
    svg = unquote(url[len("data:image/svg+xml,") :])
    assert '<rect width="512" height="512" rx="256" fill="#3399FF"/>' in svg
    assert "🐦</text>" in svg


def test_emoji_avatar_data_url_escapes_svg_special_characters():
    url = emoji_avatar_data_url("<&>", "#000000")

    svg = unquote(url[len("data:image/svg+xml,") :])
    assert "&lt;&amp;&gt;</text>" in svg
