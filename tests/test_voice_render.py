from swingbird.voice_render import render_for_speech


def test_strips_link_after_a_label():
    text = "Confirmed and relayed: buzz://message?channel=abc123&id=def456"
    assert render_for_speech(text) == "Confirmed and relayed."


def test_strips_link_on_its_own_trailing_paragraph():
    text = "The agent said the fix is done.\n\nbuzz://message?channel=abc&id=123"
    assert render_for_speech(text) == "The agent said the fix is done."


def test_strips_a_link_with_no_recognized_surrounding_shape():
    text = "See buzz://message?channel=abc&id=123 for details."
    assert render_for_speech(text) == "See  for details."


def test_unquotes_a_single_quoted_proposal_instruction():
    text = "About to relay to dripbird: 'run the tests'. Confirm to send, or cancel."
    result = render_for_speech(text)
    assert result == (
        "About to relay to dripbird: run the tests. Confirm to send, or cancel."
    )


def test_unquotes_a_double_quoted_proposal_instruction():
    # repr() switches to double quotes when the string contains a single
    # quote but no double quote.
    text = (
        'About to relay to dripbird: "don\'t skip the tests". '
        "Confirm to send, or cancel."
    )
    result = render_for_speech(text)
    assert result == (
        "About to relay to dripbird: don't skip the tests. Confirm to send, or cancel."
    )


def test_unquotes_an_instruction_with_escaped_quotes_and_newlines():
    text = (
        "About to relay to dripbird (for backend): "
        "'first line\\nsecond \\'quoted\\' line'. Confirm to send, or cancel."
    )
    result = render_for_speech(text)
    assert result == (
        "About to relay to dripbird (for backend): first line\n"
        "second 'quoted' line. Confirm to send, or cancel."
    )


def test_leaves_text_with_no_proposal_shape_unchanged():
    text = "Just a plain reply with no quoting."
    assert render_for_speech(text) == text


def test_leaves_a_lookalike_colon_quote_tail_unchanged_if_not_valid_python_literal():
    text = "Note: 'unterminated. Confirm to send, or cancel."
    assert render_for_speech(text) == text


def test_strips_at_mentions_to_plain_names():
    text = "@Sonnet Relaying instruction from @Voidious: go ahead."
    expected = "Sonnet Relaying instruction from Voidious: go ahead."
    assert render_for_speech(text) == expected


def test_combines_link_stripping_and_mention_stripping():
    text = "Relayed to @backend -- buzz://message?channel=abc&id=123"
    assert render_for_speech(text) == "Relayed to backend --"


def test_strips_bold_channel_prefix():
    text = "**backend**: ship the release."
    assert render_for_speech(text) == "backend: ship the release."


def test_strips_bold_channel_and_label_prefix_with_colon_inside():
    text = "**swingbird -- F4:** wire up the follow-up window."
    assert render_for_speech(text) == "swingbird -- F4: wire up the follow-up window."


def test_strips_inline_code_spans():
    text = "Run `uv sync` before `uv run pytest`."
    assert render_for_speech(text) == "Run uv sync before uv run pytest."


def test_strips_markdown_headers():
    text = "# Recap\n\n### backend\nAll clear."
    assert render_for_speech(text) == "Recap\n\nbackend\nAll clear."


def test_leaves_bare_asterisks_and_hashes_unchanged():
    text = "5 * 3 = 15, and #4 is done."
    assert render_for_speech(text) == text
