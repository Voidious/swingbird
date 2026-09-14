from swingbird import recap
from swingbird.config import Config, OwnerConfig, RecapConfig
from swingbird.recap import build_recap
from tests.test_recap_build_behavior import CONFIG, LLM_CONFIG, RELAY_CONFIG, _llm


def test_build_recap_strips_llm_narrated_omitted_note(monkeypatch):
    # Observed live: the LLM narrated "(Additional open items omitted.)" --
    # same countless shape as its "(Additional open items remain.)" cousin
    # (see test_recap.py), but naming the fold itself as an omission instead
    # of describing what's left over. The original regex only accepted a
    # trailing "remain(s/ing)?", so this phrasing sailed through unstripped
    # and stacked next to our own real note.
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**swingbird**: primary item text. (Additional open items omitted.)",
        items=[
            {
                "channel": "swingbird",
                "label": "F4",
                "summary": "primary",
                "instruction": "do the primary thing",
            },
            {
                "channel": "swingbird",
                "label": "F5",
                "summary": "secondary",
                "instruction": "do the secondary thing",
            },
        ],
    )

    result = build_recap(llm, CONFIG)

    assert result.text == (
        "**swingbird**: primary item text. (1 additional open item.)"
    )


def test_build_recap_drops_llm_narrated_count_sentence_in_detailed_mode(
    monkeypatch,
):
    # Observed live: right after our own correct "**backend -- 1 additional
    # open item:** F5" fold paragraph, the LLM narrated a whole extra
    # sentence-paragraph of its own -- "There are 7 more open items for
    # backend." -- with a count that didn't even agree with the real one.
    # `_LLM_COUNT_NOTE_RE` only strips a trailing "(...)" from an existing
    # paragraph, not a freestanding sentence, so this shape sailed through
    # unstripped and just sat there, redundant with (and contradicting) our
    # grounded note.
    config = Config(
        llm=LLM_CONFIG,
        relay=RELAY_CONFIG,
        channels=CONFIG.channels,
        owner=OwnerConfig(pubkey="owner-pubkey", name="Voidious"),
        recap=RecapConfig(max_detailed_items=1),
    )
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**backend -- F4:** prose covering the first item.\n\n"
        "There are 7 more open items for backend.",
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "primary",
                "instruction": "do the primary thing",
            },
            {
                "channel": "backend",
                "label": "F5",
                "summary": "secondary",
                "instruction": "do the secondary thing",
            },
        ],
    )

    result = build_recap(llm, config, detail="detailed")

    assert result.text == (
        "**backend -- F4:** prose covering the first item.\n\n"
        "**backend -- 1 additional open item:** F5"
    )


def test_build_recap_strips_llm_narrated_count_sentence_trailing_real_content(
    monkeypatch,
):
    # Observed live in concise mode: the LLM tacked the freestanding-sentence
    # shape onto the *end* of the primary item's own paragraph, right before
    # our own correct parenthetical note -- "...it. There are 10 additional
    # open items. (9 additional open items.)" -- rather than as a paragraph
    # of its own. `_LLM_COUNT_SENTENCE_RE` previously only matched via
    # `fullmatch` against a whole paragraph, so this mid-paragraph case
    # sailed through unstripped, leaving both the bogus and the real count
    # stacked at the end of the same paragraph.
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**swingbird**: primary item text. There are 10 additional open items.",
        items=[
            {
                "channel": "swingbird",
                "label": "F4",
                "summary": "primary",
                "instruction": "do the primary thing",
            },
            {
                "channel": "swingbird",
                "label": "F5",
                "summary": "secondary",
                "instruction": "do the secondary thing",
            },
        ],
    )

    result = build_recap(llm, CONFIG)

    assert result.text == "**swingbird**: primary item text. (1 additional open item.)"


def test_build_recap_drops_llm_narrated_count_sentence_naming_the_channel(
    monkeypatch,
):
    # Same freestanding-sentence shape as the test above, but naming the
    # channel between "open" and "items" ("open backend items") instead of
    # after it, and with a trailing "beyond the N described above" clause --
    # observed live as "There are 8 additional open swingbird items beyond
    # the three described above." Still just noise once our own grounded
    # note is in place, whether or not its count happens to agree with ours.
    config = Config(
        llm=LLM_CONFIG,
        relay=RELAY_CONFIG,
        channels=CONFIG.channels,
        owner=OwnerConfig(pubkey="owner-pubkey", name="Voidious"),
        recap=RecapConfig(max_detailed_items=1),
    )
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**backend -- F4:** prose covering the first item.\n\n"
        "There are 1 additional open backend items beyond the one described "
        "above.",
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "primary",
                "instruction": "do the primary thing",
            },
            {
                "channel": "backend",
                "label": "F5",
                "summary": "secondary",
                "instruction": "do the secondary thing",
            },
        ],
    )

    result = build_recap(llm, config, detail="detailed")

    assert result.text == (
        "**backend -- F4:** prose covering the first item.\n\n"
        "**backend -- 1 additional open item:** F5"
    )


def test_build_recap_drops_llm_narrated_bare_prefix_count_sentence(monkeypatch):
    # Observed live: right after our own correct "**backend -- 1 additional
    # open item:** F5" fold paragraph, the LLM separately narrated a bare
    # "**backend:** 1 additional open item is listed above." paragraph of
    # its own -- no "there is/are" or "<channel> has/have" subject at all
    # this time, just the count stated directly after a bare channel prefix
    # (colon inside the bold, unlike the format guard's own "**channel**:").
    # `_LLM_COUNT_SENTENCE_RE` never matched this since it requires a subject
    # before the count; `_BARE_CHANNEL_PREFIX_RE`/`_COUNT_ONLY_TEXT_RE` catch
    # it directly instead.
    config = Config(
        llm=LLM_CONFIG,
        relay=RELAY_CONFIG,
        channels=CONFIG.channels,
        owner=OwnerConfig(pubkey="owner-pubkey", name="Voidious"),
        recap=RecapConfig(max_detailed_items=1),
    )
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**backend -- F4:** prose covering the first item.\n\n"
        "**backend:** 1 additional open item is listed above.",
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "primary",
                "instruction": "do the primary thing",
            },
            {
                "channel": "backend",
                "label": "F5",
                "summary": "secondary",
                "instruction": "do the secondary thing",
            },
        ],
    )

    result = build_recap(llm, config, detail="detailed")

    assert result.text == (
        "**backend -- F4:** prose covering the first item.\n\n"
        "**backend -- 1 additional open item:** F5"
    )


def test_build_recap_drops_llm_narrated_has_sentence_after_real_fold_paragraph(
    monkeypatch,
):
    # Observed live: right after our own correct "**backend -- 1 additional
    # open item:** F5" fold paragraph (the real one `_append_item_counts`
    # itself generates -- not an LLM invention, even though it happens to
    # share that paragraph's exact "**channel -- N additional open items:**"
    # shape), the LLM separately narrated its own leftover "has"-phrased
    # sentence -- "backend has 1 additional open item beyond these one." --
    # which `_LLM_COUNT_SENTENCE_RE` didn't catch yet, since it only matched
    # a "there is/are" subject, not "<channel> has/have".
    config = Config(
        llm=LLM_CONFIG,
        relay=RELAY_CONFIG,
        channels=CONFIG.channels,
        owner=OwnerConfig(pubkey="owner-pubkey", name="Voidious"),
        recap=RecapConfig(max_detailed_items=1),
    )
    monkeypatch.setattr(recap, "fetch_messages_since", lambda *a, **k: [])
    llm, _ = _llm(
        "**backend -- F4:** prose covering the first item.\n\n"
        "backend has 1 additional open item beyond this one.",
        items=[
            {
                "channel": "backend",
                "label": "F4",
                "summary": "primary",
                "instruction": "do the primary thing",
            },
            {
                "channel": "backend",
                "label": "F5",
                "summary": "secondary",
                "instruction": "do the secondary thing",
            },
        ],
    )

    result = build_recap(llm, config, detail="detailed")

    assert result.text == (
        "**backend -- F4:** prose covering the first item.\n\n"
        "**backend -- 1 additional open item:** F5"
    )
