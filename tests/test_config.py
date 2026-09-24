import pytest

from swingbird.config import ConfigError, load_config

VALID = """
[llm]
base_url = "https://api.moonshot.ai/v1"
model = "kimi-k2.6"
api_key_env = "MOONSHOT_API_KEY"

[relay]
url = "wss://relay.example.com"
private_key_env = "SWINGBIRD_PRIVATE_KEY"

[owner]
pubkey = "owner-pubkey"
name = "Voidious"

[[channels]]
id = "chan-1"
name = "swingbird-dev"
write = true
agents = ["Sonnet"]
"""


def write(tmp_path, text):
    path = tmp_path / "swingbird.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_valid_config(tmp_path):
    config = load_config(write(tmp_path, VALID))

    assert config.llm.base_url == "https://api.moonshot.ai/v1"
    assert config.llm.model == "kimi-k2.6"
    assert config.llm.api_key_env == "MOONSHOT_API_KEY"
    assert config.relay.url == "wss://relay.example.com"
    assert config.relay.private_key_env == "SWINGBIRD_PRIVATE_KEY"
    assert config.owner.pubkey == "owner-pubkey"
    assert config.owner.name == "Voidious"
    assert len(config.channels) == 1
    channel = config.channels[0]
    assert channel.id == "chan-1"
    assert channel.name == "swingbird-dev"
    assert channel.write is True
    assert channel.agents == ("Sonnet",)
    assert config.dispatch.reply_wait_seconds == 180
    assert config.identity.name == "swingbird"
    assert config.recap.stale_after_days == 30
    assert channel.goal is None
    assert config.llm.provider is None


def test_channel_by_name_found_and_missing(tmp_path):
    config = load_config(write(tmp_path, VALID))

    assert config.channel_by_name("swingbird-dev") is not None
    assert config.channel_by_name("does-not-exist") is None


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(tmp_path / "nope.toml")


def test_invalid_toml(tmp_path):
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(write(tmp_path, "not valid ][ toml"))


def test_missing_llm_section(tmp_path):
    text = """
[relay]
url = "wss://relay.example.com"
private_key_env = "SWINGBIRD_PRIVATE_KEY"

[[channels]]
id = "chan-1"
name = "swingbird-dev"
write = true
agents = []
"""
    with pytest.raises(ConfigError, match=r"\[llm\] section"):
        load_config(write(tmp_path, text))


def test_llm_missing_required_field(tmp_path):
    text = """
[llm]
base_url = "https://api.moonshot.ai/v1"
model = "kimi-k2.6"

[relay]
url = "wss://relay.example.com"
private_key_env = "SWINGBIRD_PRIVATE_KEY"

[[channels]]
id = "chan-1"
name = "swingbird-dev"
write = true
agents = []
"""
    with pytest.raises(ConfigError, match="api_key_env"):
        load_config(write(tmp_path, text))


def test_llm_missing_model(tmp_path):
    text = """
[llm]
base_url = "https://api.moonshot.ai/v1"
api_key_env = "MOONSHOT_API_KEY"

[relay]
url = "wss://relay.example.com"
private_key_env = "SWINGBIRD_PRIVATE_KEY"

[[channels]]
id = "chan-1"
name = "swingbird-dev"
write = true
agents = []
"""
    with pytest.raises(ConfigError, match=r"\[llm\] is missing required field: model"):
        load_config(write(tmp_path, text))


def test_llm_missing_base_url_without_provider(tmp_path):
    text = """
[llm]
model = "some-model"
api_key_env = "MOONSHOT_API_KEY"

[relay]
url = "wss://relay.example.com"
private_key_env = "SWINGBIRD_PRIVATE_KEY"

[[channels]]
id = "chan-1"
name = "swingbird-dev"
write = true
agents = []
"""
    with pytest.raises(
        ConfigError, match=r"\[llm\] is missing required field: base_url"
    ):
        load_config(write(tmp_path, text))


@pytest.mark.parametrize(
    "provider,base_url,api_key_env",
    [
        ("moonshot", "https://api.moonshot.ai/v1", "MOONSHOT_API_KEY"),
        ("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    ],
)
def test_llm_provider_fills_in_defaults(tmp_path, provider, base_url, api_key_env):
    text = f"""
[llm]
provider = "{provider}"
model = "some-model"

[relay]
url = "wss://relay.example.com"
private_key_env = "SWINGBIRD_PRIVATE_KEY"

[owner]
pubkey = "owner-pubkey"
name = "Voidious"

[[channels]]
id = "chan-1"
name = "swingbird-dev"
write = true
agents = []
"""
    config = load_config(write(tmp_path, text))

    assert config.llm.provider == provider
    assert config.llm.base_url == base_url
    assert config.llm.api_key_env == api_key_env


def test_llm_explicit_base_url_and_api_key_env_override_provider(tmp_path):
    text = """
[llm]
provider = "openai"
model = "some-model"
base_url = "https://my-proxy.example.com/v1"
api_key_env = "MY_OPENAI_KEY"

[relay]
url = "wss://relay.example.com"
private_key_env = "SWINGBIRD_PRIVATE_KEY"

[owner]
pubkey = "owner-pubkey"
name = "Voidious"

[[channels]]
id = "chan-1"
name = "swingbird-dev"
write = true
agents = []
"""
    config = load_config(write(tmp_path, text))

    assert config.llm.base_url == "https://my-proxy.example.com/v1"
    assert config.llm.api_key_env == "MY_OPENAI_KEY"


def test_llm_rejects_unknown_provider(tmp_path):
    text = """
[llm]
provider = "bogus"
model = "some-model"

[relay]
url = "wss://relay.example.com"
private_key_env = "SWINGBIRD_PRIVATE_KEY"

[[channels]]
id = "chan-1"
name = "swingbird-dev"
write = true
agents = []
"""
    with pytest.raises(ConfigError, match=r"\[llm\].provider must be one of"):
        load_config(write(tmp_path, text))


def test_missing_relay_section(tmp_path):
    text = """
[llm]
base_url = "https://api.moonshot.ai/v1"
model = "kimi-k2.6"
api_key_env = "MOONSHOT_API_KEY"

[[channels]]
id = "chan-1"
name = "swingbird-dev"
write = true
agents = []
"""
    with pytest.raises(ConfigError, match=r"\[relay\] section"):
        load_config(write(tmp_path, text))


def test_relay_missing_required_field(tmp_path):
    text = """
[llm]
base_url = "https://api.moonshot.ai/v1"
model = "kimi-k2.6"
api_key_env = "MOONSHOT_API_KEY"

[relay]
url = "wss://relay.example.com"

[[channels]]
id = "chan-1"
name = "swingbird-dev"
write = true
agents = []
"""
    with pytest.raises(ConfigError, match="private_key_env"):
        load_config(write(tmp_path, text))


def test_missing_owner_section(tmp_path):
    text = """
[llm]
base_url = "https://api.moonshot.ai/v1"
model = "kimi-k2.6"
api_key_env = "MOONSHOT_API_KEY"

[relay]
url = "wss://relay.example.com"
private_key_env = "SWINGBIRD_PRIVATE_KEY"

[[channels]]
id = "chan-1"
name = "swingbird-dev"
write = true
agents = []
"""
    with pytest.raises(ConfigError, match=r"\[owner\] section"):
        load_config(write(tmp_path, text))


def test_owner_missing_required_field(tmp_path):
    text = """
[llm]
base_url = "https://api.moonshot.ai/v1"
model = "kimi-k2.6"
api_key_env = "MOONSHOT_API_KEY"

[relay]
url = "wss://relay.example.com"
private_key_env = "SWINGBIRD_PRIVATE_KEY"

[owner]
pubkey = "owner-pubkey"

[[channels]]
id = "chan-1"
name = "swingbird-dev"
write = true
agents = []
"""
    with pytest.raises(ConfigError, match="name"):
        load_config(write(tmp_path, text))


def test_channels_missing(tmp_path):
    text = """
[llm]
base_url = "https://api.moonshot.ai/v1"
model = "kimi-k2.6"
api_key_env = "MOONSHOT_API_KEY"

[relay]
url = "wss://relay.example.com"
private_key_env = "SWINGBIRD_PRIVATE_KEY"
"""
    with pytest.raises(ConfigError, match=r"\[\[channels\]\]"):
        load_config(write(tmp_path, text))


def test_channels_empty(tmp_path):
    text = """
[llm]
base_url = "https://api.moonshot.ai/v1"
model = "kimi-k2.6"
api_key_env = "MOONSHOT_API_KEY"

[relay]
url = "wss://relay.example.com"
private_key_env = "SWINGBIRD_PRIVATE_KEY"

channels = []
"""
    with pytest.raises(ConfigError, match=r"\[\[channels\]\]"):
        load_config(write(tmp_path, text))


def test_channel_missing_required_field(tmp_path):
    text = """
[llm]
base_url = "https://api.moonshot.ai/v1"
model = "kimi-k2.6"
api_key_env = "MOONSHOT_API_KEY"

[relay]
url = "wss://relay.example.com"
private_key_env = "SWINGBIRD_PRIVATE_KEY"

[[channels]]
id = "chan-1"
name = "swingbird-dev"
write = true
"""
    with pytest.raises(ConfigError, match="agents"):
        load_config(write(tmp_path, text))


def test_duplicate_channel_id(tmp_path):
    text = """
[llm]
base_url = "https://api.moonshot.ai/v1"
model = "kimi-k2.6"
api_key_env = "MOONSHOT_API_KEY"

[relay]
url = "wss://relay.example.com"
private_key_env = "SWINGBIRD_PRIVATE_KEY"

[[channels]]
id = "chan-1"
name = "swingbird-dev"
write = true
agents = []

[[channels]]
id = "chan-1"
name = "other-project"
write = false
agents = []
"""
    with pytest.raises(ConfigError, match="duplicate channel id"):
        load_config(write(tmp_path, text))


def test_local_override_merges_onto_base(tmp_path):
    path = write(tmp_path, VALID)
    (tmp_path / ".swingbird.toml").write_text(
        '[relay]\nurl = "wss://private-relay.example"\n'
    )

    config = load_config(path)

    assert config.relay.url == "wss://private-relay.example"
    assert config.relay.private_key_env == "SWINGBIRD_PRIVATE_KEY"
    assert config.llm.model == "kimi-k2.6"


def test_no_local_override_leaves_base_unchanged(tmp_path):
    config = load_config(write(tmp_path, VALID))

    assert config.relay.url == "wss://relay.example.com"


def test_local_override_invalid_toml(tmp_path):
    path = write(tmp_path, VALID)
    (tmp_path / ".swingbird.toml").write_text("not valid ][ toml")

    with pytest.raises(ConfigError, match=r"invalid TOML in.*\.swingbird\.toml"):
        load_config(path)


def test_dispatch_reply_wait_seconds_is_configurable(tmp_path):
    config = load_config(
        write(tmp_path, VALID + "\n[dispatch]\nreply_wait_seconds = 5\n")
    )

    assert config.dispatch.reply_wait_seconds == 5


def test_dispatch_section_not_a_table(tmp_path):
    # A bare `key = value` assignment belongs to whichever table header
    # precedes it in TOML, so `dispatch = 5` only lands as a genuine
    # top-level key if it comes before every other section.
    with pytest.raises(ConfigError, match=r"\[dispatch\] must be a table"):
        load_config(write(tmp_path, "dispatch = 5\n\n" + VALID))


@pytest.mark.parametrize("value", ["0", "-1", "true", '"90"'])
def test_dispatch_reply_wait_seconds_rejects_invalid_values(tmp_path, value):
    text = VALID + f"\n[dispatch]\nreply_wait_seconds = {value}\n"
    with pytest.raises(
        ConfigError, match="reply_wait_seconds must be a positive integer"
    ):
        load_config(write(tmp_path, text))


def test_identity_name_is_configurable(tmp_path):
    config = load_config(write(tmp_path, VALID + '\n[identity]\nname = "custom-bot"\n'))

    assert config.identity.name == "custom-bot"


def test_identity_section_not_a_table(tmp_path):
    with pytest.raises(ConfigError, match=r"\[identity\] must be a table"):
        load_config(write(tmp_path, "identity = 5\n\n" + VALID))


@pytest.mark.parametrize("value", ['""', '"   "', "5", "true"])
def test_identity_name_rejects_invalid_values(tmp_path, value):
    text = VALID + f"\n[identity]\nname = {value}\n"
    with pytest.raises(
        ConfigError, match=r"\[identity\].name must be a non-empty string"
    ):
        load_config(write(tmp_path, text))


def test_identity_description_defaults_to_none(tmp_path):
    config = load_config(write(tmp_path, VALID))

    assert config.identity.description is None


def test_identity_description_is_configurable(tmp_path):
    text = VALID + '\n[identity]\nname = "swingbird"\ndescription = "TPM bot"\n'
    config = load_config(write(tmp_path, text))

    assert config.identity.description == "TPM bot"


@pytest.mark.parametrize("value", ['""', '"   "', "5", "true"])
def test_identity_description_rejects_invalid_values(tmp_path, value):
    text = VALID + f"\n[identity]\ndescription = {value}\n"
    with pytest.raises(
        ConfigError, match=r"\[identity\].description must be a non-empty string"
    ):
        load_config(write(tmp_path, text))


def test_identity_avatar_defaults_to_none(tmp_path):
    config = load_config(write(tmp_path, VALID))

    assert config.identity.avatar is None


def test_identity_avatar_emoji_style_is_configurable(tmp_path):
    text = (
        VALID
        + '\n[identity.avatar]\nstyle = "emoji"\nemoji = "🐦"\ncolor = "#3399FF"\n'
    )
    config = load_config(write(tmp_path, text))

    assert config.identity.avatar.style == "emoji"
    assert config.identity.avatar.emoji == "🐦"
    assert config.identity.avatar.color == "#3399FF"


def test_identity_avatar_not_a_table(tmp_path):
    text = VALID + "\n[identity]\navatar = 5\n"
    with pytest.raises(ConfigError, match=r"\[identity.avatar\] must be a table"):
        load_config(write(tmp_path, text))


@pytest.mark.parametrize("style", ['"image"', '"animated"', '"bogus"'])
def test_identity_avatar_rejects_unsupported_styles(tmp_path, style):
    text = (
        VALID
        + f'\n[identity.avatar]\nstyle = {style}\nemoji = "🐦"\ncolor = "#3399FF"\n'
    )
    with pytest.raises(ConfigError, match=r"\[identity.avatar\].style must be one of"):
        load_config(write(tmp_path, text))


@pytest.mark.parametrize("value", ['""', '"   "', "5", "true"])
def test_identity_avatar_rejects_invalid_emoji(tmp_path, value):
    text = (
        VALID
        + f'\n[identity.avatar]\nstyle = "emoji"\nemoji = {value}\ncolor = "#3399FF"\n'
    )
    with pytest.raises(
        ConfigError, match=r"\[identity.avatar\].emoji must be a non-empty string"
    ):
        load_config(write(tmp_path, text))


@pytest.mark.parametrize("value", ['""', '"3399FF"', '"#39F"', "5", "true"])
def test_identity_avatar_rejects_invalid_color(tmp_path, value):
    text = (
        VALID + f'\n[identity.avatar]\nstyle = "emoji"\nemoji = "🐦"\ncolor = {value}\n'
    )
    with pytest.raises(
        ConfigError, match=r"\[identity.avatar\].color must be a hex color"
    ):
        load_config(write(tmp_path, text))


def test_duplicate_channel_name(tmp_path):
    text = """
[llm]
base_url = "https://api.moonshot.ai/v1"
model = "kimi-k2.6"
api_key_env = "MOONSHOT_API_KEY"

[relay]
url = "wss://relay.example.com"
private_key_env = "SWINGBIRD_PRIVATE_KEY"

[[channels]]
id = "chan-1"
name = "swingbird-dev"
write = true
agents = []

[[channels]]
id = "chan-2"
name = "swingbird-dev"
write = false
agents = []
"""
    with pytest.raises(ConfigError, match="duplicate channel name"):
        load_config(write(tmp_path, text))


def test_channel_goal_is_configurable(tmp_path):
    text = VALID.replace(
        'agents = ["Sonnet"]',
        'agents = ["Sonnet"]\ngoal = "Preparing the 0.8.0 release"',
    )
    config = load_config(write(tmp_path, text))

    assert config.channels[0].goal == "Preparing the 0.8.0 release"


def test_channel_goal_defaults_to_none(tmp_path):
    config = load_config(write(tmp_path, VALID))

    assert config.channels[0].goal is None


@pytest.mark.parametrize("value", ['""', '"   "', "5", "true"])
def test_channel_goal_rejects_invalid_values(tmp_path, value):
    text = VALID.replace('agents = ["Sonnet"]', f'agents = ["Sonnet"]\ngoal = {value}')
    with pytest.raises(ConfigError, match="invalid goal"):
        load_config(write(tmp_path, text))


def test_recap_stale_after_days_is_configurable(tmp_path):
    config = load_config(write(tmp_path, VALID + "\n[recap]\nstale_after_days = 60\n"))

    assert config.recap.stale_after_days == 60


def test_recap_section_not_a_table(tmp_path):
    with pytest.raises(ConfigError, match=r"\[recap\] must be a table"):
        load_config(write(tmp_path, "recap = 5\n\n" + VALID))


@pytest.mark.parametrize("value", ["0", "-1", "true", '"30"'])
def test_recap_stale_after_days_rejects_invalid_values(tmp_path, value):
    text = VALID + f"\n[recap]\nstale_after_days = {value}\n"
    with pytest.raises(
        ConfigError, match="stale_after_days must be a positive integer"
    ):
        load_config(write(tmp_path, text))


def test_recap_max_messages_per_channel_defaults(tmp_path):
    config = load_config(write(tmp_path, VALID))

    assert config.recap.max_messages_per_channel == 1000


def test_recap_max_messages_per_channel_is_configurable(tmp_path):
    config = load_config(
        write(tmp_path, VALID + "\n[recap]\nmax_messages_per_channel = 250\n")
    )

    assert config.recap.max_messages_per_channel == 250


@pytest.mark.parametrize("value", ["0", "-1", "true", '"250"'])
def test_recap_max_messages_per_channel_rejects_invalid_values(tmp_path, value):
    text = VALID + f"\n[recap]\nmax_messages_per_channel = {value}\n"
    with pytest.raises(
        ConfigError, match="max_messages_per_channel must be a positive integer"
    ):
        load_config(write(tmp_path, text))


def test_recap_max_detailed_items_defaults(tmp_path):
    config = load_config(write(tmp_path, VALID))

    assert config.recap.max_detailed_items == 3


def test_recap_max_detailed_items_is_configurable(tmp_path):
    config = load_config(write(tmp_path, VALID + "\n[recap]\nmax_detailed_items = 6\n"))

    assert config.recap.max_detailed_items == 6


@pytest.mark.parametrize("value", ["0", "-1", "true", '"6"'])
def test_recap_max_detailed_items_rejects_invalid_values(tmp_path, value):
    text = VALID + f"\n[recap]\nmax_detailed_items = {value}\n"
    with pytest.raises(
        ConfigError, match="max_detailed_items must be a positive integer"
    ):
        load_config(write(tmp_path, text))


def test_recap_closed_item_window_days_defaults(tmp_path):
    config = load_config(write(tmp_path, VALID))

    assert config.recap.closed_item_window_days == 90


def test_recap_closed_item_window_days_is_configurable(tmp_path):
    config = load_config(
        write(tmp_path, VALID + "\n[recap]\nclosed_item_window_days = 60\n")
    )

    assert config.recap.closed_item_window_days == 60


@pytest.mark.parametrize("value", ["0", "-1", "true", '"60"'])
def test_recap_closed_item_window_days_rejects_invalid_values(tmp_path, value):
    text = VALID + f"\n[recap]\nclosed_item_window_days = {value}\n"
    with pytest.raises(
        ConfigError, match="closed_item_window_days must be a positive integer"
    ):
        load_config(write(tmp_path, text))


def test_voice_defaults_to_disabled(tmp_path):
    config = load_config(write(tmp_path, VALID))

    assert config.voice.enabled is False
    assert config.voice.wake_word == "swingbird"
    assert config.voice.mic.type == "onboard"
    assert config.voice.output.type == "onboard"
    assert config.voice.stt.model == "small"
    assert config.voice.tts is None
    assert config.voice.wake_word_window_seconds == 30
    assert config.voice.follow_up_window_seconds == 30
    assert config.voice.barge_in_trigger_frames == 3
    assert config.voice.barge_in_vad_threshold == 0.8


def test_voice_section_not_a_table(tmp_path):
    with pytest.raises(ConfigError, match=r"\[voice\] must be a table"):
        load_config(write(tmp_path, "voice = 5\n\n" + VALID))


def test_voice_enabled_rejects_non_boolean(tmp_path):
    text = VALID + '\n[voice]\nenabled = "yes"\n'
    with pytest.raises(ConfigError, match=r"\[voice\].enabled must be a boolean"):
        load_config(write(tmp_path, text))


def test_voice_enabled_with_tts_configures_fully(tmp_path):
    text = (
        VALID
        + '\n[voice]\nenabled = true\nwake_word = "birdie"\n'
        + '\n[voice.mic]\ntype = "usb"\n'
        + '\n[voice.output]\ntype = "usb"\n'
        + '\n[voice.stt]\nmodel = "small.en"\n'
        + '\n[voice.tts]\nvoice = "en_US-lessac-medium"\n'
    )
    config = load_config(write(tmp_path, text))

    assert config.voice.enabled is True
    assert config.voice.wake_word == "birdie"
    assert config.voice.mic.type == "usb"
    assert config.voice.output.type == "usb"
    assert config.voice.stt.model == "small.en"
    assert config.voice.tts.voice == "en_US-lessac-medium"


@pytest.mark.parametrize("value", ['""', '"   "', "5", "true"])
def test_voice_wake_word_rejects_invalid_values(tmp_path, value):
    text = VALID + f"\n[voice]\nwake_word = {value}\n"
    with pytest.raises(
        ConfigError, match=r"\[voice\].wake_word must be a non-empty string"
    ):
        load_config(write(tmp_path, text))


def test_voice_mic_section_not_a_table(tmp_path):
    text = VALID + "\n[voice]\nmic = 5\n"
    with pytest.raises(ConfigError, match=r"\[voice.mic\] must be a table"):
        load_config(write(tmp_path, text))


def test_voice_mic_rejects_unsupported_type(tmp_path):
    text = VALID + '\n[voice.mic]\ntype = "bluetooth"\n'
    with pytest.raises(ConfigError, match=r"\[voice.mic\].type must be one of"):
        load_config(write(tmp_path, text))


def test_voice_output_section_not_a_table(tmp_path):
    text = VALID + "\n[voice]\noutput = 5\n"
    with pytest.raises(ConfigError, match=r"\[voice.output\] must be a table"):
        load_config(write(tmp_path, text))


def test_voice_output_rejects_unsupported_type(tmp_path):
    text = VALID + '\n[voice.output]\ntype = "bluetooth"\n'
    with pytest.raises(ConfigError, match=r"\[voice.output\].type must be one of"):
        load_config(write(tmp_path, text))


def test_voice_stt_section_not_a_table(tmp_path):
    text = VALID + "\n[voice]\nstt = 5\n"
    with pytest.raises(ConfigError, match=r"\[voice.stt\] must be a table"):
        load_config(write(tmp_path, text))


def test_voice_stt_rejects_unsupported_model(tmp_path):
    text = VALID + '\n[voice.stt]\nmodel = "medium"\n'
    with pytest.raises(ConfigError, match=r"\[voice.stt\].model must be one of"):
        load_config(write(tmp_path, text))


def test_voice_tts_required_when_enabled(tmp_path):
    text = VALID + "\n[voice]\nenabled = true\n"
    with pytest.raises(
        ConfigError, match=r"\[voice.tts\] is required when \[voice\].enabled is true"
    ):
        load_config(write(tmp_path, text))


def test_voice_tts_not_required_when_disabled(tmp_path):
    config = load_config(write(tmp_path, VALID))

    assert config.voice.tts is None


def test_voice_tts_not_a_table(tmp_path):
    text = VALID + "\n[voice]\ntts = 5\n"
    with pytest.raises(ConfigError, match=r"\[voice.tts\] must be a table"):
        load_config(write(tmp_path, text))


@pytest.mark.parametrize("value", ['""', '"   "', "5", "true"])
def test_voice_tts_voice_rejects_invalid_values(tmp_path, value):
    text = VALID + f"\n[voice.tts]\nvoice = {value}\n"
    with pytest.raises(
        ConfigError, match=r"\[voice.tts\].voice must be a non-empty string"
    ):
        load_config(write(tmp_path, text))


def test_voice_wake_word_window_seconds_is_configurable(tmp_path):
    text = VALID + "\n[voice]\nwake_word_window_seconds = 45\n"
    config = load_config(write(tmp_path, text))

    assert config.voice.wake_word_window_seconds == 45


@pytest.mark.parametrize("value", ["0", "-1", "true", '"60"'])
def test_voice_wake_word_window_seconds_rejects_invalid_values(tmp_path, value):
    text = VALID + f"\n[voice]\nwake_word_window_seconds = {value}\n"
    with pytest.raises(
        ConfigError,
        match=r"\[voice\].wake_word_window_seconds must be a positive integer",
    ):
        load_config(write(tmp_path, text))


def test_voice_follow_up_window_seconds_is_configurable(tmp_path):
    text = VALID + "\n[voice]\nfollow_up_window_seconds = 45\n"
    config = load_config(write(tmp_path, text))

    assert config.voice.follow_up_window_seconds == 45


@pytest.mark.parametrize("value", ["0", "-1", "true", '"60"'])
def test_voice_follow_up_window_seconds_rejects_invalid_values(tmp_path, value):
    text = VALID + f"\n[voice]\nfollow_up_window_seconds = {value}\n"
    with pytest.raises(
        ConfigError,
        match=r"\[voice\].follow_up_window_seconds must be a positive integer",
    ):
        load_config(write(tmp_path, text))


def test_voice_barge_in_trigger_frames_is_configurable(tmp_path):
    text = VALID + "\n[voice]\nbarge_in_trigger_frames = 6\n"
    config = load_config(write(tmp_path, text))

    assert config.voice.barge_in_trigger_frames == 6


@pytest.mark.parametrize("value", ["0", "-1", "true", '"6"'])
def test_voice_barge_in_trigger_frames_rejects_invalid_values(tmp_path, value):
    text = VALID + f"\n[voice]\nbarge_in_trigger_frames = {value}\n"
    with pytest.raises(
        ConfigError,
        match=r"\[voice\].barge_in_trigger_frames must be a positive integer",
    ):
        load_config(write(tmp_path, text))


def test_voice_barge_in_vad_threshold_is_configurable(tmp_path):
    text = VALID + "\n[voice]\nbarge_in_vad_threshold = 0.6\n"
    config = load_config(write(tmp_path, text))

    assert config.voice.barge_in_vad_threshold == 0.6


@pytest.mark.parametrize("value", ["0", "-0.1", "1.1", "true", '"0.6"'])
def test_voice_barge_in_vad_threshold_rejects_invalid_values(tmp_path, value):
    text = VALID + f"\n[voice]\nbarge_in_vad_threshold = {value}\n"
    with pytest.raises(
        ConfigError,
        match=r"\[voice\].barge_in_vad_threshold must be a number between 0 and 1",
    ):
        load_config(write(tmp_path, text))
