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
    path.write_text(text)
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
    assert config.dispatch.reply_wait_seconds == 90


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
