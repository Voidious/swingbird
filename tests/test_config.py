import pytest

from swingbird.config import ConfigError, load_config

VALID = """
[llm]
base_url = "https://api.moonshot.ai/v1"
model = "kimi-k2.6"
api_key_env = "MOONSHOT_API_KEY"

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
    assert len(config.channels) == 1
    channel = config.channels[0]
    assert channel.id == "chan-1"
    assert channel.name == "swingbird-dev"
    assert channel.write is True
    assert channel.agents == ("Sonnet",)


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

[[channels]]
id = "chan-1"
name = "swingbird-dev"
write = true
agents = []
"""
    with pytest.raises(ConfigError, match="api_key_env"):
        load_config(write(tmp_path, text))


def test_channels_missing(tmp_path):
    text = """
[llm]
base_url = "https://api.moonshot.ai/v1"
model = "kimi-k2.6"
api_key_env = "MOONSHOT_API_KEY"
"""
    with pytest.raises(ConfigError, match=r"\[\[channels\]\]"):
        load_config(write(tmp_path, text))


def test_channels_empty(tmp_path):
    text = """
[llm]
base_url = "https://api.moonshot.ai/v1"
model = "kimi-k2.6"
api_key_env = "MOONSHOT_API_KEY"

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


def test_duplicate_channel_name(tmp_path):
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

[[channels]]
id = "chan-2"
name = "swingbird-dev"
write = false
agents = []
"""
    with pytest.raises(ConfigError, match="duplicate channel name"):
        load_config(write(tmp_path, text))
