import bech32
import pytest

from swingbird.nostr_crypto import (
    NostrCryptoError,
    parse_private_key,
    pubkey_hex,
    sign_event,
    verify_event,
)

PRIVATE_KEY_HEX = "1" * 64  # 32 bytes, well below curve order
PRIVATE_KEY_BYTES = bytes.fromhex(PRIVATE_KEY_HEX)


def test_parse_private_key_hex():
    assert parse_private_key(PRIVATE_KEY_HEX) == PRIVATE_KEY_BYTES


def test_parse_private_key_nsec_roundtrips_to_same_bytes():
    conv = bech32.convertbits(PRIVATE_KEY_BYTES, 8, 5)
    nsec = bech32.bech32_encode("nsec", conv)

    assert parse_private_key(nsec) == PRIVATE_KEY_BYTES


def test_parse_private_key_rejects_garbage_nsec():
    with pytest.raises(NostrCryptoError, match="invalid nsec"):
        parse_private_key("nsec1notavalidbech32string")


def test_parse_private_key_rejects_non_hex():
    with pytest.raises(NostrCryptoError, match="not valid hex or nsec"):
        parse_private_key("not-hex-and-not-nsec")


def test_parse_private_key_rejects_wrong_length_hex():
    with pytest.raises(NostrCryptoError, match="must be 32 bytes"):
        parse_private_key("abcd")


def test_pubkey_hex_is_64_chars():
    assert len(pubkey_hex(PRIVATE_KEY_BYTES)) == 64


def test_sign_and_verify_round_trip():
    event = sign_event(PRIVATE_KEY_BYTES, 1234, 22242, [["relay", "wss://x"]], "")

    assert verify_event(event) is True
    assert event["pubkey"] == pubkey_hex(PRIVATE_KEY_BYTES)


def test_verify_rejects_tampered_content():
    event = sign_event(PRIVATE_KEY_BYTES, 1234, 9, [], "original")
    event["content"] = "tampered"

    assert verify_event(event) is False


def test_verify_rejects_tampered_id():
    event = sign_event(PRIVATE_KEY_BYTES, 1234, 9, [], "hello")
    event["id"] = "0" * 64

    assert verify_event(event) is False


def test_verify_rejects_bad_signature_length():
    event = sign_event(PRIVATE_KEY_BYTES, 1234, 9, [], "hello")
    event["sig"] = "ab"

    assert verify_event(event) is False


def test_verify_rejects_invalid_pubkey():
    event = sign_event(PRIVATE_KEY_BYTES, 1234, 9, [], "hello")
    event["pubkey"] = "not-hex"

    assert verify_event(event) is False


def test_verify_rejects_missing_fields():
    assert verify_event({"id": "abc"}) is False
