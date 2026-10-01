import subprocess

import pytest

_real_run = subprocess.run


@pytest.fixture(autouse=True)
def _forbid_real_buzz_cli(monkeypatch):
    """Fail fast if a test would shell out to the real `buzz` CLI.

    With a relay configured via the environment, an unmocked `buzz` call
    blocks on the network (hangs when the relay is down) instead of failing.
    Tests that exercise `run_buzz_cli` install their own `subprocess.run`
    fake, which overrides this one.
    """

    def guarded_run(args, *a, **kw):
        if args and args[0] == "buzz":
            raise AssertionError(f"test invoked the real buzz CLI: {args!r}")
        return _real_run(args, *a, **kw)

    monkeypatch.setattr(subprocess, "run", guarded_run)
