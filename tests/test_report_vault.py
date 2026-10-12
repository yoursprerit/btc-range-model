import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import report_vault as rv  # noqa: E402
import executed_book as eb  # noqa: E402


@pytest.fixture
def src(tmp_path):
    d = tmp_path / "src" / "accounts" / "U1111111"
    (d / "executed_archive").mkdir(parents=True)
    (d / "executed_book_live.json").write_text(json.dumps({"account": "U1111111", "cash": 123456}))
    (d / "executed_archive" / "2026-10-09_live.json").write_text('{"as_of": "2026-10-09"}')
    return tmp_path / "src"


def test_roundtrip_and_layout(src, tmp_path):
    v = tmp_path / "vault"
    assert rv.seal_tree(src, v, "pw") == 2
    out = rv.open_tree(v, "pw", tmp_path / "out")
    assert json.loads((out / "accounts/U1111111/executed_book_live.json").read_text())["cash"] == 123456
    assert [p.parent.name for p in eb.account_report_paths(out)] == ["U1111111"]


def test_nothing_readable_in_the_vault(src, tmp_path):
    v = tmp_path / "vault"
    rv.seal_tree(src, v, "pw")
    blob = b"".join(p.read_bytes() for p in v.iterdir()) + "".join(p.name for p in v.iterdir()).encode()
    for secret in (b"U1111111", b"123456", b"2026-10-09", b"executed"):
        assert secret not in blob


def test_wrong_passphrase_and_tamper(src, tmp_path):
    v = tmp_path / "vault"
    rv.seal_tree(src, v, "pw")
    with pytest.raises(rv.VaultError):
        rv.open_tree(v, "other", tmp_path / "o1")
    f = next(p for p in v.iterdir() if p.name != rv.MANIFEST)
    b = bytearray(f.read_bytes()); b[-1] ^= 1; f.write_bytes(bytes(b))
    with pytest.raises(rv.VaultError):
        rv.open_tree(v, "pw", tmp_path / "o2")
    assert not (tmp_path / "o2").exists()          # partial output cleaned up


def test_swapping_sealed_files_is_detected(src, tmp_path):
    v = tmp_path / "vault"
    rv.seal_tree(src, v, "pw")
    a, b = [p for p in v.iterdir() if p.name != rv.MANIFEST]
    ba, bb = a.read_bytes(), b.read_bytes()
    a.write_bytes(bb); b.write_bytes(ba)
    with pytest.raises(rv.VaultError):
        rv.open_tree(v, "pw", tmp_path / "o3")


def test_reseal_removes_stale_and_missing_vault(src, tmp_path):
    v = tmp_path / "vault"
    rv.seal_tree(src, v, "pw")
    (src / "accounts/U1111111/executed_archive/2026-10-09_live.json").unlink()
    assert rv.seal_tree(src, v, "pw") == 1
    assert len(list(v.glob("*.bin"))) == 2         # one report + manifest
    with pytest.raises(rv.VaultError):
        rv.open_tree(tmp_path / "nope", "pw")


def test_passphrase_from_env(monkeypatch):
    monkeypatch.setenv("IBKR_VAULT_KEY", " s3cret ")
    assert rv.passphrase() == "s3cret"


def test_executor_publishes_only_ciphertext_when_vault_key_set(monkeypatch, tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import ibkr_execute_accounts as acc
    monkeypatch.delenv("IBKR_PUBLISH_REPORTS", raising=False)
    monkeypatch.delenv("IBKR_VAULT_KEY", raising=False)
    assert acc.publishing_enabled() is False
    monkeypatch.setenv("IBKR_VAULT_KEY", "pw")
    assert acc.vault_enabled() and acc.publishing_enabled() is True
    assert acc.publishing_enabled(no_push=True) is False
