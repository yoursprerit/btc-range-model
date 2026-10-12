"""Encrypted vault for the live execution reports.

The per-account reports (``data/overall/accounts/<ACCT>/executed_book_live.json``
plus ``executed_archive/``) hold account numbers, balances and holdings, so they
never go to GitHub in the clear.  This module seals them into a flat folder of
opaque files that ARE safe to publish, and opens them again (into a temp folder)
for the Executed Book page.

* **Cipher** -- AES-256-GCM.  Every file gets a fresh random salt + nonce; the key
  is ``scrypt(passphrase, salt)``.  The relative path is bound in as associated
  data, so a sealed file cannot be renamed or swapped for another without the
  tag failing.
* **Opaque names** -- ``<hmac(passphrase, relpath)>.bin``; nothing about an
  account id, date or file kind leaks through the file name.  A sealed
  ``manifest.bin`` maps the opaque names back to relative paths.
* **Sizes/timing** are not hidden (file count and lengths are visible).

Passphrase: ``IBKR_VAULT_KEY`` (env or Streamlit secret).  Losing it loses the
vault -- the plaintext on the executor PC is the source of truth and can be
re-sealed at any time.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"RVLT1"
SALT_LEN, NONCE_LEN = 16, 12
MANIFEST = "manifest.bin"
VAULT_DIRNAME = "vault"
_SCRYPT = dict(n=2 ** 14, r=8, p=1, dklen=32)
_ENV = "IBKR_VAULT_KEY"


class VaultError(Exception):
    """Wrong passphrase, tampered/corrupt file, or missing vault."""


def passphrase(explicit: str | None = None) -> str | None:
    """The vault passphrase: *explicit*, else ``IBKR_VAULT_KEY`` from the
    environment or Streamlit secrets; None when none is configured."""
    if explicit:
        return explicit
    val = os.environ.get(_ENV, "").strip()
    if val:
        return val
    try:
        import streamlit as st
        return str(st.secrets.get(_ENV, "")).strip() or None
    except Exception:
        return None


def _key(pw: str, salt: bytes) -> bytes:
    return hashlib.scrypt(pw.encode("utf-8"), salt=salt, **_SCRYPT)


def seal(data: bytes, pw: str, aad: str) -> bytes:
    salt, nonce = os.urandom(SALT_LEN), os.urandom(NONCE_LEN)
    ct = AESGCM(_key(pw, salt)).encrypt(nonce, data, aad.encode("utf-8"))
    return MAGIC + salt + nonce + ct


def unseal(blob: bytes, pw: str, aad: str) -> bytes:
    head = len(MAGIC) + SALT_LEN + NONCE_LEN
    if len(blob) < head + 16 or not blob.startswith(MAGIC):
        raise VaultError("not a vault file")
    salt = blob[len(MAGIC):len(MAGIC) + SALT_LEN]
    nonce = blob[len(MAGIC) + SALT_LEN:head]
    try:
        return AESGCM(_key(pw, salt)).decrypt(nonce, blob[head:], aad.encode("utf-8"))
    except InvalidTag as e:
        raise VaultError("wrong passphrase or tampered file") from e


def opaque_name(relpath: str, pw: str) -> str:
    mac = hmac.new(pw.encode("utf-8"), b"name:" + relpath.encode("utf-8"), hashlib.sha256)
    return mac.hexdigest()[:40] + ".bin"


def _rel(p: Path, root: Path) -> str:
    return p.relative_to(root).as_posix()


def seal_tree(src_root, vault_dir, pw: str) -> int:
    """Encrypt every ``*.json`` under *src_root* into *vault_dir*.

    *src_root* is the folder holding ``accounts/<ACCT>/...``.  Files whose
    content is unchanged are rewritten too (fresh nonce) -- callers that want
    no-op commits should rely on git, not on this.  Stale sealed files from a
    previous seal are removed.  Returns the number of files sealed."""
    src_root, vault_dir = Path(src_root), Path(vault_dir)
    files = sorted(p for p in src_root.rglob("*.json")
                   if p.is_file() and not p.name.startswith("."))
    vault_dir.mkdir(parents=True, exist_ok=True)
    manifest, keep = {}, {MANIFEST}
    for p in files:
        rel = _rel(p, src_root)
        name = opaque_name(rel, pw)
        (vault_dir / name).write_bytes(seal(p.read_bytes(), pw, rel))
        manifest[name] = rel
        keep.add(name)
    (vault_dir / MANIFEST).write_bytes(
        seal(json.dumps(manifest, sort_keys=True).encode(), pw, MANIFEST))
    for old in vault_dir.glob("*.bin"):
        if old.name not in keep:
            old.unlink()
    return len(files)


def open_tree(vault_dir, pw: str, dest=None) -> Path:
    """Decrypt a vault into *dest* (default: a fresh temp folder) and return it.

    The result mirrors the original layout, so ``accounts/<ACCT>/...`` can be
    handed straight to ``executed_book.account_report_paths``.  Raises
    :class:`VaultError` on a missing vault, wrong passphrase or tampering."""
    vault_dir = Path(vault_dir)
    mpath = vault_dir / MANIFEST
    if not mpath.is_file():
        raise VaultError("no vault here")
    manifest = json.loads(unseal(mpath.read_bytes(), pw, MANIFEST))
    dest = Path(dest) if dest else Path(tempfile.mkdtemp(prefix="reportvault-"))
    try:
        for name, rel in manifest.items():
            target = (dest / rel).resolve()
            if dest.resolve() not in target.parents:      # path traversal guard
                raise VaultError("manifest path escapes the vault")
            f = vault_dir / name
            if not f.is_file():
                raise VaultError("manifest lists a missing file")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(unseal(f.read_bytes(), pw, rel))
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    return dest


def fetch_vault(repo: str, branch: str, path: str, dest, token: str | None = None) -> int:
    """Download a published vault folder from GitHub into *dest* (for a private
    repo pass a read-only *token*).  Returns the number of files fetched."""
    def get(url: str) -> bytes:
        req = urllib.request.Request(url, headers={
            "Accept": "application/vnd.github+json",
            **({"Authorization": f"Bearer {token}"} if token else {})})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read()
    listing = json.loads(get(
        f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"))
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for item in listing:
        name = item.get("name", "")
        if item.get("type") != "file" or not name.endswith(".bin") or "/" in name:
            continue
        (dest / name).write_bytes(get(item["download_url"]))
        n += 1
    return n
