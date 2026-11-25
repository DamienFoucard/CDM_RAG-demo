"""Verify and capture the pinned Microsoft/CDM source inputs."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


UPSTREAM_REPOSITORY = "https://github.com/microsoft/CDM"
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
LICENSE_FILES = ("LICENSE", "LICENSE.md", "LICENSE.txt")


def _git(source_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "git command failed"
        raise ValueError(f"source checkout validation failed: {detail}")
    return result.stdout.strip()


def _normalized_repository(value: str) -> str:
    value = value.strip()
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")
    if value.startswith("ssh://git@github.com/"):
        value = "https://github.com/" + value.removeprefix("ssh://git@github.com/")
    parsed = urlsplit(value)
    if parsed.hostname and parsed.hostname.lower() == "github.com":
        path = parsed.path.rstrip("/")
        if path.lower().endswith(".git"):
            path = path[:-4]
        return f"https://github.com{path}".lower()
    return value.rstrip("/").removesuffix(".git").lower()


def verify_checkout(source_root: Path, requested_commit: str) -> str:
    """Require a clean checkout of the exact requested upstream commit."""
    if not isinstance(requested_commit, str) or not COMMIT_PATTERN.fullmatch(requested_commit):
        raise ValueError("source commit must be a full 40-character SHA")
    root = source_root.resolve(strict=True)
    repo_root = Path(_git(root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if repo_root != root:
        raise ValueError("source path must be the root of the Microsoft/CDM checkout")
    actual_commit = _git(root, "rev-parse", "--verify", "HEAD^{commit}").lower()
    if actual_commit != requested_commit.lower():
        raise ValueError("checked-out source commit does not match the requested commit")
    origin = _git(root, "remote", "get-url", "origin")
    if _normalized_repository(origin) != _normalized_repository(UPSTREAM_REPOSITORY):
        raise ValueError("source checkout origin must be Microsoft/CDM")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("source checkout must be clean before it is captured")
    return actual_commit


def _safe_file(source_root: Path, candidate: Path) -> Path:
    path = candidate.resolve(strict=True)
    try:
        path.relative_to(source_root)
    except ValueError as exc:
        raise ValueError("source import escapes the checked-out repository") from exc
    if not path.is_file():
        raise ValueError(f"source input is not a file: {path.relative_to(source_root)}")
    return path


def _resolve_import(source_root: Path, document: Path, corpus_path: str) -> Path | None:
    """Resolve local CDM corpus imports, returning None for non-local imports."""
    value = corpus_path.strip().split("#", 1)[0]
    scheme = urlsplit(value).scheme
    if not value or "\\" in value or scheme in {"http", "https"}:
        return None
    if value.startswith("cdm:/"):
        relative = value.removeprefix("cdm:/")
        candidates = [source_root / "schemaDocuments" / relative]
    elif value.startswith("/"):
        relative = value.lstrip("/")
        candidates = [source_root / "schemaDocuments" / relative, source_root / relative]
    else:
        candidates = [document.parent / value, source_root / "schemaDocuments" / value]
    for candidate in candidates:
        if candidate.exists():
            return _safe_file(source_root, candidate)
    if value.startswith("cdm:/"):
        return None
    raise ValueError(
        f"local CDM import {corpus_path!r} referenced by "
        f"{document.relative_to(source_root).as_posix()} is missing"
    )


def collect_source_inputs(
    source_root: Path,
    primary_paths: list[Path],
    commit: str,
) -> dict[str, Any]:
    """Hash the scoped source files, license, and transitive local imports."""
    root = source_root.resolve(strict=True)
    if not COMMIT_PATTERN.fullmatch(commit):
        raise ValueError("source commit must be a full 40-character SHA")
    pending = [_safe_file(root, path if path.is_absolute() else root / path) for path in primary_paths]
    license_path = next((root / name for name in LICENSE_FILES if (root / name).is_file()), None)
    if license_path is None:
        raise ValueError("Microsoft/CDM license file was not found in the source checkout")
    pending.append(_safe_file(root, license_path))

    captured: dict[str, Path] = {}
    external_imports: set[str] = set()
    while pending:
        path = pending.pop()
        relative = path.relative_to(root).as_posix()
        if relative in captured:
            continue
        captured[relative] = path
        if not path.name.endswith((".cdm.json", ".manifest.cdm.json")):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid source JSON: {relative}") from exc
        imports = payload.get("imports", [])
        if not isinstance(imports, list):
            raise ValueError(f"imports must be a list in {relative}")
        for item in imports:
            if not isinstance(item, dict):
                raise ValueError(f"invalid import entry in {relative}")
            corpus_path = item.get("corpusPath") or item.get("uri")
            if not isinstance(corpus_path, str) or not corpus_path.strip():
                raise ValueError(f"import entry has no corpus path in {relative}")
            imported = _resolve_import(root, path, corpus_path)
            if imported is None:
                external_imports.add(corpus_path)
            else:
                pending.append(imported)

    files: list[dict[str, Any]] = []
    for relative, path in sorted(captured.items()):
        data = path.read_bytes()
        files.append({
            "path": relative,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        })
    source_hash = hashlib.sha256(json.dumps(
        {"files": files, "external_imports": sorted(external_imports)},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    license_relative = license_path.relative_to(root).as_posix()
    return {
        "repository": UPSTREAM_REPOSITORY,
        "commit": commit.lower(),
        "source_hash": source_hash,
        "license_file": license_relative,
        "license_sha256": next(item["sha256"] for item in files if item["path"] == license_relative),
        "files": files,
        "external_imports": sorted(external_imports),
    }


def copy_source_inputs(source_root: Path, provenance: dict[str, Any], destination: Path) -> None:
    """Copy captured inputs into an immutable release staging directory."""
    root = source_root.resolve(strict=True)
    destination = destination.resolve()
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700, exist_ok=False)
    try:
        for entry in provenance.get("files", []):
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("source provenance contains an unsafe path")
            source = _safe_file(root, root / relative)
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
                raise ValueError(f"source input changed after provenance capture: {relative.as_posix()}")
            target = staging / relative
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(0o600)
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
