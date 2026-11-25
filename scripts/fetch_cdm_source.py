#!/usr/bin/env python3
"""Fetch the pinned Microsoft/CDM source inputs for catalog maintenance."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.schema.cdm_source import UPSTREAM_REPOSITORY, verify_checkout


def git(*args: str, input_text: str | None = None) -> str:
    result = subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True, input=input_text,
    )
    return result.stdout.strip()


def upstream_pin() -> tuple[str, str]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    upstream = config["tool"]["cdm-rag"]["upstream"]
    repository = upstream["repository"]
    commit = upstream["commit"].lower()
    if repository.removesuffix(".git").rstrip("/").lower() != UPSTREAM_REPOSITORY.lower():
        raise ValueError("pyproject.toml upstream repository does not match the source verifier")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("pyproject.toml must pin a full 40-character upstream commit SHA")
    return repository, commit


def expected_paths() -> list[str]:
    catalog = json.loads((ROOT / "seed" / "catalog.json").read_text(encoding="utf-8"))
    provenance = catalog.get("provenance", {})
    paths = [entry.get("path") for entry in provenance.get("files", [])]
    if not paths or any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("seed/catalog.json has no usable source file list")
    for value in paths:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe CDM source path in catalog provenance: {value}")
    return paths


def verify_source_files(source: Path, paths: list[str]) -> None:
    missing = [path for path in paths if not (source / path).is_file()]
    if missing:
        sample = ", ".join(missing[:4])
        raise ValueError(f"fetched CDM checkout is missing {len(missing)} pinned source files: {sample}")


def has_full_schema_documents(source: Path) -> bool:
    sparse_file = source / ".git" / "info" / "sparse-checkout"
    if not sparse_file.is_file():
        return True
    patterns = {line.strip() for line in sparse_file.read_text(encoding="utf-8").splitlines()}
    return "/schemaDocuments/" in patterns


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the pinned Microsoft/CDM source checkout.")
    parser.add_argument(
        "--destination", type=Path, default=ROOT / ".cache" / "microsoft-cdm",
        help="Checkout destination (default: .cache/microsoft-cdm)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Materialize all schemaDocuments for upstream updates that changed the source scope",
    )
    args = parser.parse_args()

    try:
        repository, commit = upstream_pin()
        destination = args.destination.expanduser().resolve()
        paths = expected_paths()
        if args.full:
            verify_paths = ["LICENSE"]
        else:
            verify_paths = paths

        if destination.exists():
            verify_checkout(destination, commit)
            if args.full and not has_full_schema_documents(destination):
                raise ValueError(
                    "destination already contains a sparse source subset; use a new --destination for --full"
                )
            verify_source_files(destination, verify_paths)
            print(f"Microsoft/CDM {commit} is already available at {destination}")
            return

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".cdm-fetch-", dir=destination.parent) as temp:
            checkout = Path(temp) / "microsoft-cdm"
            git("clone", "--filter=blob:none", "--no-checkout", "--no-tags", repository, str(checkout))
            git("-C", str(checkout), "fetch", "--depth=1", "origin", commit)
            git("-C", str(checkout), "sparse-checkout", "init", "--no-cone")
            if args.full:
                patterns = "/LICENSE\n/schemaDocuments/\n"
            else:
                patterns = "".join(f"/{path}\n" for path in paths)
            git("-C", str(checkout), "sparse-checkout", "set", "--no-cone", "--stdin", input_text=patterns)
            git("-C", str(checkout), "checkout", "--detach", commit)
            verify_checkout(checkout, commit)
            verify_source_files(checkout, verify_paths)
            checkout.rename(destination)
        print(f"Fetched Microsoft/CDM {commit} to {destination}")
        materialized = "LICENSE and all schemaDocuments" if args.full else f"{len(paths)} source paths"
        print(f"Materialized {materialized}; use --full to fetch all schemaDocuments.")
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
        raise SystemExit(f"Could not fetch pinned Microsoft/CDM source: {detail}") from exc


if __name__ == "__main__":
    main()
