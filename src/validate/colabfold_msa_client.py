"""ColabFold MMseqs2 MSA Server Client for AlphaFold 3 validation.

Fetches multiple sequence alignments from a ColabFold-compatible MMseqs2 API
server (default: https://api.colabfold.com) and formats the output into
AlphaFold 3 Jackhmmer-compatible A3M format.
"""
from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
import random
import re
import shutil
import string
import tarfile
import tempfile
import time
from typing import Any, Sequence

import requests

logger = logging.getLogger(__name__)

DEFAULT_MSA_SERVER_URL = os.environ.get(
    "MMSEQS_SERVICE_HOST_URL", "https://api.colabfold.com"
).rstrip("/")
DEFAULT_USER_AGENT = "partidesign/1.0"
MAX_REQUEST_ATTEMPTS = 5
MAX_SUBMIT_WAIT_ATTEMPTS = 60
MAX_STATUS_WAIT_ATTEMPTS = 720

_A3M_STRIP_LOWER = str.maketrans("", "", string.ascii_lowercase)


class ColabFoldMSAError(RuntimeError):
    """Raised when ColabFold MSA service encounters an error."""


class ColabFoldMSATimeout(ColabFoldMSAError):
    """Raised when ColabFold MSA service times out."""


def _safe_extract_tar(tar_gz: tarfile.TarFile, destination: str | Path) -> None:
    """Extract tar safely, preventing directory traversal / Zip Slip vulnerabilities."""
    destination_path = Path(destination).resolve()
    for member in tar_gz.getmembers():
        member_path = (destination_path / member.name).resolve()
        if member_path != destination_path and destination_path not in member_path.parents:
            raise ColabFoldMSAError(f"Refusing to extract unsafe tar member: {member.name}")
    if hasattr(tarfile, "data_filter"):
        tar_gz.extractall(destination_path, filter="data")
    else:
        tar_gz.extractall(destination_path)


def _clean_sequence(sequence: str) -> str:
    """Normalize sequence string by stripping whitespace and converting to uppercase."""
    return "".join(sequence.split()).upper()


def _parse_a3m_entries(a3m_text: str) -> list[tuple[str, str]]:
    """Parse A3M text into list of (header, sequence) tuples."""
    entries: list[tuple[str, str]] = []
    current_header: str | None = None
    seq_parts: list[str] = []

    for line in a3m_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header is not None:
                entries.append((current_header, "".join(seq_parts)))
                seq_parts = []
            current_header = line
        else:
            if current_header is not None:
                seq_parts.append(line.replace(" ", ""))

    if current_header is not None and seq_parts:
        entries.append((current_header, "".join(seq_parts)))
    return entries


def deduplicate_and_format_a3m(
    a3m_texts: Sequence[str],
    query_sequence: str,
    max_sequences: int | None = None,
) -> str:
    """Merge and deduplicate A3M records, ensuring compatibility with AF3 Jackhmmer format.

    - The first record is guaranteed to be ``>query`` followed by the exact query sequence.
    - Subsequent records are deduplicated in order based on insertion-stripped characters.
    - Rows that do not match the expected column count after removing lowercases are skipped.
    """
    clean_query = _clean_sequence(query_sequence)
    query_len = len(clean_query)

    seen_aligned: set[str] = set()
    result_entries: list[tuple[str, str]] = [(">query", clean_query)]
    seen_aligned.add(clean_query)

    for a3m_text in a3m_texts:
        if not a3m_text or not a3m_text.strip():
            continue
        entries = _parse_a3m_entries(a3m_text)
        for header, seq in entries:
            # Check length of non-lowercase columns
            non_insert_len = len(seq.translate(_A3M_STRIP_LOWER))
            if non_insert_len != query_len:
                continue

            stripped = seq.translate(_A3M_STRIP_LOWER)
            if stripped in seen_aligned:
                continue
            seen_aligned.add(stripped)

            clean_header = header.strip()
            if not clean_header.startswith(">"):
                clean_header = ">" + clean_header

            result_entries.append((clean_header, seq))
            if max_sequences is not None and len(result_entries) >= max_sequences:
                break
        if max_sequences is not None and len(result_entries) >= max_sequences:
            break

    lines: list[str] = []
    for header, seq in result_entries:
        lines.append(header)
        lines.append(seq)
    return "\n".join(lines) + "\n"


class ColabFoldMSAClient:
    """Client for querying ColabFold MMseqs2 MSA Server."""

    def __init__(
        self,
        server_url: str | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_seconds: float = 600.0,
    ) -> None:
        raw_url = (server_url or DEFAULT_MSA_SERVER_URL).strip()
        if not raw_url.startswith(("http://", "https://")):
            raw_url = f"https://{raw_url}"
        self.server_url = raw_url.rstrip("/")
        self.user_agent = user_agent
        self.timeout_seconds = float(timeout_seconds)

    def _request_with_retries(
        self, method: str, url: str, operation: str, **kwargs: Any
    ) -> requests.Response:
        headers = dict(kwargs.pop("headers", {}))
        headers.setdefault("User-Agent", self.user_agent)

        last_error: Exception | None = None
        for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
            try:
                response = requests.request(
                    method, url, timeout=30.0, headers=headers, **kwargs
                )
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt >= MAX_REQUEST_ATTEMPTS:
                    raise ColabFoldMSAError(
                        f"{operation} failed after {MAX_REQUEST_ATTEMPTS} attempts: {exc}"
                    ) from exc
                sleep_time = min(30.0, 2.0**attempt + random.uniform(0.5, 2.0))
                logger.warning(
                    "%s failed (attempt %d/%d). Retrying in %.1fs: %s",
                    operation,
                    attempt,
                    MAX_REQUEST_ATTEMPTS,
                    sleep_time,
                    exc,
                )
                time.sleep(sleep_time)

        raise ColabFoldMSAError(f"{operation} failed: {last_error}")

    def _submit(self, sequence: str, mode: str = "env") -> dict[str, Any]:
        """Submit query sequence to ColabFold MMseqs2 server."""
        query_payload = f">101\n{sequence}\n"
        data = {"q": query_payload, "mode": mode}
        response = self._request_with_retries(
            "POST",
            f"{self.server_url}/ticket/msa",
            "Submitting query to ColabFold MSA server",
            data=data,
        )
        try:
            return response.json()
        except ValueError as exc:
            raise ColabFoldMSAError(
                f"ColabFold server returned invalid JSON on submit: {response.text[:500]!r}"
            ) from exc

    def _status(self, ticket_id: str) -> dict[str, Any]:
        """Fetch status of submitted ticket."""
        response = self._request_with_retries(
            "GET",
            f"{self.server_url}/ticket/{ticket_id}",
            f"Fetching status for ticket {ticket_id}",
        )
        try:
            return response.json()
        except ValueError as exc:
            raise ColabFoldMSAError(
                f"ColabFold server returned invalid JSON on status: {response.text[:500]!r}"
            ) from exc

    def _download(self, ticket_id: str, dest_tar_path: Path) -> None:
        """Download tar.gz result archive."""
        response = self._request_with_retries(
            "GET",
            f"{self.server_url}/result/download/{ticket_id}",
            f"Downloading MSA result for ticket {ticket_id}",
        )
        dest_tar_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_tar_path.with_name(f"{dest_tar_path.name}.tmp")
        try:
            with tmp_path.open("wb") as handle:
                handle.write(response.content)
            # Verify it is valid tar.gz before replacing
            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.getmembers()
            os.replace(tmp_path, dest_tar_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def search(self, sequence: str, work_dir: Path | None = None) -> dict[str, Any]:
        """Perform MMseqs2 MSA search and return AF3 Jackhmmer-compatible results.

        Returns a dictionary matching AF3MSACache expected structure:
        {
            "modality": "protein",
            "sequence": sequence,
            "unpaired_msa": <str>,
            "paired_msa": <str>,
            "templates": [],
            "no_hit": bool,
        }
        """
        clean_seq = _clean_sequence(sequence)
        start_time = time.time()
        deadline = start_time + self.timeout_seconds

        # 1. Submit query
        out: dict[str, Any] = {}
        for attempt in range(MAX_SUBMIT_WAIT_ATTEMPTS):
            if time.time() > deadline:
                raise ColabFoldMSATimeout(
                    f"ColabFold MSA search timed out during submission after {self.timeout_seconds}s"
                )
            out = self._submit(clean_seq, mode="env")
            status = out.get("status")
            if status not in ("UNKNOWN", "RATELIMIT"):
                break
            sleep_time = min(15.0, 5.0 + random.uniform(1.0, 4.0))
            logger.info("ColabFold server returned status %s. Sleeping %.1fs...", status, sleep_time)
            time.sleep(sleep_time)

        status = out.get("status")
        if status == "ERROR":
            raise ColabFoldMSAError(
                "ColabFold MMseqs2 API returned an ERROR status. Check if sequence is valid."
            )
        if status == "MAINTENANCE":
            raise ColabFoldMSAError("ColabFold MMseqs2 API is currently undergoing maintenance.")

        ticket_id = out.get("id")
        if not ticket_id:
            raise ColabFoldMSAError(f"ColabFold response did not contain a ticket id: {out}")

        logger.info("Submitted MSA query to %s (ticket: %s)", self.server_url, ticket_id)

        # 2. Poll until complete
        current_status = status
        for _ in range(MAX_STATUS_WAIT_ATTEMPTS):
            if time.time() > deadline:
                raise ColabFoldMSATimeout(
                    f"ColabFold MSA search timed out while waiting for ticket {ticket_id}"
                )
            if current_status not in ("UNKNOWN", "RUNNING", "PENDING"):
                break
            sleep_time = min(15.0, 5.0 + random.uniform(0.5, 2.5))
            time.sleep(sleep_time)
            status_out = self._status(ticket_id)
            current_status = status_out.get("status")

        if current_status == "COMPLETE":
            pass
        elif current_status == "ERROR":
            raise ColabFoldMSAError(f"ColabFold MSA ticket {ticket_id} failed with status ERROR.")
        else:
            raise ColabFoldMSAError(
                f"ColabFold MSA ticket {ticket_id} ended with unexpected status: {current_status}"
            )

        # 3. Download & extract archive
        scratch_dir = Path(tempfile.mkdtemp(prefix="colabfold-msa-", dir=work_dir))
        tar_path = scratch_dir / "out.tar.gz"
        try:
            self._download(ticket_id, tar_path)
            with tarfile.open(tar_path, "r:gz") as tar_gz:
                _safe_extract_tar(tar_gz, scratch_dir)

            a3m_files: list[Path] = []
            # Standard order: uniref.a3m first, then environmental database hits
            uniref_path = scratch_dir / "uniref.a3m"
            if uniref_path.is_file():
                a3m_files.append(uniref_path)
            for file_path in sorted(scratch_dir.glob("*.a3m")):
                if file_path != uniref_path and file_path.is_file():
                    a3m_files.append(file_path)

            a3m_texts = [p.read_text(encoding="utf-8", errors="replace") for p in a3m_files]
            unpaired_msa = deduplicate_and_format_a3m(a3m_texts, clean_seq)

            # In AF3 official docs (docs/input.md), custom unpaired MSAs run pairedMSA-free
            # by setting pairedMsa to query-only (or empty). Setting query-only satisfies
            # all column-width and query identity checks in PartiDesign validation.
            paired_msa = f">query\n{clean_seq}\n"

            # Check depth (number of sequences in unpaired MSA)
            unpaired_depth = sum(1 for line in unpaired_msa.splitlines() if line.startswith(">"))
            no_hit = unpaired_depth <= 1

            return {
                "modality": "protein",
                "sequence": clean_seq,
                "unpaired_msa": unpaired_msa,
                "paired_msa": paired_msa,
                "templates": [],
                "no_hit": no_hit,
            }
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)
