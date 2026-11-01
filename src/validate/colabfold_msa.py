"""ColabFold protein search and shared, sequence-indexed structural templates.

Only the search HTTP protocol is used; no ColabFold prediction environment is
loaded. Template alignment follows ColabFold's local HHsearch profile step.
"""
from __future__ import annotations

import datetime as dt
import gzip
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.parse import quote

import requests


class ColabFoldSearchError(RuntimeError):
    """An incomplete search must not become a successful query-only cache."""


def hhsearch_binary(settings):
    """Locate only the requested executable, without model or database scans."""
    root = settings.get("binaries_dir")
    if root:
        path = Path(root) / "hhsearch"
        return str(path) if path.is_file() and os.access(path, os.X_OK) else None
    located = shutil.which("hhsearch")
    if located:
        return located
    path = Path(sys.prefix) / "bin/hhsearch"
    if path.is_file() and os.access(path, os.X_OK):
        return str(path)
    try:
        from colabfold_binaries import binary_path
        path = Path(binary_path("hhsearch"))
    except (ImportError, KeyError, ValueError, FileNotFoundError):
        return None
    return str(path) if path.is_file() and os.access(path, os.X_OK) else None


class _Client:
    def __init__(self, settings):
        self.url = settings.get("server_url", "https://api.colabfold.com").rstrip("/")
        self.deadline = time.monotonic() + float(settings["timeout_seconds"])
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            "PartiDesign/0.1 (+https://github.com/MegaGatlingPea/PartiDesign-official)"
        )

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("ColabFold search exceeded timeout_seconds")
        return remaining

    def sleep(self, seconds):
        time.sleep(min(seconds, self.remaining()))
        self.remaining()

    def request(self, method, path, **kwargs):
        last = None
        for attempt in range(3):
            remaining = self.remaining()
            try:
                response = self.session.request(
                    method, path if path.startswith("https://") else self.url + path,
                    timeout=(min(10, remaining), min(120, remaining)), **kwargs,
                )
                if response.status_code not in {429, 502, 503, 504}:
                    try:
                        response.raise_for_status()
                    except requests.HTTPError:
                        response.close()
                        raise
                    return response
                last = ColabFoldSearchError(f"ColabFold returned HTTP {response.status_code}")
                try:
                    delay = max(1, float(response.headers.get("Retry-After", "5")))
                except ValueError:
                    delay = 5
                response.close()
            except (requests.Timeout, requests.ConnectionError) as exc:
                last = exc
                delay = 3
            if attempt < 2:
                self.sleep(delay)
        raise ColabFoldSearchError(f"ColabFold request failed after 3 attempts: {last}") from last

    def download(self, path, destination):
        response = self.request("GET", path, stream=True)
        try:
            with Path(destination).open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    self.remaining()
                    handle.write(chunk)
        finally:
            response.close()

    def search(self, sequence, directory):
        response = self.request("POST", "/ticket/msa", data={"q": f">101\n{sequence}\n", "mode": "env"})
        ticket = response.json()
        for attempt in range(3):
            if ticket.get("id") or ticket.get("status") not in {"RATELIMIT", "UNKNOWN"}:
                break
            if attempt == 2:
                break
            self.sleep(10)
            ticket = self.request("POST", "/ticket/msa", data={"q": f">101\n{sequence}\n", "mode": "env"}).json()
        identity = ticket.get("id")
        if not isinstance(identity, str) or not identity:
            raise ColabFoldSearchError(f"ColabFold did not accept the search: {ticket}")
        path_id = quote(identity, safe="")
        while ticket.get("status") in {"UNKNOWN", "PENDING", "RUNNING"}:
            self.sleep(5)
            ticket = self.request("GET", "/ticket/" + path_id).json()
        if ticket.get("status") != "COMPLETE":
            raise ColabFoldSearchError(f"ColabFold search failed: {ticket}")
        target = directory / "result.tar.gz"
        self.download("/result/download/" + path_id, target)
        return target, identity


def _msa_archive(path, sequence):
    from .af3_msa import _fasta
    records, candidates, seen = [], [], set()
    with tarfile.open(path, "r:gz") as archive:
        members = {Path(m.name).name: m for m in archive if m.isfile()}
        if "uniref.a3m" not in members:
            raise ColabFoldSearchError("ColabFold result has no uniref.a3m")
        if "pdb70.m8" not in members:
            raise ColabFoldSearchError("ColabFold result has no template-search result (pdb70.m8)")
        for name in ("uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m"):
            if name not in members:
                continue
            text = archive.extractfile(members[name]).read().decode()
            chunks = [_fasta(chunk) for chunk in text.split("\0") if chunk.strip()]
            chunks = [chunk for chunk in chunks if chunk and chunk[0][0].split()[0] == "101"]
            if len(chunks) != 1 or chunks[0][0][1] != sequence:
                raise ColabFoldSearchError(f"{name}: returned query differs from requested sequence")
            for title, aligned in chunks[0]:
                if aligned not in seen:
                    records.append((title, aligned))
                    seen.add(aligned)
        if "pdb70.m8" in members:
            for row in archive.extractfile(members["pdb70.m8"]).read().decode().splitlines():
                columns = row.split()
                if len(columns) >= 2 and columns[0] == "101" and columns[1] not in candidates:
                    candidates.append(columns[1])
    return "".join(f">{title}\n{aligned}\n" for title, aligned in records), candidates[:20]


def _template_archive(path, directory):
    """Read known profile files and mmCIFs, never extract archive paths."""
    structures = {}
    profile_names = {f"pdb70_{part}.{suffix}" for part in ("a3m", "hhm", "cs219") for suffix in ("ffdata", "ffindex")}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name in profile_names:
                with archive.extractfile(member) as source, (directory / name).open("wb") as target:
                    shutil.copyfileobj(source, target)
            elif re.fullmatch(r"[A-Za-z0-9]{4}\.cif(?:\.gz)?", name):
                raw = archive.extractfile(member).read()
                if name.endswith(".gz"):
                    raw = gzip.decompress(raw)
                structures[name[:4].lower()] = raw.decode()
    for suffix in ("ffdata", "ffindex"):
        if not (directory / f"pdb70_a3m.{suffix}").is_file():
            raise ColabFoldSearchError(f"Template bundle has no pdb70_a3m.{suffix}")
    # HHsearch's ColabFold template subset does not need a CS219 prefilter.
    if not (directory / "pdb70_cs219.ffindex").is_file():
        shutil.copyfile(directory / "pdb70_a3m.ffindex", directory / "pdb70_cs219.ffindex")
    (directory / "pdb70_cs219.ffdata").touch()
    return structures


def _hhr_parser():
    # Reuse the bundled parser, without loading any model or model assets.
    from partidesign.paths import community_source
    source = str(community_source("mol1"))
    if source not in sys.path:
        sys.path.insert(0, source)
    from mol.data.template.template_parser import HHRParser
    return HHRParser


def _hhsearch(msa, directory, binary, timeout):
    query = directory / "query.a3m"
    output = directory / "templates.hhr"
    query.write_text(msa)
    command = [binary, "-i", str(query), "-o", str(output), "-d", str(directory / "pdb70"),
               "-cpu", "1", "-maxseq", "1000000", "-Z", "20", "-B", "20"]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
                               text=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
    if completed.returncode:
        raise ColabFoldSearchError(f"HHsearch exited {completed.returncode}: {completed.stderr[-4000:]}")
    if not output.is_file():
        raise ColabFoldSearchError("HHsearch did not publish its alignment")
    return _hhr_parser().parse(output.read_text())


def _profile_sequences(directory):
    from .af3_msa import _fasta
    result = {}
    with (directory / "pdb70_a3m.ffdata").open("rb") as data:
        for row in (directory / "pdb70_a3m.ffindex").read_text().splitlines():
            name, offset, size = row.split()[:3]
            data.seek(int(offset))
            records = _fasta(data.read(int(size)).rstrip(b"\0").decode())
            protein = [(title, seq) for title, seq in records if not title.startswith(("ss_", "Consensus"))]
            if protein:
                result[name] = protein[0][1].replace("-", "").replace(".", "").upper()
    return result


def _single_chain(mmcif, chain):
    """Keep full polymer sequence metadata even where coordinates are absent."""
    import numpy as np
    from Bio.Data.PDBData import protein_letters_3to1_extended
    from biotite.structure.io.pdbx import CIFFile, CIFCategory, CIFBlock
    parsed = CIFFile.read(io.StringIO(mmcif))
    block = next(iter(parsed.values()))
    atom = block["atom_site"]
    auth = atom["auth_asym_id"].as_array()
    labels = atom["label_asym_id"].as_array()
    asym = block["struct_asym"]
    poly = block["entity_poly"]
    protein_entities = set(poly["entity_id"].as_array()[
        np.char.find(np.char.lower(poly["type"].as_array().astype(str)), "polypeptide") >= 0])
    protein_labels = asym["id"].as_array()[np.isin(asym["entity_id"].as_array(), list(protein_entities))]
    is_protein = np.isin(labels, protein_labels)
    mask = (auth == chain) & is_protein
    if not mask.any():
        mask = (labels == chain) & is_protein
    selected = set(labels[mask])
    if len(selected) != 1:
        raise ValueError(f"Template chain {chain!r} does not identify one polymer chain")
    label = next(iter(selected))
    asym = block["struct_asym"]
    entity = str(asym["entity_id"].as_array()[asym["id"].as_array() == label][0])
    sequence = block["entity_poly_seq"]
    seq_mask = sequence["entity_id"].as_array() == entity
    nums = sequence["num"].as_array(int)[seq_mask]
    names = sequence["mon_id"].as_array()[seq_mask]
    order = np.argsort(nums, kind="stable")
    # Repeated numbering denotes microheterogeneity; pick one physical position.
    monomers = {}
    for i in order:
        monomers.setdefault(int(nums[i]), str(names[i]))
    seq = "".join(protein_letters_3to1_extended.get(name.upper(), "X") for name in monomers.values())
    if not seq:
        raise ValueError("Template has no polymer sequence")
    revisions = block.get("pdbx_audit_revision_history")
    dates = []
    if revisions is not None and "revision_date" in revisions:
        for value in revisions["revision_date"].as_array():
            try:
                dates.append(dt.date.fromisoformat(str(value)))
            except ValueError:
                pass
    if not dates:
        raise ValueError("Template first-public-release date is unavailable")
    release = min(dates).isoformat()
    target = CIFBlock()
    retain = ("entry", "chem_comp", "exptl", "refine", "pdbx_audit_revision_history")
    for name in retain:
        if name in block:
            target[name] = block[name]
    filters = {"struct_asym": ("id", label), "entity": ("id", entity),
               "entity_poly": ("entity_id", entity), "entity_poly_seq": ("entity_id", entity),
               "pdbx_poly_seq_scheme": ("asym_id", label),
               "pdbx_struct_mod_residue": ("label_asym_id", label)}
    for name, (column, value) in filters.items():
        category = block.get(name)
        if category is not None and column in category:
            keep = category[column].as_array() == value
            if keep.any():
                fields = {field: values.as_array()[keep] for field, values in category.items()}
                if name == "entity_poly" and "pdbx_strand_id" in fields:
                    fields["pdbx_strand_id"] = [str(auth[mask][0])] * int(keep.sum())
                target[name] = CIFCategory(fields)
    model = atom["pdbx_PDB_model_num"].as_array()
    atom_mask = (labels == label) & (model == model[mask][0]) & (atom["label_seq_id"].as_array() != ".")
    target["atom_site"] = CIFCategory({field: values.as_array()[atom_mask] for field, values in atom.items()})
    result = CIFFile()
    result["template"] = target
    output = io.StringIO()
    result.write(output)
    return output.getvalue(), seq, release


def _sequence_bridge(profile, template):
    """Map HHsearch profile residues to complete mmCIF polymer positions."""
    start = template.find(profile)
    if start >= 0:
        if template.find(profile, start + 1) >= 0:
            raise ValueError("Template profile sequence has ambiguous placement in SEQRES")
        return {i: start + i for i in range(len(profile))}
    from Bio.Align import PairwiseAligner
    aligner = PairwiseAligner(mode="global", match_score=3, mismatch_score=-4,
                              open_gap_score=-1, extend_gap_score=-0.1)
    aligner.target_end_gap_score = 0
    aligner.query_end_gap_score = 0
    alignment = aligner.align(template, profile)[0]
    mapping = {int(q): int(t) for (ts, te), (qs, qe) in zip(*alignment.aligned)
               for t, q in zip(range(ts, te), range(qs, qe))}
    matches = sum(profile[q] == template[t] or "X" in (profile[q], template[t]) for q, t in mapping.items())
    if len(mapping) < 0.9 * len(profile) or matches < 0.9 * len(mapping):
        raise ValueError("HHsearch profile does not align reliably to this template's SEQRES")
    return mapping


def _templates(msa, candidates, client, directory, settings, binary):
    path = directory / "templates.tar.gz"
    client.download("/template/" + quote(",".join(candidates), safe=",_"), path)
    structures = _template_archive(path, directory)
    hits = _hhsearch(msa, directory, binary, client.remaining())
    profiles = _profile_sequences(directory)
    cutoff = dt.date.fromisoformat(settings["max_template_date"])
    templates, skipped = [], []
    for hit in hits:
        client.remaining()
        name = hit.name.split()[0]
        match = re.fullmatch(r"([A-Za-z0-9]{4})_(\S+)", name)
        if not match:
            skipped.append({"hit": name, "reason": "not a PDB chain"})
            continue
        pdb, chain = match[1].lower(), match[2]
        if pdb not in structures:
            # A subset template bundle may omit a whole mmCIF. Download this
            # specific structure, rather than installing a structural database.
            raw_path = directory / (pdb + ".cif.gz")
            try:
                client.download(f"https://files.rcsb.org/download/{pdb.upper()}.cif.gz", raw_path)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    skipped.append({"hit": name, "reason": "structure unavailable"})
                    continue
                raise
            structures[pdb] = gzip.decompress(raw_path.read_bytes()).decode()
        try:
            text, sequence, release = _single_chain(structures[pdb], chain)
            if dt.date.fromisoformat(release) > cutoff:
                skipped.append({"hit": name, "reason": "after max_template_date", "release_date": release})
                continue
            if name not in profiles:
                raise ValueError("HHsearch profile sequence unavailable")
            profile = profiles[name]
            for letter, position in zip(hit.hit_sequence, hit.indices_hit, strict=True):
                if position >= 0 and (position >= len(profile) or
                        (letter != profile[position] and "X" not in (letter, profile[position]))):
                    raise ValueError("HHsearch residue indices disagree with the profile sequence")
            bridge = _sequence_bridge(profile, sequence)
            mapping = {int(q): bridge[int(t)] for q, t in hit.query_to_hit_mapping.items() if int(t) in bridge}
            if not mapping:
                raise ValueError("No aligned template residues")
        except (KeyError, IndexError, ValueError) as exc:
            skipped.append({"hit": name, "reason": str(exc)})
            continue
        templates.append({"mmcif": text, "query_to_template_map": mapping, "release_date": release,
                          "pdb_id": pdb, "chain_id": chain, "source": "colabfold"})
        if len(templates) == 4:
            break
    return templates, skipped


def search_colabfold(sequence, settings, directory):
    """Search one protein context; caller owns the shared server-serial lock."""
    binary = hhsearch_binary(settings)
    if not binary:
        raise ColabFoldSearchError("ColabFold templates require hhsearch; install colabfold-binaries")
    if not isinstance(sequence, str) or not re.fullmatch(r"[A-Z]+", sequence):
        raise ValueError("ColabFold requires one canonical protein query")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="colabfold-", dir=directory) as raw_directory:
        work = Path(raw_directory)
        client = _Client(settings)
        try:
            archive, ticket = client.search(sequence, work)
            msa, candidates = _msa_archive(archive, sequence)
            templates, skipped = _templates(msa, candidates, client, work, settings, binary) if candidates else ([], [])
            return {"modality": "protein", "sequence": sequence, "unpaired_msa": msa,
                    "paired_msa": f">query\n{sequence}\n", "templates": templates,
                    "no_hit": msa.count(">") == 1 and not templates,
                    "provenance": {"source": "colabfold", "server_url": client.url, "ticket_id": ticket,
                                   "database_revision": "unreported", "template_candidates": len(candidates),
                                   "template_candidate_ids": candidates,
                                   "template_skipped": skipped}}
        finally:
            client.session.close()
