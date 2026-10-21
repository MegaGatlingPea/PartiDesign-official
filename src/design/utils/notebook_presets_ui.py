"""Lossless native widget forms for complete notebook task presets.

The payload remains authoritative. Widgets edit existing native values directly;
rendering does not drop fields that are not editable. Notebook relaxation always
replaces output structures in place.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import ipywidgets as W

from . import notebook as backend


_MODALITIES = ("protein", "peptide", "rna", "dna", "ligand")
_LABELS = {
    "sample_num": "Samples", "design_chains": "Design chains",
    "design_length": "Design lengths", "secondary_structure": "Secondary structure",
    "polymer_atom": "Motif atoms", "motif_min_length": "Min length",
    "motif_max_length": "Max length", "cyclic_offset": "Cyclic offset",
    "cyclic_offset_chains": "Cyclic chains", "rm_token_type": "Omit identities",
    "postprocess": "Postprocess", "override": "Replace output", "score": "Score",
    "cdrs": "CDRs", "VH": "VH", "VL": "VL", "VHH": "VHH",
    "ccd": "CCD", "smiles": "SMILES", "resid": "Residue", "atom1": "Atom 1",
    "atom2": "Atom 2", "omit_atoms": "Omit atoms", "token": "Contig",
}


def _label(value: Any) -> str:
    return _LABELS.get(str(value), str(value).replace("_", " ").capitalize())


def _layout(**kwargs: Any) -> W.Layout:
    return W.Layout(width="100%", min_width="0", **kwargs)


def _column(children: list[W.Widget], *classes: str) -> W.VBox:
    box = W.VBox(children, layout=_layout(grid_gap="8px"))
    for name in classes:
        box.add_class(name)
    return box


def _note(text: str) -> W.Label:
    widget = W.Label(text, layout=_layout())
    widget.add_class("pd-native-help")
    return widget


def _section(title: str, children: list[W.Widget]) -> W.VBox:
    heading = W.Label(title, layout=_layout())
    heading.add_class("pd-native-heading")
    return _column([heading, *children], "pd-section", "pd-native-section")


class NativeTaskForm:
    """Edit a complete native task without reducing it to a single-chain form.

    ``panels`` contains stable widgets for steps 0 Input and design, 1 Motif,
    2 Hotspot, 3 Advanced and 4 Save settings. ``controls`` maps native tuple
    paths to scalar widgets; motif selectors use ``("motif_atoms", chain,
    source_selector)``. An empty protein atom selection means all observed atoms.
    ``on_change(name, payload)`` receives an independent copy for each edit.
    ``on_motif_focus(selector)`` receives the focused source residue selector;
    focus and residue-list navigation leave the payload unchanged.
    """

    def __init__(self, name, payload, atoms=None, on_change=None, on_motif_focus=None):
        self.on_change = on_change
        self.on_motif_focus = on_motif_focus
        self.controls: dict[tuple, W.Widget] = {}
        self.panels = {
            index: _column([], "pd-native-panel", f"pd-native-step-{index}")
            for index in range(5)
        }
        self._syncing = False
        self.set_value(name, payload, atoms=atoms)

    @property
    def value(self):
        return self._name, deepcopy(self._payload)

    def set_value(self, name, payload, atoms=None):
        """Replace the task and source snapshot without emitting a change."""
        if not isinstance(name, str) or not isinstance(payload, Mapping):
            raise TypeError("A native task requires a name and a payload mapping.")
        self._validate_distribution(payload)
        self._syncing = True
        try:
            self._name = name
            self._payload = deepcopy(dict(payload))
            self._inplace_relax()
            self._explicit_lengths = {}
            self.atoms = atoms
            self._motif_focus_key = None
            self._length_error = _note("")
            self._length_error.layout.display = "none"
            self.controls.clear()
            self._render_input()
            self._render_motif()
            self._render_hotspot()
            self._render_advanced()
            self._render_save()
        finally:
            self._syncing = False
        return self

    def _get(self, path):
        value = self._payload
        for part in path:
            value = value[part]
        return value

    def _notify(self):
        if not self._syncing and self.on_change is not None:
            self.on_change(self._name, deepcopy(self._payload))

    def _inplace_relax(self):
        if self._payload.get("postprocess"):
            self._payload["override"] = True

    @staticmethod
    def _distribution_chain(payload):
        chains = [(chain, modality) for entry in payload.get("design_chains", [])
                  if isinstance(entry, Mapping) for chain, modality in entry.items()]
        if len(chains) != 1 or chains[0][1] != "ligand":
            return None
        chain = chains[0][0]
        condition = payload.get("condition") or {}
        ligand = (condition.get("ligand") or {}).get(chain) or {}
        if ligand.get("mode") in {"smiles", "ccd", "type_bond", "backbone"}:
            return None
        if chain in (condition.get("backbone") or []):
            return None
        return chain

    @classmethod
    def _validate_distribution(cls, payload):
        def contains(value):
            if isinstance(value, Mapping):
                return any(contains(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return any(contains(item) for item in value)
            return value == "from_distribution"

        lengths = payload.get("design_length")
        if not contains(lengths):
            return
        chain = cls._distribution_chain(payload)
        if (chain is None or not isinstance(lengths, Mapping)
                or dict(lengths) != {chain: "from_distribution"}):
            raise ValueError("From distribution requires one ligand design chain with a free or atom-motif length. Choose explicit lengths before changing its modality or fixing its chemistry.")

    def _validate_length_edit(self, path, value):
        if path[0] not in {"design_chains", "design_length", "condition"}:
            return
        if path[0] == "condition" and path[1:2] not in {("ligand",), ("backbone",)}:
            return
        candidate = deepcopy(self._payload)
        parent = candidate
        for part in path[:-1]:
            parent = parent[part]
        parent[path[-1]] = deepcopy(value)
        self._validate_distribution(candidate)

    def _update(self, path, value):
        if self._syncing:
            return
        self._validate_length_edit(path, value)
        parent = self._payload
        for part in path[:-1]:
            parent = parent[part]
        if not isinstance(parent, Mapping) or path[-1] in parent:
            previous = parent[path[-1]]
            if previous == value and type(previous) is type(value):
                return
        parent[path[-1]] = deepcopy(value)
        self._inplace_relax()
        if path[:2] == ("condition", "contig") and path[-1] == "token":
            self._render_motif()
        if path[0] == "design_chains" or path[-1:] == ("mode",):
            self._render_advanced()
        self._notify()

    def _enum(self, path, value):
        if path and path[0] == "design_chains":
            return _MODALITIES
        if path[-1:] == ("format",):
            return ("normal", "antibody")
        if len(path) >= 4 and path[:2] == ("condition", "ligand") and path[-1] == "mode":
            return ("smiles", "ccd", "type_bond", "atom_motif")
        if path[-1:] == ("order",):
            return ("single", "double", "triple", "aromatic", "any")
        if "coarse" in path and "secondary_structure" in path:
            modalities = {chain: modality for entry in self._payload.get("design_chains", [])
                          if isinstance(entry, Mapping) for chain, modality in entry.items()}
            if modalities.get(path[-1]) in {"rna", "dna"}:
                return ("paired", "unpaired")
            return ("alpha", "beta")
        if "cdrs" in path and value in {"CDR1", "CDR2", "CDR3"}:
            return ("CDR1", "CDR2", "CDR3")
        if "lengths" in path and "antibody" in path and value in {"native", "official"}:
            return ("native", "official")
        return None

    def _scalar(self, path, value, title=None, *, readonly=False, single_line=False):
        kwargs = {"description": title or _label(path[-1]),
                  "style": {"description_width": "94px"}, "layout": _layout()}
        enum = self._enum(path, value) if isinstance(value, str) else None
        if isinstance(value, bool):
            widget = W.Checkbox(value=value, indent=False, **kwargs)
        elif isinstance(value, int):
            widget = W.IntText(value=value, **kwargs)
        elif isinstance(value, float):
            widget = W.FloatText(value=value, **kwargs)
        elif enum is not None:
            options = tuple(dict.fromkeys((*enum, value)))
            widget = W.Dropdown(options=options, value=value, **kwargs)
        elif isinstance(value, str):
            cls = W.Textarea if not single_line and ("\n" in value or len(value) > 86) else W.Text
            extra = {"rows": 3} if cls is W.Textarea else {}
            widget = cls(value=value, continuous_update=False, **extra, **kwargs)
        else:
            widget = W.Text(value="null" if value is None else repr(value),
                            continuous_update=False, **kwargs)
            readonly = True
        widget.disabled = readonly
        widget.add_class("pd-native-control")
        self.controls[path] = widget
        if not readonly:
            def changed(change):
                if self._syncing:
                    return
                try:
                    self._validate_length_edit(path, change["new"])
                except ValueError as exc:
                    self._syncing = True
                    try:
                        widget.value = change["old"]
                    finally:
                        self._syncing = False
                    self._length_error.value = str(exc)
                    self._length_error.layout.display = ""
                    return
                self._length_error.value = ""
                self._length_error.layout.display = "none"
                self._update(path, change["new"])
            widget.observe(changed, names="value")
        return widget

    def _editor(self, path, value, title=None, *, readonly=False):
        title = title or _label(path[-1])
        if isinstance(value, Mapping):
            children = [self._editor((*path, key), item, _label(key), readonly=readonly)
                        for key, item in value.items()]
            if not children:
                children = [_note("No entries are specified ({}).")]
            return _section(title, children)
        if isinstance(value, list):
            children = [self._editor((*path, index), item, f"Item {index + 1}", readonly=readonly)
                        for index, item in enumerate(value)]
            if not children:
                children = [_note("No entries are specified ([]).")]
            return _section(title, children)
        return self._scalar(path, value, title, readonly=readonly)

    def _readout(self, title, value):
        widget = W.Text(value=str(value or ""), description=title, disabled=True,
                        continuous_update=False, style={"description_width": "94px"},
                        layout=_layout())
        widget.add_class("pd-native-readout")
        return widget

    def _source_path(self):
        if self._payload.get("format") == "antibody":
            return (self._payload.get("antigen") or {}).get("input")
        return self._payload.get("input")

    def _render_input(self):
        self.name_control = W.Text(value=self._name, description="Task name",
            continuous_update=False, style={"description_width": "94px"}, layout=_layout())
        self.controls[("task_name",)] = self.name_control

        def rename(change):
            if not self._syncing and change["new"] != self._name:
                self._name = change["new"]
                self._notify()

        self.name_control.observe(rename, names="value")
        core = [self._length_error, self.name_control]
        core.append(self._scalar(("sample_num",), self._payload.get("sample_num", 1)))
        if "format" in self._payload:
            core.append(self._scalar(("format",), self._payload["format"], readonly=True))
        core.append(self._readout("Source", self._source_path() or "No source structure"))
        sections = [_section("Native task", core)]
        if "design_chains" in self._payload:
            rows = []
            for index, entry in enumerate(self._payload["design_chains"]):
                if isinstance(entry, Mapping):
                    for chain, modality in entry.items():
                        row = _column([self._readout("Chain ID", chain),
                            self._scalar(("design_chains", index, chain), modality, "Modality")],
                            "pd-role-row", "pd-native-chain-row")
                        rows.append(row)
                else:
                    rows.append(self._editor(("design_chains", index), entry))
            sections.append(_section("Design chains", rows))
        if "crop" in self._payload:
            sections.append(self._editor(("crop",), self._payload["crop"], "Source crop"))
        antigen = self._payload.get("antigen")
        if isinstance(antigen, Mapping):
            children = [self._readout("Source", antigen.get("input") or "No antigen source")]
            children.extend(self._editor(("antigen", key), value, _label(key))
                            for key, value in antigen.items() if key not in {"input", "hotspot"})
            sections.append(_section("Antigen", children))
        antibody = self._payload.get("antibody")
        if isinstance(antibody, Mapping):
            children = [self._editor(("antibody", key), antibody[key], _label(key),
                                    readonly=key == "chains")
                        for key in ("scaffold", "chains", "template") if key in antibody]
            if "chains" in antibody:
                children.append(self._readout("Modality", "Protein (antibody)"))
            sections.append(_section("Antibody scaffold", children))
        self.panels[0].children = tuple(sections)

    def _source_overview(self, title="Reference source"):
        children = [self._readout("Source", self._source_path() or "No source structure")]
        if self.atoms is not None:
            try:
                rows = backend.chain_summary(self.atoms)
                children.extend(_note(f"{row['chain']} · {row['modality']} · "
                    f"{row['residues']} residues · {row['atoms']} atoms") for row in rows)
            except (ValueError, AttributeError) as exc:
                children.append(_note(f"Source inventory unavailable: {exc}"))
        else:
            children.append(_note("Load the source structure to inspect copied residue atoms."))
        return _section(title, children)

    def _inspect(self, chain, token):
        # A native output chain may copy a differently named source chain.
        source_chains = []
        for segment in token.split("/"):
            match = backend._SOURCE_SEGMENT.fullmatch(segment.strip())
            if match:
                source_chains.append(match.group(1) or match.group(2))
        source_chain = source_chains[0] if source_chains else chain
        return backend.inspect_contig(self.atoms, source_chain, token)

    def _matching_override(self, mapping, record):
        if not isinstance(mapping, Mapping):
            return None, None
        for selector, choice in mapping.items():
            try:
                if backend.source_selector(self.atoms, selector) == [record["uid"]]:
                    return selector, choice
            except (ValueError, TypeError):
                continue
        return None, None

    def _set_atoms(self, chain, record, choice):
        if self._syncing:
            return
        names = backend.motif_atom_names(self.atoms, record["selector"], choice)
        entry = self._payload["condition"]["contig"][chain]
        original = entry.get("polymer_atom")
        mapping = deepcopy(dict(original or {}))
        key, _ = self._matching_override(mapping, record)
        if names is None:
            if key is None:
                return
            del mapping[key]
        else:
            mapping[key or record["selector"]] = names
        if mapping == original:
            return
        entry["polymer_atom"] = mapping
        self._render_motif()
        self._notify()

    def _motif_record(self, selector):
        if self.atoms is None:
            return None
        try:
            selected = backend.source_selector(self.atoms, selector)
        except (ValueError, TypeError):
            return None
        if len(selected) != 1:
            return None
        candidates = [(key, record) for key, record in self._motif_records.items()
                      if record["uid"] == int(selected[0])]
        return next((item for item in candidates if item[0] == self._motif_focus_key),
                    candidates[0] if candidates else None)

    def focus_motif_residue(self, selector):
        """Focus an existing copied residue without modifying the task payload."""
        matched = self._motif_record(selector)
        if matched is None:
            return False
        key, _ = matched
        self._show_motif_residue(key, notify=True)
        return True

    def toggle_motif_atoms(self, selector, names):
        """Toggle one protein atom or a same-residue bond's atom pair as a batch.

        The first click on an all-atom residue starts a custom selection. An
        empty custom selection returns to all observed atoms. Unsupported
        residues return False; unresolved atom names are rejected atomically.
        """
        matched = self._motif_record(selector)
        if matched is None:
            return False
        (chain, _), record = matched
        if record["modality"] != "protein":
            return False
        if isinstance(names, str) or not isinstance(names, (list, tuple, set)) or not names:
            raise ValueError("Choose at least one observed protein atom.")
        if not all(isinstance(name, str) and name for name in names):
            raise ValueError("Protein atom names must be non-empty strings.")
        clicked = set(names)
        observed = record["available_atoms"]
        missing = clicked.difference(observed)
        if missing:
            raise ValueError(f"These atoms in {record['selector']} are missing or unresolved: "
                             + ", ".join(sorted(missing)))
        entry = self._payload["condition"]["contig"][chain]
        _, current = self._matching_override(entry.get("polymer_atom"), record)
        existing = backend.motif_atom_names(self.atoms, record["selector"], current)
        selected = set(existing or ())
        if selected == set(observed):
            selected.clear()
        if clicked.issubset(selected):
            selected.difference_update(clicked)
        else:
            selected.update(clicked)
        self._motif_focus_key = (chain, record["selector"])
        self._set_atoms(chain, record, [name for name in observed if name in selected])
        return True

    def _atom_editor(self, chain, record, mapping):
        selector, current = self._matching_override(mapping, record)
        observed = record["available_atoms"]
        title = f"{record['selector']} · {record['name']}"
        children = []
        kwargs = {"style": {"description_width": "94px"}, "layout": _layout()}
        if record["modality"] == "protein":
            try:
                names = backend.motif_atom_names(self.atoms, record["selector"], current)
            except ValueError as exc:
                names = None
                children.append(self._readout("Native atoms", str(current)))
                children.append(_note(str(exc)))
            selected = tuple(names or ())
            picker = W.SelectMultiple(options=observed, value=selected, rows=min(5, max(3, len(observed))),
                                      description="Atoms", **kwargs)
            self.controls[("motif_atoms", chain, record["selector"])] = picker
            picker.observe(lambda change: self._set_atoms(chain, record, list(change["new"])), names="value")
            children.append(picker)
            if current == "tip":
                children.append(_note("Native selection: side-chain tip."))
            children.append(_note("An empty selection keeps all observed atoms."))
        else:
            group = record["atom_choices"].get("sugar_base", {})
            is_full = current is None or current == "full" or current == "all" or current == []
            if isinstance(current, list) and set(current) == set(observed):
                is_full = True
            is_group = (isinstance(current, list) and group.get("enabled")
                        and set(current) == set(group.get("atoms") or []))
            mode = "full" if is_full else "sugar_base" if is_group else "legacy"
            options = [("All observed atoms", "full")]
            if group.get("enabled"):
                options.append(("Base and non-backbone atoms", "sugar_base"))
            if mode == "legacy":
                options.insert(0, ("Keep existing selection", "legacy"))
                children.append(self._readout("Legacy atoms", ", ".join(current) if isinstance(current, list) else str(current)))
                children.append(_note("The native selection is retained. Choose a supported group to replace it."))
            picker = W.Dropdown(options=options, value=mode, description="Atoms", **kwargs)
            self.controls[("motif_atoms", chain, record["selector"])] = picker

            def choose(change):
                if change["new"] != "legacy":
                    self._set_atoms(chain, record, change["new"])

            picker.observe(choose, names="value")
            children.append(picker)
            children.append(self._readout("Observed", ", ".join(observed)))
            if group.get("missing_atoms"):
                children.append(_note("Unresolved group atoms: " + ", ".join(group["missing_atoms"])))
        box = _section(title, children)
        box.add_class("pd-native-residue-atoms")
        return box

    def _motif_residue_label(self, key):
        chain, _ = key
        record = self._motif_records[key]
        entry = self._payload["condition"]["contig"][chain]
        _, current = self._matching_override(entry.get("polymer_atom"), record)
        if current is None or current == "full" or current == "all" or current == []:
            count = None
        elif isinstance(current, list):
            count = None if set(current) == set(record["available_atoms"]) else len(current)
        else:
            try:
                selected = backend.motif_atom_names(self.atoms, record["selector"], current)
            except ValueError:
                return f"{record['selector']} · {record['name']} · Native selection"
            count = len(selected) if selected is not None else None
        summary = "All atoms" if count is None else f"{count} selected"
        return f"{record['selector']} · {record['name']} · {summary}"

    def _show_motif_residue(self, key, *, notify=False):
        changed = key != self._motif_focus_key
        self._motif_focus_key = key
        self._motif_display_syncing = True
        try:
            chain = key[0] if key is not None else self.motif_chain.value
            if self.motif_chain.value != chain:
                self.motif_chain.value = chain
            self._motif_contig_box.children = self._motif_chain_panels.get(chain, ())
            keys = [item for item in self._motif_records if item[0] == chain]
            options = [(self._motif_residue_label(item), item) for item in keys]
            self.motif_residues.options = options
            self.motif_residues.value = key
            self._motif_detail.children = ((self._motif_editors[key],) if key is not None
                                           else (_note("No copied polymer residues are selected."),))
        finally:
            self._motif_display_syncing = False
        if changed and notify and key is not None and self.on_motif_focus is not None:
            self.on_motif_focus(self._motif_records[key]["selector"])

    def _render_motif(self):
        for path in list(self.controls):
            if path[:2] == ("condition", "contig") or path[:1] == ("motif_atoms",):
                del self.controls[path]
        self._motif_records = {}
        self._motif_editors = {}
        self._motif_chain_panels = {}
        contigs = (self._payload.get("condition") or {}).get("contig")
        if isinstance(contigs, Mapping):
            for chain, entry in contigs.items():
                path = ("condition", "contig", chain)
                if not isinstance(entry, Mapping):
                    self._motif_chain_panels[chain] = (self._editor(path, entry, "Contig"),)
                    continue
                children = []
                if "token" in entry:
                    children.append(self._scalar((*path, "token"), entry["token"], "Contig",
                                                 single_line=True))
                limits = []
                for key, title in (("motif_min_length", "Min"), ("motif_max_length", "Max")):
                    if key in entry:
                        control = self._scalar((*path, key), entry[key], title)
                        control.layout.width = "auto"
                        control.layout.flex = "1 1 0"
                        control.style.description_width = "32px"
                        limits.append(control)
                if limits:
                    limit_row = W.HBox(limits, layout=_layout(flex_flow="row nowrap", grid_gap="10px"))
                    limit_row.add_class("pd-native-motif-length-row")
                    children.append(limit_row)
                children.extend(self._editor((*path, key), value, _label(key))
                                for key, value in entry.items() if key not in {
                                    "token", "polymer_atom", "motif_min_length", "motif_max_length"})
                if self.atoms is None:
                    children.append(_note("Load source atoms to choose Motif residue atoms."))
                    if "polymer_atom" in entry:
                        children.append(self._editor((*path, "polymer_atom"), entry["polymer_atom"], readonly=True))
                else:
                    try:
                        info = self._inspect(chain, entry.get("token", ""))
                    except (ValueError, TypeError, AttributeError) as exc:
                        children.append(_note(f"Contig atom inspection: {exc}"))
                        if "polymer_atom" in entry:
                            children.append(self._editor((*path, "polymer_atom"), entry["polymer_atom"], readonly=True))
                    else:
                        children.append(_note(f"{info['copied_count']} copied residues · "
                            f"{info['minimum_length']}–{info['maximum_length']} total residues"))
                        for record in info["residues"]:
                            key = (chain, record["selector"])
                            self._motif_records[key] = record
                            self._motif_editors[key] = self._atom_editor(chain, record, entry.get("polymer_atom"))
                self._motif_chain_panels[chain] = tuple(children)
        chains = tuple(self._motif_chain_panels)
        self.motif_chain = W.Dropdown(options=chains, description="Design chain",
            style={"description_width": "94px"}, layout=_layout())
        self.motif_chain.add_class("pd-native-control")
        self.motif_chain.layout.display = "" if len(chains) > 1 else "none"
        self._motif_contig_box = _column([], "pd-native-motif-contig")
        self.motif_residues = W.Select(options=(), rows=5, layout=_layout())
        self.motif_residues.add_class("pd-native-motif-list")
        residue_list = _column([self.motif_residues], "pd-native-motif-list-column")
        residue_list.layout.flex = "1 1 200px"
        self._motif_detail = _column([], "pd-native-motif-detail")
        self._motif_detail.layout.flex = "1 1 240px"
        browser = W.HBox([residue_list, self._motif_detail],
            layout=_layout(flex_flow="row wrap", align_items="flex-start", grid_gap="10px"))
        browser.add_class("pd-native-motif-browser")

        def choose_chain(change):
            if self._motif_display_syncing:
                return
            keys = [key for key in self._motif_records if key[0] == change["new"]]
            self._show_motif_residue(keys[0] if keys else None, notify=True)

        def choose_residue(change):
            if not self._motif_display_syncing and change["new"] is not None:
                self._show_motif_residue(change["new"], notify=True)

        self.motif_chain.observe(choose_chain, names="value")
        self.motif_residues.observe(choose_residue, names="value")
        key = self._motif_focus_key
        if key not in self._motif_records:
            key = next(iter(self._motif_records), None)
        self._show_motif_residue(key)
        contig_children = ([self.motif_chain] if len(chains) > 1 else [])
        contig_children.append(self._motif_contig_box if chains else
                               self._readout("Contig", "No copied polymer motif"))
        selection = "No reference structure loaded."
        if self.atoms is not None:
            try:
                source_chains = {record["chain"] for record in self._motif_records.values()}
                selection = " · ".join(f"{row['chain']}: {row['modality']}, {row['residues']} residues"
                    for row in backend.chain_summary(self.atoms)
                    if not source_chains or row["chain"] in source_chains)
            except (ValueError, AttributeError) as exc:
                selection = f"Source inventory unavailable: {exc}"
        contig_title = f"Contig · {chains[0]}" if len(chains) == 1 else "Contig"
        self.panels[1].children = (
            _section("Selection", [_note(selection)]),
            _section(contig_title, contig_children),
            _section("Atoms", [browser]),
        )

    def _render_hotspot(self):
        children = []
        if "hotspot" in self._payload:
            children.append(self._editor(("hotspot",), self._payload["hotspot"], "Selectors"))
        antigen = self._payload.get("antigen")
        if isinstance(antigen, Mapping) and "hotspot" in antigen:
            children.append(self._editor(("antigen", "hotspot"), antigen["hotspot"], "Selectors"))
        if not children:
            children.append(self._readout("Selectors", "No hotspots specified"))
        children.append(_note("Selectors use the loaded source chain and residue numbering."))
        self.panels[2].children = (_section("Hotspots", children),)

    def _render_lengths(self):
        lengths = self._payload["design_length"]
        chain = self._distribution_chain(self._payload)
        if (chain is None or not isinstance(lengths, Mapping)
                or set(lengths) != {chain}):
            return self._editor(("design_length",), lengths)
        value = lengths[chain]
        mode = W.Dropdown(description="Length mode", options=[
            ("Specify lengths", "explicit"), ("From distribution", "from_distribution")],
            value="from_distribution" if value == "from_distribution" else "explicit",
            style={"description_width": "94px"}, layout=_layout())
        mode.add_class("pd-native-control")
        self.controls[("length_mode", chain)] = mode

        def choose(change):
            if self._syncing:
                return
            self._length_error.value = ""
            self._length_error.layout.display = "none"
            current = self._payload["design_length"][chain]
            if change["new"] == "from_distribution":
                self._explicit_lengths[chain] = deepcopy(current)
                new = "from_distribution"
            else:
                new = self._explicit_lengths.get(chain, [20])
            self._update(("design_length", chain), new)
            self._render_advanced()

        mode.observe(choose, names="value")
        if value == "from_distribution":
            field = self._scalar(("design_length", chain), value, "Atoms", readonly=True)
            field.layout.display = "none"
            children = [mode]
        else:
            children = [mode, self._editor(("design_length", chain), value, "Atoms")]
        return _section("Design lengths", children)

    def _render_advanced(self):
        for path in list(self.controls):
            if path[:1] in {("design_length",), ("length_mode",)}:
                del self.controls[path]
        sections = []
        if "design_length" in self._payload:
            sections.append(self._render_lengths())
        condition = self._payload.get("condition")
        if isinstance(condition, Mapping):
            for key, value in condition.items():
                if key != "contig":
                    sections.append(self._editor(("condition", key), value, _label(key)))
        elif "condition" in self._payload:
            sections.append(self._editor(("condition",), condition))
        antibody = self._payload.get("antibody")
        if isinstance(antibody, Mapping):
            for key, value in antibody.items():
                if key not in {"scaffold", "chains", "template"}:
                    sections.append(self._editor(("antibody", key), value, "Antibody " + _label(key).lower()))
        owned = {"input", "sample_num", "format", "design_chains", "crop", "antigen",
                 "antibody", "condition", "hotspot", "design_length", "score", "postprocess", "override"}
        sections.extend(self._editor((key,), value) for key, value in self._payload.items() if key not in owned)
        if not sections:
            sections.append(_note("No additional native conditions are specified."))
        sections.append(self._length_error)
        self.panels[3].children = tuple(sections)

    def _render_save(self):
        children = []
        for key, default in (("score", False), ("postprocess", False)):
            children.append(self._scalar((key,), self._payload.get(key, default)))
        self.panels[4].children = (_section("Task output settings", children),)


__all__ = ["NativeTaskForm"]
