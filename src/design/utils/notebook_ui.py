"""Widget front end for the first PartiDesign authoring notebook.

Prepare source structures and task files without starting inference, loading a
checkpoint, or inventing coordinates for a free design.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import re
import shlex
import tempfile
from typing import Any
import traitlets
import anywidget

import ipywidgets as W
import numpy as np
import yaml
from IPython.display import display
from ipymolstar import PDBeMolstar

from ..core.crop import source_selector
from . import notebook as backend
from .notebook_presets_ui import NativeTaskForm
from .notebook_help import HelpTooltip
from .notebook_runtime_ui import NotebookRuntimePanel


_MODALITY_NAMES = {"protein": "Protein", "peptide": "Peptide", "rna": "RNA", "dna": "DNA", "ligand": "Ligand"}
_PICK_NAMES = {"chain": "Chain", "residue": "Residue", "atom": "Atom"}
_LIGAND_MODES = {"free": "New ligand", "smiles": "Use SMILES", "ccd": "CCD component", "type_bond": "Source chemistry, new pose", "backbone": "Source pose, new chemistry", "atom_motif": "Source atom motif"}
_COARSE_SS = {"protein": [("None", ""), ("Alpha helix", "alpha"), ("Beta structure", "beta")],
              "rna": [("None", ""), ("Paired", "paired"), ("Unpaired", "unpaired")]}


# These colors describe the categorical B values written by PartiDesign.
_CATEGORY_COLORS = {
    0: ("Context", "#d8dfe8"),
    20: ("Free design", "#b8cdeb"),
    40: ("Fixed identity", "#9867cf"),
    60: ("Fixed centers", "#24b7b5"),
    80: ("Motif atoms", "#eea13b"),
    100: ("Hotspot", "#df5353"),
}


def _rgb(color: str) -> dict[str, int]:
    return dict(zip(("r", "g", "b"), (int(color[i:i + 2], 16) for i in (1, 3, 5))))


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def _yaml(text: str) -> dict:
    try:
        value = yaml.load(text or "{}", Loader=_UniqueLoader)
    except (yaml.YAMLError, TypeError) as exc:
        raise ValueError("Cannot read YAML. Check indentation, brackets, and keys.") from exc
    if not isinstance(value, dict):
        raise ValueError("YAML must contain a mapping.")
    return value


def _merge(left: dict, right: dict) -> dict:
    result = deepcopy(left)
    for key, value in right.items():
        result[key] = (
            _merge(result[key], value)
            if isinstance(result.get(key), dict) and isinstance(value, dict)
            else deepcopy(value)
        )
    return result


def _chain_selector(chain: str) -> str:
    return chain if re.fullmatch(r"[A-Za-z]", chain) else f"[{chain}]"


def _tokens(text: str) -> list[str]:
    return [part for part in re.split(r"[,\s]+", text.strip()) if part]



class _NotebookMolstar(PDBeMolstar):
    _esm = Path(__file__).with_name("notebook_molstar.js")
    viewer_error = traitlets.Unicode("").tag(sync=True)
    selection_scope = traitlets.Enum(["chain", "residue", "atom"], default_value="residue").tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Keep MolStar's base rules, then scope the workbench skin to our roots.
        self._css = self._css.replace("pdbe-molstar@3.2.0", "pdbe-molstar@3.3.2") + "\n" + Path(__file__).with_name("notebook_theme.css").read_text()


class _PresetLoadButton(anywidget.AnyWidget):
    _esm = Path(__file__).with_name("notebook_preset_load.js")
    entries = traitlets.List(traitlets.Dict()).tag(sync=True)
    request = traitlets.Dict().tag(sync=True)
    ack = traitlets.Unicode("").tag(sync=True)

class NotebookTaskEditor:
    """A current task draft, with separate source, motif and hotspot selections."""

    def __init__(self, repo_root=None, work_dir="configs/customs", *, preset=None):
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[3]).expanduser().resolve()
        self.work_dir = self._path(work_dir)
        self.raw_dir = self.work_dir / "raw"
        self.input_file = None
        self._raw_source = None
        self._source_filter_error = None
        self._source_directory = None
        self._contact_records = []
        self.atoms = None
        self.display_mode = "empty"
        self.last_preview = None
        self._syncing_controls = True
        self._roles = {}
        self._role_controls = {}
        self._baskets = {key: OrderedDict() for key in ("motif", "hotspot")}
        self._motif_rows = []
        self._focused_motif_selector = None
        self._pending_atom_choices = {}
        self.motif_atom_controls = {}
        self.motif_condition = {}
        self._motif_chain = None
        self._contig_applied = ""
        self._imported_task = None
        self._import_pending = False
        self._draft_error = None
        self._result_colors = []
        self._building_yaml = False
        self._area = "basic"
        self._form_widgets = []
        self._form_extras = {}
        self._native_form = None
        self._native_panels = None
        self._loaded_preset_key = None
        self.example_tasks = [entry for entry in backend.list_example_tasks(self.repo_root)
            if entry["payload"].get("format", "normal") != "antibody"]
        self._example_by_key = {entry["key"]: entry for entry in self.example_tasks}
        self._make_controls()
        self.view = _NotebookMolstar(theme="light", bg_color="#eceff4", height="100%", width="100%",
            subscribe_events=True, click_focus=False, granularity="element",
            hide_water=True, hide_controls=False, sequence_panel=True,
            pdbe_link=False, loading_overlay=True, reactive=True)
        self.view.observe(self._on_click, names="click_event")
        self.view.observe(self._scope_changed, names="selection_scope")
        self.view.observe(lambda ch: self._status("Could not display the structure: " + ch["new"]) if ch["new"] else None, names="viewer_error")
        self.view.add_class("pd-molstar")
        self.view.layout = W.Layout(width="100%", min_width="0", min_height="0", flex="1 1 0")
        self.viewer_note = self._info("No input structure")
        self.viewer_note.add_class("pd-view-caption")
        self.viewer_header = W.Box([self.viewer_note], layout=W.Layout(width="100%", min_width="0"))
        self.viewer_header.add_class("pd-view-header")
        self.viewer_box = W.VBox([self.viewer_header, self.view])
        self.status = self._info("Choose an input and design modality.")
        self.summary = W.Label()
        self.summary_help = self._help("Current task details")
        self.summary.observe(lambda change: setattr(self.summary_help, "help_text", change["new"]), names="value")
        self.log = W.Output(layout=W.Layout(max_height="140px", overflow="auto"))
        self.steps = W.ToggleButtons(options=[("1  Input & design", 0), ("2  Motif", 1), ("3  Hotspot", 2), ("4  Advanced conditions", 3), ("5  Save", 4), ("6  Runtime", 5)], value=0,
            style={"button_width": "auto"}, layout=W.Layout(width="100%", margin="0"))
        self.steps.observe(self._step_changed, names="value")
        self.runtime = NotebookRuntimePanel(self)
        self.active_panel = W.VBox([self.source_panel])
        self.basic_row = W.Box([self.active_panel, self.viewer_box])
        self.basic_row.add_class("pd-workspace")
        self.active_panel.add_class("pd-form-card")
        self.viewer_box.add_class("pd-viewer")
        for card in (self.active_panel, self.viewer_box):
            card.add_class("pd-card")
        self.steps.add_class("pd-steps")
        self.footer = W.Box([self._button("Back", lambda: self._next_step(-1)), self._button("Next", lambda: self._next_step(1), "primary"), self.status, self.summary_help])
        self.footer.add_class("pd-footer")
        self.widget = W.VBox([self._header("PartiDesign · Design setup"), self.steps, self.basic_row, self.footer])
        self.widget.add_class("pd-workbench")
        self.widget.add_class("pd-nord-light")
        self._install_reactivity()
        self._syncing_controls = False
        self.refresh_controls()
        self._refresh_draft()
        self.runtime.sync_task(reset_sources=True)
        if preset is not None:
            self.load_preset(preset)

    def _path(self, path):
        value = Path(path).expanduser()
        return value.resolve() if value.is_absolute() else (self.repo_root / value).resolve()

    @staticmethod
    def _text(value="", description="", **kwargs):
        return W.Text(value=str(value), description=description, continuous_update=False,
            style={"description_width": "94px"}, layout=W.Layout(width="100%", min_width="0", margin="2px 0"), **kwargs)

    @staticmethod
    def _area_text(value="", rows=4, disabled=False):
        return W.Textarea(value=value, rows=rows, disabled=disabled, continuous_update=False, layout=W.Layout(width="100%", min_width="0", margin="2px 0"))

    @staticmethod
    def _note(value="", rows=2):
        return NotebookTaskEditor._info(value)

    @staticmethod
    def _info(value=""):
        note = W.Label(value=value, layout=W.Layout(width="100%", min_width="0", margin="0"))
        note.add_class("pd-inline-info")
        note.tooltip = value
        note.observe(lambda change: setattr(note, "tooltip", change["new"]), names="value")
        return note

    @staticmethod
    def _help(text):
        help_button = HelpTooltip(tooltip=text,
            layout=W.Layout(width="22px", min_width="22px", height="22px", margin="0"))
        help_button.add_class("pd-help")
        return help_button

    @staticmethod
    def _actions(*controls):
        for control in controls:
            if isinstance(control, W.Label):
                control.layout.width = "auto"
                control.layout.flex = "1 1 0"
        row = W.Box(list(controls))
        row.add_class("pd-actions")
        return row

    @staticmethod
    def _heading(value, *, small=False):
        heading = W.Label(value, style={"font_weight": "600", "font_size": "13px" if small else "21px"},
            layout=W.Layout(width="100%", min_width="0", margin="2px 0 6px 0"))
        heading.add_class("pd-section-title" if small else "pd-title")
        return heading

    @staticmethod
    def _checkbox(text, value=False):
        control = W.Checkbox(value=value, description=text, indent=False,
            layout=W.Layout(width="auto", min_width="0", margin="0", align_items="center"))
        control.add_class("pd-check")
        return control

    def _header(self, title):
        header = W.Box([self._heading(title)])
        header.add_class("pd-header")
        return header

    def _button(self, text, action, style=""):
        button = W.Button(description=text, button_style=style, style={"font_weight": "500"}, layout=W.Layout(width="auto", min_width="100px", margin="4px 6px 4px 0", padding="0 10px"))
        def clicked(_):
            try:
                action()
            except Exception as exc:
                self._status(f"Please check: {exc}")
        button.on_click(clicked)
        return button

    @staticmethod
    def _section(title, *children):
        section = W.VBox([NotebookTaskEditor._heading(title, small=True), *children], layout=W.Layout(padding="10px", margin="0 0 8px 0", border="1px solid #e1e6ee", width="100%", grid_gap="4px"))
        section.add_class("pd-section")
        return section

    def _status(self, text):
        self.status.value = str(text)
        self.status.tooltip = str(text)

    def _make_controls(self):
        self.preset_group = W.Dropdown(options=["All", *dict.fromkeys(entry["group"] for entry in self.example_tasks)], value="All", description="Category", style={"description_width": "94px"}, layout=W.Layout(width="100%", min_width="0", margin="2px 0"))
        self.preset = W.Dropdown(options=[(entry["label"], entry["key"]) for entry in self.example_tasks], description="Preset", style={"description_width": "94px"}, layout=W.Layout(width="100%", min_width="0", margin="2px 0"))
        self.task_kind = self._info("")
        self.loaded_preset = self._info("Custom task")
        self.preset_load = _PresetLoadButton(entries=[{"key": entry["key"], "label": entry["label"]} for entry in self.example_tasks],
            layout=W.Layout(width="auto", min_width="100px"))
        self.preset_load.observe(self._preset_load_requested, names="request")
        self.preset_panel = W.VBox([
            self._actions(self._heading("Task preset", small=True), self._help("Choose a complete task from configs/examples. Selection loads its input, design, Motif, Hotspot and advanced conditions together. Load preset restores the selected example.")),
            self.preset_group, self._actions(self.preset, self.preset_load), self.loaded_preset,
        ], layout=W.Layout(width="100%", min_width="0", grid_gap="4px"))
        self.preset_panel.add_class("pd-task-preset")
        self.task_name = self._text("my_design", "Task name")
        self.modality = W.Dropdown(options=[(label, value) for value, label in _MODALITY_NAMES.items()], value="protein", description="Design modality", style={"description_width": "94px"}, layout=W.Layout(width="100%", min_width="0", margin="2px 0"))
        self.design_chain = self._text("P", "Design chain")
        self.source_path = self._text("", "Structure file", placeholder="Path to a PDB or CIF file")
        self.rcsb_id = self._text("", "PDB ID", placeholder="e.g. 1CRN")
        self.assembly_id = self._text("", "Assembly", placeholder="Optional; blank uses the deposited structure")
        self.input_upload = W.FileUpload(accept=".pdb,.cif,.mmcif", multiple=False, description="Choose file", layout=W.Layout(width="auto", max_width="100%", min_width="0", margin="2px 0"))
        self.keep_chains = self._text("", "Keep chains", placeholder="All if blank; e.g. A,B")
        self.exclude_names = self._text("HOH,WAT,DOD", "Remove")
        self.clean_path = self._text("", "Input file")
        self.input_dir = self._text(self.raw_dir, "Source folder")
        self.chain_table = W.VBox([self._note("No structure loaded. De novo design is also available.")])
        self.source_method = W.Tab(children=[W.VBox([self.rcsb_id, self.assembly_id, self._button("Fetch structure", self.fetch_source, "primary")]),
            W.VBox([self.source_path, self._actions(self._button("Load file", lambda: self.load_source(self.source_path.value), "primary"), self.input_upload, self._button("Use upload", self.load_upload))]),
            W.VBox([self._actions(self._button("Use de novo design", self.clear_source), self._help("Generate without source coordinates. The viewer stays available for structures and results."))])])
        for i, text in enumerate(["RCSB", "File", "De novo"]):
            self.source_method.set_title(i, text)
        cleaning = W.Accordion(children=[W.VBox([self.exclude_names])], selected_index=None)
        cleaning.set_title(0, "Clean input")
        self.source_panel = W.VBox([self.preset_panel, self.task_name, self.modality, self.design_chain, self.source_method, self.keep_chains, cleaning,
            self._actions(self._heading("Chain roles", small=True), self._help("Context enters the task. Reference supplies selectable Motif geometry or automatic Hotspot contacts. Unused Reference chains are excluded when saving.")), self.chain_table])
        self.motif_enabled = self._checkbox("Use a motif")
        self.motif_source = W.Dropdown(options=[], description="Reference chain", style={"description_width": "94px"}, layout=W.Layout(width="100%", min_width="0", margin="2px 0"))
        self.motif_selector = self._text("", "Residue range", placeholder="e.g. B12-20/B35")
        self.hotspot_selector = self._text("", "Residue range", placeholder="e.g. A42/A45-47")
        self.motif_picks = W.SelectMultiple(options=[], rows=4, layout=W.Layout(width="100%", min_width="0", margin="2px 0"))
        self.hotspot_picks = W.SelectMultiple(options=[], rows=4, layout=W.Layout(width="100%", min_width="0", margin="2px 0"))
        # Shared state alias for code using the old control. The visible scope
        # selector lives in the viewer and is independent of task modality.
        self.pick_level = W.ToggleButtons(options=[("Chain", "chain"), ("Residue", "residue"), ("Atom", "atom")], value="residue", description="Select scope")
        self.pick_level.add_class("pd-pick-level")
        self.click_info = self._note("Click residues in the structure, or enter a range.")
        self.motif_contig = self._text("", "Contig", placeholder="e.g. 10-20/B12-20/15")
        self.contig_summary = self._note("Select motif residues first.")
        self.motif_atoms_box = W.VBox([self._info("Apply motif residues to choose their atoms.")])
        self.mark_summary = self._area_text("No motif selected.", rows=2, disabled=True)
        self.ligand_motif_atoms = W.SelectMultiple(options=[], rows=6, description="Keep atoms", style={"description_width": "94px"}, layout=W.Layout(width="100%", min_width="0", margin="2px 0"))
        self.motif_picks.rows = 6
        residue_column = W.VBox([self._heading("Selected residues", small=True), self.motif_picks], layout=W.Layout(flex="1 1 150px", min_width="0", width="auto"))
        atom_column = W.VBox([self._heading("Atoms", small=True), self.motif_atoms_box], layout=W.Layout(flex="1 1 170px", min_width="0", width="auto"))
        browser = W.Box([residue_column, atom_column], layout=W.Layout(width="100%", min_width="0", flex_flow="row wrap", grid_gap="10px"))
        browser.add_class("pd-motif-browser")
        self.motif_tools = W.VBox([self.motif_source,
            self._actions(self.motif_selector, self._button("Add", lambda: self.add_batch("motif"))), browser,
            self._actions(self._button("Remove", lambda: self.remove_picks("motif")), self._button("Clear", lambda: self.clear_picks("motif")), self._button("Apply residues", self.mark_motif, "primary")),
            self._actions(self.motif_contig, self._button("Apply contig", self.apply_contig), self._help("A Contig combines new lengths and source segments, e.g. 10/B2/15. Residue identity is retained; atom choices restrict coordinate conditions only.")), self.contig_summary])
        self.motif_tools.add_class("pd-motif-tools")
        self.ligand_motif_tools = W.VBox([self._actions(self._heading("Source atom motif", small=True), self._help("Click ligand atoms or select their names below. An empty restriction retains all observed source atoms.")), self.ligand_motif_atoms])
        self.motif_panel = W.VBox([self._actions(self.motif_enabled, self._help("Motif is optional. Selecting residues keeps all observed atoms by default.")), self.motif_tools, self.ligand_motif_tools])
        self.hotspot_selectors = self._text("", "Applied hotspots")
        self.hotspot_selectors.disabled = True
        self.target_selectors = self._text("", "Context")
        self.target_selectors.disabled = True
        manual_hotspots = W.VBox([self._actions(self._heading("Context contacts", small=True), self._help("Click Context residues to select hotspots. Click again to deselect. The pending list only becomes a task condition after Apply hotspots.")), self.hotspot_selector,
            self._button("Add range", lambda: self.add_batch("hotspot")), self.hotspot_picks,
            self._actions(self._button("Remove selected", lambda: self.remove_picks("hotspot")), self._button("Clear selection", lambda: self.clear_picks("hotspot"))),
            self._button("Apply hotspots", self.mark_hotspot, "primary")])
        self.contact_references = W.SelectMultiple(options=[], rows=3, description="Reference", style={"description_width": "94px"}, layout=W.Layout(width="100%", min_width="0", margin="2px 0"))
        self.contact_cutoff = W.FloatText(value=6.0, description="Cutoff (Å)", style={"description_width": "94px"}, layout=W.Layout(width="100%", min_width="0", margin="2px 0"))
        self.contact_candidates = W.SelectMultiple(options=[], rows=6, layout=W.Layout(width="100%", min_width="0", margin="2px 0"))
        self.contact_note = self._note("Choose Reference chains to find nearby Context residues. Motif selection is independent.", rows=3)
        automatic_hotspots = W.VBox([self._actions(self._heading("Reference contacts", small=True), self._help("Use all resolved heavy atoms in selected Reference chains, independently of Motif. Preview Context contacts within the cutoff, then merge or replace the applied hotspots.")), self.contact_references, self.contact_cutoff, self.contact_note,
            self._button("Find contacts", self.find_hotspots, "primary"), self.contact_candidates,
            W.HBox([self._button("Merge hotspots", lambda: self.apply_contact_hotspots(merge=True)),
                    self._button("Replace hotspots", lambda: self.apply_contact_hotspots(merge=False))])])
        self.hotspot_tabs = W.Tab(children=[manual_hotspots, automatic_hotspots])
        self.hotspot_tabs.set_title(0, "Pick hotspots")
        self.hotspot_tabs.set_title(1, "From Reference")
        self.hotspot_panel = W.VBox([self.hotspot_tabs, self.hotspot_selectors,
            self._button("Clear hotspots", self.clear_hotspots)])
        self.ligand_length_mode = W.Dropdown(options=[("Explicit count / range", "explicit"), ("From distribution", "distribution")], value="explicit", description="Length plan", style={"description_width": "94px"}, layout=W.Layout(width="100%", min_width="0", margin="2px 0"))
        self.length_plan = self._text("", "Design length", placeholder="e.g. 60; lengths 40,60; range 40:100:20")
        self.length_hint = self._note("Set the length, then check and save the task.")
        self.samples = W.IntText(value=1, description="Samples / length", style={"description_width": "94px"}, layout=W.Layout(width="100%", min_width="0", margin="2px 0"))
        self.score = self._checkbox("Calculate PartiScore")
        self.postprocess = self._checkbox("Relax structures")
        self.override = self._checkbox("Keep relaxed structures", value=True)
        self.ligand_mode = W.Dropdown(options=[(label, value) for value, label in _LIGAND_MODES.items()], value="free", description="Ligand mode", style={"description_width": "94px"}, layout=W.Layout(width="100%", min_width="0", margin="2px 0"))
        self.ligand_source = W.Dropdown(options=[], description="Source ligand", style={"description_width": "94px"}, layout=W.Layout(width="100%", min_width="0", margin="2px 0"))
        self.ligand_value = self._text("", "SMILES / CCD")
        self.ligand_hint = self._note()
        self.ligand_box = W.VBox([self.ligand_mode, self.ligand_source, self.ligand_value, self.ligand_hint])
        self.source_panel.children = (*self.source_panel.children, self.ligand_box)
        self.task_yaml = self._area_text(rows=10, disabled=True)
        self.yaml_status = self._note("The task updates with your settings.")
        self.task_file = self._text(self.work_dir / "tasks" / "my_design.yaml", "Task file")
        self.output_dir = self._text(self.work_dir / "inference", "Output root")
        self.command = self._area_text(rows=3, disabled=True)
        self.command.layout.display = "none"
        self.command.observe(lambda change: setattr(self.command.layout, "display", "" if change["new"] else "none"), names="value")
        self.score_options = W.Box([self.score, self.postprocess], layout=W.Layout(display="flex", flex_flow="row wrap", align_items="center", justify_content="flex-start", width="100%", min_width="0", grid_gap="10px", margin="8px 0"))
        self.score_options.add_class("pd-options")
        self.review_panel = W.VBox([self.ligand_length_mode, self.length_plan, self.samples,
            self._actions(self._help("Explicit counts/ranges define the generated size. From distribution samples ligand heavy-atom counts from the Context pocket. Contig, sequence or source chemistry may determine length instead."), self.length_hint),
            self.score_options, self._actions(self._button("Check task", self.check_task, "primary"), self._help("Compile representative length plans and show supplied conditions. No inference or sampling starts. Checking never applies pending Motif or Hotspot selections.")), self.task_file,
            self._button("Save task & command", self.export_task, "success"), self.command])
        self.fixed_sequence = self._text("", "Full sequence")
        self.secondary_structure = W.Dropdown(options=_COARSE_SS["protein"], description="Global SS", style={"description_width": "94px"}, layout=W.Layout(width="100%", min_width="0", margin="2px 0"))
        self.token_ss_yaml = self._area_text("{}", 4)
        self.pair_ss_yaml = self._area_text("{}", 4)
        self.ss_hint = self._note()
        self.backbone = self._checkbox("Keep source backbone centers")
        self.backbone_source = W.Dropdown(options=[], description="Backbone chain", style={"description_width": "94px"}, layout=W.Layout(width="100%", min_width="0", margin="2px 0"))
        self.backbone_hint = self._note()
        self.cyclic_offset = self._checkbox("Cyclic offset")
        self.condition_yaml = self._area_text("{}", 7)
        self.context_mode = W.Dropdown(options=[("Auto", "auto"), ("Rigid", "rigid"), ("Flexible", "flex")], value="auto", description="Context mode", style={"description_width": "94px"}, layout=W.Layout(width="100%", min_width="0", margin="2px 0"))
        self.checkpoint = self._text("", "Checkpoint", placeholder="Optional; use the default checkpoint")
        self.preview_dir = self._text(self.repo_root / ".cache" / "notebook_user_v3" / "preview", "Preview")
        self.import_yaml = self._area_text("", 13)
        self.import_status = self._note("Paste a full task YAML, then apply it.")
        self.bond_yaml = self._area_text("[]", 6)
        bonded = [entry for entry in self.example_tasks if (entry["payload"].get("condition") or {}).get("bond")]
        self.bond_example = W.Dropdown(options=[(entry["label"], entry["key"]) for entry in bonded], description="Example", style={"description_width": "94px"}, layout=W.Layout(width="100%", min_width="0", margin="2px 0"))
        self.bond_example_text = self._area_text(rows=5, disabled=True)
        bond_tools = self._actions(self.bond_example, self._button("Load task", lambda: self.load_preset(self.bond_example.value)))
        bond_tools.add_class("pd-condition-tools")
        bonds = W.VBox([self._actions(self.cyclic_offset, self._help("Bonds use final design positions. Negative positions count from the end. A complete example also fixes required residue identities and omitted atoms; Load task replaces the current task with that example.")), self.bond_yaml,
            self._heading("Chemical bond examples", small=True), bond_tools, self.bond_example_text])
        ss = W.VBox([self.secondary_structure, self._heading("Per-position secondary structure", small=True), self.token_ss_yaml, self._heading("Pair constraints", small=True), self.pair_ss_yaml,
            self._help("Use final design positions. Protein: helix/sheet/loop and adjacent/not_adjacent. RNA/DNA: paired/unpaired per position plus matching pair endpoints.")])
        details = W.Accordion(children=[W.VBox([self.fixed_sequence, self.backbone_source, self.backbone,
                self._help("A full sequence fixes identity and length. Backbone keeps source center coordinates while redesigning identity. Motif atom coordinates are edited in step 2.")]), ss, bonds,
            W.VBox([self._heading("Additional condition fields", small=True), self.condition_yaml, self._help("Additional native condition fields merge with the visible controls. Use the full task YAML for joint design, antibody layouts, crops and omission settings.")]),
            W.VBox([self.context_mode]),
            W.VBox([self.import_status, self._actions(self._button("Edit current task", self.edit_current_yaml), self._help("Apply edited YAML before checking or saving. Full task mode preserves multichain and antibody settings; Use form task restores your form.")), self.import_yaml,
                self._actions(self._button("Apply YAML", self.apply_yaml, "primary"), self._button("Use form task", self.use_form_task)), self.task_yaml])], selected_index=None)
        for i, title in enumerate(["Sequence & backbone", "Secondary structure & pairing", "Chemical bonds & cyclization", "Other conditions", "Generation options", "Full task YAML"]):
            details.set_title(i, title)
        self.advanced_details = details

        self.legend = W.VBox([self._heading("Condition colors", small=True), *[W.Label(f"{name} · B={value}", layout=W.Layout(border_left=f"5px solid {color}", padding="2px 7px", margin="0")) for value, (name, color) in _CATEGORY_COLORS.items()]])
        self.advanced_panel = W.VBox([details,
            self._actions(self._button("Preview conditions", self.check_task, "primary"), self._button("Source", self.show_source), self._help("Source shows the loaded input. Preview shows only supplied coordinates and exactly the task's applied conditions. Free design regions appear after generation."))])
        for panel in (self.source_panel, self.motif_panel, self.motif_tools, self.hotspot_panel,
                      self.review_panel, self.ligand_box, self.advanced_panel):
            panel.layout.grid_gap = "6px"
            panel.layout.width = "100%"
        self._form_widgets = [self.task_name, self.modality, self.design_chain, self.length_plan, self.ligand_length_mode,
            self.samples, self.fixed_sequence, self.secondary_structure, self.token_ss_yaml, self.pair_ss_yaml,
            self.backbone, self.backbone_source, self.ligand_mode, self.ligand_source, self.ligand_value,
            self.ligand_motif_atoms, self.motif_enabled, self.motif_source, self.motif_contig,
            self.score, self.postprocess, self.override, self.cyclic_offset, self.condition_yaml, self.bond_yaml]

    @property
    def selection(self):
        return self._baskets["hotspot" if self.steps.value == 2 else "motif"]

    @property
    def picks(self):
        return self.hotspot_picks if self.steps.value == 2 else self.motif_picks

    @property
    def selector_text(self):
        return self.hotspot_selector if self.steps.value == 2 else self.motif_selector

    def _install_reactivity(self):
        for control in self._form_widgets:
            control.observe(self._form_changed, names="value")
        self.import_yaml.observe(self._import_changed, names="value")
        self.preset_group.observe(self._preset_group_changed, names="value")
        self.preset.observe(self._preset_changed, names="value")
        self.bond_example.observe(self._bond_example_changed, names="value")
        self._bond_example_changed()
        for control in (self.keep_chains, self.exclude_names):
            control.observe(self._source_filters_changed, names="value")
        for control in (self.contact_references, self.contact_cutoff):
            control.observe(self._contact_options_changed, names="value")
        self.contact_candidates.observe(lambda change: self._highlight_contacts(change["new"]), names="value")
        self.pick_level.observe(self._pick_level_changed, names="value")
        self.context_mode.observe(self._invalidate_check, names="value")
        for control in (self.output_dir, self.checkpoint, self.task_file):
            control.observe(lambda ch: setattr(self.command, "value", ""), names="value")
        self.motif_picks.observe(lambda ch: self._highlight_pick_list("motif", ch["new"]), names="value")
        self.hotspot_picks.observe(lambda ch: self._highlight_pick_list("hotspot", ch["new"]), names="value")

    def _pick_level_changed(self, change):
        if not self._syncing_controls and hasattr(self, "view"):
            self.view.selection_scope = change["new"]

    def _scope_changed(self, change):
        self.pick_level.value = change["new"]
        self._status(f"Select scope: {_PICK_NAMES[change['new']]}. Atom scope also selects bond endpoints.")

    def _next_step(self, delta):
        self.steps.value = max(0, min(5, self.steps.value + delta))

    def _panel_for_step(self, index):
        if index == 5:
            return self.runtime.widget
        if self._imported_task and self._native_form is not None:
            if self._native_panels is None:
                yaml_editor = W.Accordion(children=[self.advanced_details.children[5]], selected_index=None)
                yaml_editor.set_title(0, "Full task YAML")
                self._native_panels = [
                    W.VBox([self.preset_panel, self._native_form.panels[0]], layout=W.Layout(width="100%", min_width="0")),
                    self._native_form.panels[1], self._native_form.panels[2],
                    W.VBox([self._native_form.panels[3], yaml_editor, self.advanced_panel.children[-1]], layout=W.Layout(width="100%", min_width="0")),
                    W.VBox([self._native_form.panels[4], self._actions(
                        self._button("Check task", self.check_task, "primary"),
                        self._button("Save task & command", self.export_task, "success")), self.command],
                        layout=W.Layout(width="100%", min_width="0")),
                ]
            return self._native_panels[index]
        return [self.source_panel, self.motif_panel, self.hotspot_panel, self.advanced_panel, self.review_panel][index]

    def _step_changed(self, change):
        index = change["new"]
        self.active_panel.children = (self._panel_for_step(index),)
        if index in {1, 2} and self.input_file is not None:
            self.show_source()
        elif index < 5 and self.display_mode in {"result", "refold"}:
            self.show_source()
        if index == 2:
            self._status("Select Context residues, then choose Apply hotspots." if self._context_chains() else "No Context selected. You can skip hotspots.")
        if index == 5:
            self.runtime.sync_task()
            self.runtime.refresh_jobs()
        self._refresh_picks()

    def _activate_native_form(self):
        if self._imported_task:
            name, payload = self._imported_task
            if self._native_form is None:
                self._native_form = NativeTaskForm(name, payload, atoms=self.atoms, on_change=self._native_changed, on_motif_focus=self._native_motif_focus)
            else:
                self._native_form.set_value(name, payload, atoms=self.atoms)
        self.active_panel.children = (self._panel_for_step(self.steps.value),)

    def _native_changed(self, name, payload):
        self._imported_task = (name, deepcopy(payload))
        self._import_pending = False
        self._syncing_controls = True
        self._building_yaml = True
        try:
            self.task_name.value = name
            self.samples.value = payload.get("sample_num", 1)
            self.import_yaml.value = yaml.safe_dump({name: payload}, sort_keys=False)
            hotspot = (payload.get("antigen") or {}).get("hotspot") if payload.get("format") == "antibody" else payload.get("hotspot")
            self._hotspots = []
            if hotspot and self.atoms is not None:
                self._hotspots = [self._record(int(np.flatnonzero(self.atoms.residue_uid == uid)[0]))
                    for uid in source_selector(self.atoms, hotspot)]
            self.hotspot_selectors.value = "/".join(self._selector(record) for record in self._hotspots)
        except (ValueError, KeyError) as exc:
            self._status(f"Check the edited source selector: {exc}")
        finally:
            self._building_yaml = self._syncing_controls = False
        self._refresh_draft()
        self.runtime.sync_task()
        if self.display_mode in {"source", "imported"}:
            self._color_source()

    def _fresh_chain(self):
        prefix = {"protein": "P", "peptide": "P", "rna": "R", "dna": "D", "ligand": "L"}[self.modality.value]
        existing = set() if self.atoms is None else set(map(str, self.atoms.chain_id))
        chain, index = prefix, 2
        while chain in existing:
            chain = f"{prefix}{index}"
            index += 1
        return chain

    def _source_modality(self, chain):
        if self.atoms is None:
            return None
        values = set(str(x) for x in self.atoms.mol_type[self.atoms.chain_id == chain])
        return next(iter(values)) if len(values) == 1 else ("mixed" if values else None)

    def _compatible_chains(self, modality=None):
        canonical = modality or ("protein" if self.modality.value == "peptide" else self.modality.value)
        return [chain for chain in self._roles if self._source_modality(chain) == canonical]

    def _active_source_chain(self):
        if self.motif_enabled.value and self._motif_chain and self.motif_contig.value.strip():
            return self._motif_chain
        if self.backbone.value:
            return self.backbone_source.value
        if self.modality.value == "ligand" and self.ligand_mode.value in {"type_bond", "backbone", "atom_motif"}:
            return self.ligand_source.value
        return None

    def _context_chains(self):
        active = self._active_source_chain()
        return [chain for chain, role in self._roles.items() if role == "context" and chain != active]

    def _reference_chains(self):
        return [chain for chain, role in self._roles.items() if role == "reference"]

    def _form_changed(self, change):
        if self._syncing_controls:
            return
        self._syncing_controls = True
        try:
            owner = change["owner"]
            if owner is self.modality:
                self.motif_enabled.value = False
                self._clear_motif_state()
                self.backbone.value = False
                self.fixed_sequence.value = ""
                self.secondary_structure.value = ""
                self.token_ss_yaml.value = self.pair_ss_yaml.value = self.condition_yaml.value = "{}"
                self.bond_yaml.value = "[]"
                self._form_extras = {}
                self._loaded_preset_key = None
                self.ligand_mode.value = "free"
                self.ligand_length_mode.value = "explicit"
                self.ligand_value.value = ""
                self.cyclic_offset.value = False
                self.design_chain.value = self._fresh_chain()
                self.length_plan.value = ""
                self._baskets["motif"].clear()
            elif owner is self.motif_enabled and not self.motif_enabled.value:
                self._clear_motif_state()
                self.design_chain.value = self._fresh_chain()
            elif owner is self.motif_source:
                self._baskets["motif"].clear()
                if self._motif_chain and self._motif_chain != self.motif_source.value:
                    self._clear_motif_state()
                    self.design_chain.value = self._fresh_chain()
            elif owner is self.motif_contig and not self.motif_contig.value.strip():
                self._clear_motif_state()
                self.design_chain.value = self._fresh_chain()
            elif owner is self.backbone:
                if self.backbone.value:
                    if self.motif_enabled.value:
                        raise ValueError("Turn off the motif before using a source backbone.")
                    if not self.backbone_source.value:
                        raise ValueError("Choose a source backbone chain.")
                    self.design_chain.value = self.backbone_source.value
                else:
                    self.design_chain.value = self._fresh_chain()
            elif owner is self.backbone_source and self.backbone.value:
                self.design_chain.value = self.backbone_source.value
            elif owner is self.ligand_mode:
                self.ligand_value.value = ""
                if self.ligand_mode.value == "atom_motif":
                    self.motif_enabled.value = True
                elif self.modality.value == "ligand":
                    self.motif_enabled.value = False
                    self._clear_motif_state()
                if self.ligand_mode.value not in {"type_bond", "backbone", "atom_motif"}:
                    self.design_chain.value = self._fresh_chain()
            if self.modality.value == "ligand" and self.ligand_mode.value in {"type_bond", "backbone", "atom_motif"} and self.ligand_source.value:
                self.design_chain.value = self.ligand_source.value
            if owner is self.task_name:
                name = self.task_name.value.strip() or "my_design"
                self.task_file.value = str(self.work_dir / "tasks" / f"{name}.yaml")
            active = self._active_source_chain()
            if active:
                self._roles[active] = "reference"
            self._refresh_controls()
            self._prune_hotspots()
            self._refresh_contact_references()
            self._render_roles()
        except Exception as exc:
            self._status(str(exc))
        finally:
            self._syncing_controls = False
        self._refresh_picks()
        self._refresh_draft()
        self._color_source()

    def _invalidate_check(self, change=None):
        self.last_preview = None
        self.command.value = ""
        if self.display_mode == "preview":
            self.viewer_note.value = "Previous preview · task changed"
            self.display_mode = "stale"

    def _refresh_draft(self):
        self._invalidate_check()
        try:
            name, payload = self._task()
            self._draft_error = None
            self.task_yaml.value = yaml.safe_dump({name: payload}, sort_keys=False, allow_unicode=True)
            self.yaml_status.value = "Task updated."
        except Exception as exc:
            self._draft_error = str(exc)
            self.task_yaml.value = ""
            self.yaml_status.value = str(exc)
        self._update_summary()
        self.summary_help.help_text = self.summary.value

    def refresh_controls(self):
        before = self._syncing_controls
        self._syncing_controls = True
        try:
            self._refresh_controls()
        finally:
            self._syncing_controls = before

    def _refresh_controls(self):
        polymer = self.modality.value != "ligand"
        protein = self.modality.value in {"protein", "peptide"}
        chains = [chain for chain in self._compatible_chains() if chain in self._reference_chains()]
        for control, values in ((self.motif_source, chains), (self.backbone_source, chains), (self.ligand_source, [chain for chain in self._compatible_chains("ligand") if chain in self._reference_chains()])):
            previous = control.value
            control.options = [(chain, chain) for chain in values]
            control.value = previous if previous in values else (values[0] if values else None)
        self.motif_tools.layout.display = "" if polymer else "none"
        for control in (self.motif_source, self.motif_selector, self.motif_picks, self.motif_contig):
            control.disabled = not self.motif_enabled.value or not chains
        self.ligand_box.layout.display = "none" if polymer else ""
        self.motif_enabled.layout.display = "" if polymer else "none"
        for control in (self.fixed_sequence, self.backbone, self.backbone_source, self.secondary_structure, self.token_ss_yaml, self.pair_ss_yaml):
            control.layout.display = "" if polymer else "none"
        self.cyclic_offset.layout.display = "" if protein else "none"
        opts = _COARSE_SS["protein" if protein else "rna"] if polymer else [("Not applicable", "")]
        current = self.secondary_structure.value
        self.secondary_structure.options = opts
        self.secondary_structure.value = current if current in dict(opts).values() else ""
        self.backbone.disabled = not chains or self.motif_enabled.value
        self.backbone_hint.value = "Keep backbone center coordinates while redesigning the sequence."
        self.ss_hint.value = ("Use final design positions. Positions: helix, sheet, loop. Relationships: adjacent, not_adjacent." if protein else "Use final design positions. Use paired or unpaired; paired endpoints also need matching per-position conditions.")
        self.length_plan.description = "Heavy atoms" if not polymer else ("Residues" if protein else "Nucleotides")
        self.override.layout.display = "none"
        source_ligand = self.ligand_mode.value in {"type_bond", "backbone", "atom_motif"}
        self.ligand_source.layout.display = "" if source_ligand else "none"
        self.ligand_value.layout.display = "" if self.ligand_mode.value in {"smiles", "ccd"} else "none"
        self.ligand_value.description = "SMILES" if self.ligand_mode.value == "smiles" else "CCD ID"
        self.ligand_motif_tools.layout.display = "" if not polymer and self.ligand_mode.value == "atom_motif" else "none"
        if not polymer:
            self.mark_summary.value = ("Selected source atoms form the motif. An empty selection keeps all available atoms." if self.ligand_mode.value == "atom_motif" else "This ligand mode does not need a motif selection.")
        if self.atoms is not None and self.ligand_source.value:
            names = list(dict.fromkeys(map(str, self.atoms.atom_name[self.atoms.chain_id == self.ligand_source.value])))
            selected = tuple(name for name in self.ligand_motif_atoms.value if name in names)
            self.ligand_motif_atoms.options = names
            self.ligand_motif_atoms.value = selected
        self.ligand_hint.value = "Only selected source atoms provide coordinates. The total heavy-atom count includes them." if self.ligand_mode.value == "atom_motif" else ""
        fixed = bool(self.motif_contig.value.strip()) and self.motif_enabled.value and polymer or self.backbone.value or bool(self.fixed_sequence.value.strip()) or (not polymer and self.ligand_mode.value in {"smiles", "ccd", "type_bond", "backbone"})
        distribution = not polymer and self.ligand_length_mode.value == "distribution"
        self.ligand_length_mode.layout.display = "" if not polymer and not fixed else "none"
        self.length_plan.layout.display = "none" if fixed or distribution else ""
        self.length_hint.value = ("Length is set by the contig, sequence, or source structure." if fixed else
            "Sample the heavy-atom count from the Context pocket. Add Context before checking the task." if distribution else
            "The sample count applies to each length.")
        locked = self._imported_task is not None
        self.keep_chains.disabled = self.exclude_names.disabled = locked
        if locked:
            for control in self._form_widgets:
                control.disabled = True
        else:
            for control in self._form_widgets:
                control.disabled = False
            self.backbone.disabled = not chains or self.motif_enabled.value
        self.target_selectors.value = "/".join(self._context_chains())

    def _render_roles(self):
        if self.atoms is None:
            return
        active = self._active_source_chain()
        rows = []
        summaries = backend.chain_summary(self.atoms)
        for chain in self._roles:
            info = [x for x in summaries if x["chain"] == chain]
            desc = " / ".join(f"{_MODALITY_NAMES.get(x['modality'], x['modality'])} · {x['residues']}" for x in info)
            role = W.Dropdown(options=[("Context", "context"), ("Reference", "reference")], value=self._roles[chain], layout=W.Layout(width="135px"))
            role.disabled = chain == active or self._imported_task is not None
            role.observe(lambda ch, key=chain: self._role_changed(key, ch["new"]), names="value")
            label = f"{chain} · {desc}" + (" · Design source" if chain == active else "")
            row = W.HBox([W.Label(label, tooltip=label, layout=W.Layout(flex="1 1 0", min_width="0")), role])
            row.add_class("pd-role-row")
            rows.append(row)
            self._role_controls[chain] = role
        self.chain_table.children = tuple(rows)

    def _role_changed(self, chain, value):
        if self._syncing_controls:
            return
        self._roles[chain] = value
        self.refresh_controls()
        self._refresh_contact_references()
        removed = self._prune_hotspots()
        self.target_selectors.value = "/".join(self._context_chains())
        self._refresh_picks()
        self._refresh_draft()
        self._color_source()
        self._status(f"{chain} set to {' Context' if value == 'context' else 'Reference'}." + (f" Removed {removed} hotspots outside Context." if removed else ""))

    def _prune_hotspots(self):
        context = set(self._context_chains())
        old = list(getattr(self, "_hotspots", []))
        self._hotspots = [record for record in old if record["chain"] in context]
        self._baskets["hotspot"] = OrderedDict((key, record) for key, record in self._baskets["hotspot"].items() if record["chain"] in context)
        self.hotspot_selectors.value = "/".join(self._selector(record) for record in self._hotspots)
        return len(old) - len(self._hotspots)

    def _update_summary(self):
        if self._imported_task:
            name, payload = self._imported_task
            chains = ", ".join(f"{key} ({value})" for row in payload.get("design_chains", []) for key, value in row.items())
            self.summary.value = f"Task: {name} (Full YAML)\nDesign: {chains}\nInput: {self._task_source(payload) or 'None'}\nCheck: {'Not checked' if self.last_preview is None else 'Checked'}"
            return
        motif = "None" if not self.motif_enabled.value else (self.motif_contig.value.strip() or "Not selected")
        roles = " | ".join(f"{chain}: {'Design source' if chain == self._active_source_chain() else ('Context' if role == 'context' else 'Reference')}" for chain, role in self._roles.items())
        self.summary.value = f"Design: {_MODALITY_NAMES[self.modality.value]} {self.design_chain.value} · {self.samples.value} samples / plan\nInput: {roles or 'None'}\nMotif: {motif}\nHotspots: {self.hotspot_selectors.value or 'None'} · {'Not checked' if self.last_preview is None else 'Checked'}"

    def _cache_directory(self):
        if self._source_directory is None:
            root = self.repo_root / ".cache" / "notebook_user_v3"
            root.mkdir(parents=True, exist_ok=True)
            self._source_directory = Path(tempfile.mkdtemp(prefix="editor_", dir=root))
        return self._source_directory

    @staticmethod
    def _source_locator(atoms, row):
        return tuple(str(getattr(atoms, field)[row]) for field in
            ("chain_id", "auth_asym_id", "auth_seq_id", "auth_ins_code", "res_name", "label_seq_id"))

    def _source_filters_changed(self, change):
        if self._syncing_controls or self._raw_source is None or self._imported_task:
            return
        try:
            self.load_source(self._raw_source, _preserve=True)
        except Exception as exc:
            self._source_filter_error = str(exc)
            self._refresh_draft()
            self._status(f"Input filter not applied: {exc} Correct the filter before saving.")

    def load_source(self, path, *, keep_chains=None, exclude_res_names=None, _preserve=False):
        if self._imported_task:
            raise ValueError("Choose Use form task before changing the input.")
        raw = backend.stage_input(self._path(path), self._path(self.input_dir.value))
        keep = keep_chains if keep_chains is not None else (_tokens(self.keep_chains.value) or None)
        excluded = exclude_res_names if exclude_res_names is not None else _tokens(self.exclude_names.value)
        atoms = backend.load_input(raw, keep_chains=keep, exclude_res_names=excluded)
        saved = backend.save_input(atoms, self._cache_directory() / "source.cif")
        normalized = backend.load_input(saved)
        old_atoms, old_roles = self.atoms, dict(self._roles)
        old_hotspots = list(getattr(self, "_hotspots", []))
        old_baskets = deepcopy(self._baskets)
        old_motif = (self._motif_chain, self._contig_applied,
            {key: self._motif_choice(key) for key in self.motif_atom_controls},
            [self._source_locator(old_atoms, record["row"]) for record in self._motif_rows])
        self._syncing_controls = True
        try:
            self.atoms, self.input_file, self._raw_source = normalized, saved, raw
            self._source_filter_error = None
            self.source_path.value, self.clean_path.value = str(raw), str(saved)
            self._roles = {chain: old_roles.get(chain, "context") if _preserve else "context"
                           for chain in dict.fromkeys(map(str, normalized.chain_id))}
            if keep_chains is not None:
                self.keep_chains.value = ",".join([keep_chains] if isinstance(keep_chains, str) else keep_chains)
            if exclude_res_names is not None:
                self.exclude_names.value = ",".join(exclude_res_names)
            if not _preserve:
                self._clear_motif_state()
                self.motif_enabled.value = self.backbone.value = False
                self.ligand_mode.value = "free"
                self.ligand_value.value = ""
                self.condition_yaml.value = "{}"
                self.bond_yaml.value = "[]"
                self._form_extras = {}
                self._loaded_preset_key = None
                self._hotspots = []
                for basket in self._baskets.values():
                    basket.clear()
                self.design_chain.value = self._fresh_chain()
            else:
                locators = {}
                for row, uid in enumerate(normalized.residue_uid):
                    locators.setdefault(self._source_locator(normalized, row), self._record(row))
                def remap(record):
                    return locators.get(self._source_locator(old_atoms, record["row"]))
                self._hotspots = [new for record in old_hotspots if (new := remap(record)) is not None]
                for purpose, basket in old_baskets.items():
                    self._baskets[purpose] = OrderedDict()
                    for record in basket.values():
                        new = remap(record)
                        if new is not None:
                            self._baskets[purpose][(new["kind"], new["uid"], new["atom"])] = new
                chain, contig, choices, prior_locators = old_motif
                if chain:
                    try:
                        info = backend.inspect_contig(normalized, chain, contig)
                        if [self._source_locator(normalized, record["row"]) for record in info["residues"]] != prior_locators:
                            raise ValueError("The motif source changed.")
                        self._motif_rows = info["residues"]
                        self._render_motif_atoms(choices)
                    except Exception:
                        self._clear_motif_state()
                        self.design_chain.value = self._fresh_chain()
                if self.backbone.value and self.backbone_source.value not in self._roles:
                    self.backbone.value = False
                    self.design_chain.value = self._fresh_chain()
                if self.ligand_mode.value in {"type_bond", "backbone", "atom_motif"} and self.ligand_source.value not in self._roles:
                    self.ligand_mode.value = "free"
                    self.motif_enabled.value = False
                    self.design_chain.value = self._fresh_chain()
            self._refresh_controls()
            self._prune_hotspots()
            self._refresh_contact_references()
        finally:
            self._syncing_controls = False
        self._render_roles()
        self._refresh_picks()
        self._refresh_draft()
        self.display_mode = "stale"
        self.show_source(area=self._area)
        self._status(f"Loaded {len(self._roles)} chains and {len(normalized)} heavy atoms. Review the chain roles.")
        return saved

    def _preset_group_changed(self, change):
        if self._syncing_controls:
            return
        previous = self.preset.value
        entries = [entry for entry in self.example_tasks if change["new"] == "All" or entry["group"] == change["new"]]
        values = [entry["key"] for entry in entries]
        self._syncing_controls = True
        try:
            self.preset.options = [(entry["label"], entry["key"]) for entry in entries]
            self.preset.value = previous if previous in values else (values[0] if values else None)
        finally:
            self._syncing_controls = False
        if self.preset.value != previous:
            self._preset_changed({"new": self.preset.value})

    def _preset_changed(self, change):
        if self._syncing_controls or not change["new"]:
            return
        try:
            self.load_preset(change["new"])
        except Exception as exc:
            self._status(f"Could not load preset: {exc}")

    def _preset_load_requested(self, change):
        request = change["new"]
        if not request:
            return
        try:
            self.load_preset(request.get("key"))
        except Exception as exc:
            self._status(f"Please check: {exc}")
        finally:
            self.preset_load.ack = str(request.get("id", ""))

    def _bond_example_changed(self, change=None):
        entry = self._example_by_key.get(self.bond_example.value)
        if not entry:
            self.bond_example_text.value = ""
            return
        payload = entry["payload"]
        excerpt = {key: deepcopy(payload[key]) for key in ("design_chains", "design_length", "cyclic_offset", "rm_token_type", "condition") if key in payload}
        self.bond_example_text.value = yaml.safe_dump(excerpt, sort_keys=False)

    @staticmethod
    def _task_source(payload):
        return (payload.get("antigen") or {}).get("input") if payload.get("format") == "antibody" else payload.get("input")

    @staticmethod
    def _compact_mapping(mapping):
        return {key: value for key, value in mapping.items() if value not in ({}, [], None)}

    def load_preset(self, name=None):
        key = name or self.preset.value
        if key not in self._example_by_key:
            raise ValueError(f"Unknown example task: {key}")
        entry = self._example_by_key[key]
        prepared = backend.prepare_example_task(self.repo_root, key, self._cache_directory() / "preset")
        payload = deepcopy(prepared["payload"])
        parsed, _ = backend._task_document(prepared["name"], payload, self.repo_root)
        design_ids = [chain.chain_id for chain in parsed.design_chains]
        compatible = (prepared["format"] == "normal" and len(parsed.design_chains) == 1
            and "joint" not in (payload.get("design_length") or {}))
        self._syncing_controls = True
        try:
            self._imported_task = None
            self._import_pending = False
            self._source_filter_error = None
            self.last_preview = None
            self._result_colors = []
            self.command.value = ""
            self.rcsb_id.value = self.assembly_id.value = ""
            self._form_extras = {}
            self._loaded_preset_key = key
            self.atoms = prepared["atoms"]
            self.input_file = prepared["input_file"]
            original_source = self._task_source(entry["payload"])
            self._raw_source = backend.stage_input(self._path(original_source), self.raw_dir) if original_source else None
            self.source_path.value = str(self._raw_source or "")
            self.clean_path.value = str(self.input_file or "")
            self.keep_chains.value = ""
            self.exclude_names.value = "HOH,WAT,DOD"
            self._roles = {} if self.atoms is None else {chain: "reference" if chain in design_ids else "context" for chain in dict.fromkeys(map(str, self.atoms.chain_id))}
            self._clear_motif_state()
            self.motif_enabled.value = self.backbone.value = False
            self.fixed_sequence.value = ""
            self.token_ss_yaml.value = self.pair_ss_yaml.value = self.condition_yaml.value = "{}"
            self.bond_yaml.value = "[]"
            self.secondary_structure.value = ""
            self.cyclic_offset.value = payload.get("cyclic_offset", False)
            self.ligand_mode.value = "free"
            self.ligand_value.value = ""
            self.ligand_length_mode.value = "explicit"
            self.ligand_motif_atoms.value = ()
            self.length_plan.value = ""
            self._hotspots = []
            for basket in self._baskets.values():
                basket.clear()
            first = parsed.design_chains[0]
            self.modality.value = first.sub_modality or first.modality
            self.design_chain.value = first.chain_id
            self.task_name.value = prepared["name"]
            self.samples.value = parsed.sample_num
            self.score.value = bool(payload.get("score", False))
            self.postprocess.value = bool(payload.get("postprocess", False))
            self.override.value = True
            self.task_file.value = str(self.work_dir / "tasks" / f"{prepared['name']}.yaml")
            self.source_method.selected_index = 1 if self.input_file else 2
            self._refresh_controls()
            hotspots = (payload.get("antigen") or {}).get("hotspot") if parsed.format == "antibody" else payload.get("hotspot")
            if hotspots and self.atoms is not None:
                self._hotspots = [self._record(int(np.flatnonzero(self.atoms.residue_uid == uid)[0])) for uid in source_selector(self.atoms, hotspots)]
            self._baskets["hotspot"] = OrderedDict(((record["kind"], record["uid"], record["atom"]), record) for record in self._hotspots)
            self.hotspot_selectors.value = "/".join(self._selector(record) for record in self._hotspots)
            if compatible:
                compatible = self._preset_fits_form(payload, first.chain_id)
            if compatible:
                self._hydrate_single_task(payload, first.chain_id)
                self.task_kind.value = "Form task · " + entry["group"]
            else:
                self._imported_task = (prepared["name"], payload)
                self._building_yaml = True
                self.import_yaml.value = yaml.safe_dump({prepared["name"]: payload}, sort_keys=False)
                self._building_yaml = False
                self.import_status.value = "Full task YAML is active."
                self.task_kind.value = ("Antibody" if parsed.format == "antibody" else "Joint design" if len(parsed.design_chains) > 1 else entry["group"]) + " · complete task"
                self.advanced_details.selected_index = None
            if self.preset_group.value not in {"All", entry["group"]}:
                self.preset_group.value = entry["group"]
            self.preset.options = [(item["label"], item["key"]) for item in self.example_tasks
                if self.preset_group.value == "All" or item["group"] == self.preset_group.value]
            self.preset.value = key
            self._refresh_controls()
            self._refresh_contact_references()
        finally:
            self._building_yaml = False
            self._syncing_controls = False
        self._activate_native_form()
        self.steps.value = 0
        self.loaded_preset.value = "Loaded · " + entry["label"]
        self._render_roles()
        self._refresh_picks()
        self._refresh_draft()
        self.runtime.sync_task(reset_sources=True)
        if self.input_file:
            self.display_mode = "stale"
            self._show_file(self.input_file, "imported" if self._imported_task else "source")
        else:
            self._show_empty()
        self._status("Loaded " + entry["label"])
        return deepcopy(payload)

    def _preset_fits_form(self, payload, chain):
        condition = payload.get("condition") or {}
        for key in ("sequence", "contig", "ligand"):
            if set(condition.get(key) or {}) - {chain}:
                return False
        if self.modality.value != "ligand" and condition.get("ligand"):
            return False
        if any(value != chain for value in condition.get("backbone") or []):
            return False
        ss = condition.get("secondary_structure") or {}
        if any(set(ss.get(key) or {}) - {chain} for key in ("coarse", "token")):
            return False
        if self.modality.value not in {"rna", "dna"}:
            return True
        entry = ((payload.get("condition") or {}).get("contig") or {}).get(chain)
        if not entry or not entry.get("polymer_atom"):
            return True
        records = backend.inspect_contig(self.atoms, chain, entry["token"])["residues"]
        for record in records:
            for selector, atoms in entry["polymer_atom"].items():
                if not isinstance(atoms, list) or not atoms or record["uid"] not in source_selector(self.atoms, selector):
                    continue
                if set(atoms) == set(record["available_atoms"]):
                    continue
                group = record["atom_choices"].get("sugar_base", {})
                if not group.get("enabled") or set(atoms) != set(group["atoms"]):
                    return False
        return True

    def _hydrate_single_task(self, payload, chain):
        condition = deepcopy(payload.get("condition") or {})
        known = {"input", "sample_num", "design_chains", "design_length", "condition", "hotspot", "score", "postprocess", "override", "cyclic_offset"}
        self._form_extras = {key: deepcopy(value) for key, value in payload.items() if key not in known}
        lengths = (payload.get("design_length") or {}).get(chain)
        if lengths == "from_distribution":
            self.ligand_length_mode.value = "distribution"
        elif isinstance(lengths, list):
            self.length_plan.value = ",".join(map(str, lengths))
        elif isinstance(lengths, dict):
            self.length_plan.value = f"{lengths['min']}:{lengths['max']}:{lengths.get('step', 1)}"
        elif lengths is not None:
            self.length_plan.value = str(lengths)
        sequences = condition.get("sequence") or {}
        if isinstance(sequences.get(chain), str):
            self.fixed_sequence.value = sequences.pop(chain)
            condition["sequence"] = sequences
        ss = condition.get("secondary_structure") or {}
        coarse = ss.get("coarse") or {}
        if chain in coarse:
            self.secondary_structure.value = coarse.pop(chain)
        ss["coarse"] = coarse
        tokens = ss.get("token") or {}
        self.token_ss_yaml.value = yaml.safe_dump(tokens.pop(chain, {}), sort_keys=False)
        ss["token"] = tokens
        self.pair_ss_yaml.value = yaml.safe_dump(ss.pop("pair", {}), sort_keys=False)
        condition["secondary_structure"] = self._compact_mapping(ss)
        self.bond_yaml.value = yaml.safe_dump(condition.pop("bond", []), sort_keys=False)
        backbone = condition.get("backbone") or []
        if chain in backbone:
            if self.modality.value == "ligand":
                self.ligand_mode.value = "backbone"
                self.ligand_source.value = chain
            else:
                self.backbone_source.value = chain
                self.backbone.value = True
            condition["backbone"] = [value for value in backbone if value != chain]
        ligands = condition.get("ligand") or {}
        if chain in ligands:
            ligand = ligands.pop(chain)
            self.ligand_mode.value = ligand["mode"]
            if ligand["mode"] in {"smiles", "ccd"}:
                self.ligand_value.value = ligand[ligand["mode"]]
            else:
                self.ligand_source.value = chain
                if ligand["mode"] == "atom_motif":
                    self._refresh_controls()
                    self.ligand_motif_atoms.value = tuple(ligand.get("atoms") or [])
            condition["ligand"] = ligands
        contigs = condition.get("contig") or {}
        if chain in contigs:
            entry = contigs.pop(chain)
            self.motif_source.value = chain
            self.motif_enabled.value = True
            self._motif_chain = chain
            self.motif_contig.value = self._contig_applied = entry["token"]
            info = backend.inspect_contig(self.atoms, chain, entry["token"])
            self._motif_rows = info["residues"]
            choices = {}
            for record in self._motif_rows:
                for selector, atoms in (entry.get("polymer_atom") or {}).items():
                    if record["uid"] in source_selector(self.atoms, selector):
                        choice = atoms
                        if record["modality"] in {"rna", "dna"} and isinstance(atoms, list):
                            group = record["atom_choices"].get("sugar_base", {}).get("atoms")
                            if set(atoms) == set(record["available_atoms"]):
                                choice = "full"
                            elif set(atoms) != set(group or []):
                                raise ValueError("This nucleotide atom selection requires full task YAML.")
                            else:
                                choice = "sugar_base"
                        choices[record["selector"]] = choice
            self._render_motif_atoms(choices)
            extras = {key: value for key, value in entry.items() if key not in {"token", "polymer_atom"}}
            if extras:
                contigs[chain] = extras
            condition["contig"] = contigs
            self.contig_summary.value = f"{info['copied_count']} copied residues · {info['minimum_length']}–{info['maximum_length']} total"
        self.condition_yaml.value = yaml.safe_dump(self._compact_mapping(condition), sort_keys=False)
        self.advanced_details.selected_index = 2 if yaml.safe_load(self.bond_yaml.value) else (1 if ss else None)

    def fetch_source(self):
        raw = backend.fetch_rcsb(self.rcsb_id.value.strip(), self.raw_dir, assembly_id=self.assembly_id.value.strip() or None)
        return self.load_source(raw)

    def _uploaded(self, control):
        if not control.value:
            raise ValueError("Choose a file first.")
        item = next(iter(control.value.values())) if isinstance(control.value, dict) else control.value[0]
        name = Path(item["name"]).name
        if Path(name).suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
            raise ValueError("Choose a PDB or CIF file.")
        destination = self.raw_dir / name
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        index = 2
        while destination.exists():
            destination = self.raw_dir / f"{Path(name).stem}_{index}{Path(name).suffix}"
            index += 1
        destination.write_bytes(bytes(item["content"]))
        return destination

    def load_upload(self):
        return self.load_source(self._uploaded(self.input_upload))

    def clear_source(self):
        if self._imported_task:
            raise ValueError("Choose Use form task first.")
        self._syncing_controls = True
        try:
            self.atoms = self.input_file = self._raw_source = None
            self._source_filter_error = None
            self._roles.clear()
            self._clear_motif_state()
            self.motif_enabled.value = self.backbone.value = False
            self._hotspots = []
            self._form_extras = {}
            self._loaded_preset_key = None
            self.bond_yaml.value = "[]"
            self.hotspot_selectors.value = ""
            self.ligand_mode.value = "free"
            self.ligand_value.value = ""
            self.condition_yaml.value = "{}"
            self.design_chain.value = self._fresh_chain()
            for basket in self._baskets.values():
                basket.clear()
            self._refresh_controls()
        finally:
            self._syncing_controls = False
        self._refresh_contact_references()
        self.chain_table.children = (W.Label("No input structure."),)
        self._show_empty()
        self._refresh_picks()
        self._refresh_draft()
        self._status("De novo design selected.")

    def _require_source(self):
        if self.atoms is None:
            raise ValueError("Load a structure first.")

    def _record(self, row, kind="residue", atom=None):
        atoms = self.atoms
        label = str(atoms.label_seq_id[row])
        seq = int(label) if label not in {"", ".", "?"} else int(atoms.res_id[row])
        return {"kind": kind, "row": int(row), "uid": int(atoms.residue_uid[row]),
            "chain": str(atoms.chain_id[row]), "label_seq": label, "seq": seq,
            "ins": str(atoms.ins_code[row]), "atom": atom, "name": str(atoms.res_name[row]),
            "auth_chain": str(atoms.auth_asym_id[row]), "auth_seq": str(atoms.auth_seq_id[row])}

    def _selector(self, record):
        prefix = _chain_selector(record["chain"])
        return prefix if record["kind"] == "chain" else f"{prefix}{record['seq']}{record['ins']}"

    def _add(self, record, purpose=None, toggle=False):
        purpose = purpose or ("hotspot" if self.steps.value == 2 else "motif")
        if purpose == "hotspot":
            if record["chain"] not in self._context_chains():
                raise ValueError("Hotspots must be on Context chains.")
            record = self._record(record["row"])
        else:
            if not self.motif_enabled.value and self.modality.value != "ligand":
                raise ValueError("Enable Use a motif first.")
            if record["chain"] != self.motif_source.value:
                raise ValueError("Select residues on the chosen Reference chain.")
        key = (record["kind"], record["uid"], record["atom"])
        basket = self._baskets[purpose]
        if toggle and key in basket:
            basket.pop(key)
        else:
            basket[key] = record

    def _residues(self, purpose="motif"):
        self._require_source()
        uids = {record["uid"] for record in self._baskets[purpose].values()}
        seen, records = set(), []
        for row, uid in enumerate(self.atoms.residue_uid):
            if int(uid) in uids and int(uid) not in seen:
                records.append(self._record(row))
                seen.add(int(uid))
        if not records:
            raise ValueError("Select residues first.")
        return records

    def add_batch(self, purpose=None):
        self._require_source()
        purpose = purpose or ("hotspot" if self.steps.value == 2 else "motif")
        text = self.hotspot_selector.value if purpose == "hotspot" else self.motif_selector.value
        uids = source_selector(self.atoms, text.strip())
        # Validate the entire request before changing the pending selection.
        records = [self._record(int(np.flatnonzero(self.atoms.residue_uid == uid)[0])) for uid in uids]
        if purpose == "hotspot" and any(record["chain"] not in self._context_chains() for record in records):
            raise ValueError("The hotspot range includes residues outside Context.")
        if purpose == "motif" and any(record["chain"] != self.motif_source.value for record in records):
            raise ValueError("Motif residues must belong to the selected Reference chain.")
        for record in records:
            self._add(record, purpose)
        self._refresh_picks()
        self._status(f"Added {len(records)} residues to the selection.")

    def _click_rows(self, target, scope):
        atoms = self.atoms
        chain = target.get("label_asym_id")
        if chain is not None:
            mask = atoms.label_asym_id == str(chain)
        elif target.get("auth_asym_id") is not None:
            mask = atoms.auth_asym_id == str(target["auth_asym_id"])
        else:
            raise ValueError("The selected object has no source chain identifier.")
        if scope != "chain":
            seq = target.get("seq_id")
            if str(seq) not in {"None", "", ".", "?", "0", "-1"}:
                mask &= atoms.label_seq_id == str(seq)
            elif target.get("auth_seq_id") is not None:
                mask &= atoms.auth_seq_id == str(target["auth_seq_id"])
            ins = target.get("ins_code", target.get("auth_ins_code_id"))
            if ins not in {None, "", ".", "?"}:
                mask &= (atoms.ins_code == str(ins)) | (atoms.auth_ins_code == str(ins))
            if target.get("comp_id"):
                mask &= atoms.res_name == str(target["comp_id"])
            names = target.get("atom_id") or []
            names = [names] if isinstance(names, str) else list(names)
            if names:
                mask &= np.isin(atoms.atom_name, names)
        rows = np.flatnonzero(mask)
        if not len(rows) or (scope != "chain" and len(set(map(int, atoms.residue_uid[rows]))) != 1):
            raise ValueError("Could not identify a unique source residue. Use its residue range instead.")
        return rows

    @staticmethod
    def _toggled_atoms(current, clicked):
        selected = list(current)
        if all(name in selected for name in clicked):
            return [name for name in selected if name not in clicked]
        return list(dict.fromkeys([*selected, *clicked]))

    def _on_click(self, change):
        event = change.get("new") or {}
        event = event.get("eventData", event)
        if not event:
            return
        try:
            if self.atoms is None or self.display_mode not in {"source", "imported"}:
                return
            scope = event.get("scope", self.view.selection_scope)
            targets = event.get("targets") or [event]
            if scope == "chain":
                targets = list({(target.get("label_asym_id"), target.get("auth_asym_id")): target for target in targets}.values())
            all_rows = list(dict.fromkeys(int(row) for target in targets for row in self._click_rows(target, scope)))
            records = list({int(self.atoms.residue_uid[row]): self._record(row) for row in all_rows}.values())
            if self.steps.value not in {1, 2}:
                self._status(f"Selected {len(records)} residues · {_PICK_NAMES[scope]} scope. Open Motif or Hotspot to apply task conditions.")
                return
            purpose = "hotspot" if self.steps.value == 2 else "motif"
            if purpose == "hotspot" and any(record["chain"] not in self._context_chains() for record in records):
                raise ValueError("Hotspots must be on Context chains.")
            if self._imported_task:
                if purpose == "hotspot":
                    name, payload = deepcopy(self._imported_task)
                    selected = {record["uid"]: record for record in self._hotspots}
                    remove = all(record["uid"] in selected for record in records)
                    for record in records:
                        if remove:
                            selected.pop(record["uid"], None)
                        else:
                            selected[record["uid"]] = record
                    destination = payload.setdefault("antigen", {}) if payload.get("format") == "antibody" else payload
                    destination["hotspot"] = "/".join(self._selector(record) for record in selected.values())
                    self._native_changed(name, payload)
                    self._native_form.set_value(name, payload, atoms=self.atoms)
                    self.view.highlight = {"data": [self._query(record) for record in records]}
                    self._status("Hotspots updated in the loaded task.")
                else:
                    # Complete tasks keep their explicit contig. Picking focuses
                    # existing motif residues instead of silently changing it.
                    atom_targets = self._atom_targets(all_rows) if scope == "atom" else {}
                    if scope == "atom":
                        for record in records:
                            matches = [item for item in self._native_form._motif_records.values() if item["uid"] == record["uid"]]
                            if len(matches) != 1 or matches[0]["modality"] != "protein":
                                raise ValueError("Every selected atom or bond endpoint must belong to an existing protein Motif residue. RNA/DNA use supported atom groups.")
                    for record in records:
                        selector = self._selector(record)
                        if self._native_form.focus_motif_residue(selector) and scope == "atom":
                            self._native_form.toggle_motif_atoms(selector, atom_targets.get(selector, []))
                    if scope == "chain":
                        self.view.highlight = {"data": [{"struct_asym_id": chain} for chain in dict.fromkeys(record["chain"] for record in records)]}
                    else:
                        self.view.highlight = {"data": [dict(self._query(record), **({"atoms": atom_targets.get(self._selector(record), [])} if scope == "atom" else {})) for record in records]}
                    self._status("Selected source residues. Edit the loaded Contig to add or remove motif segments.")
                return
            if purpose == "motif" and self.modality.value == "ligand":
                if self.ligand_mode.value != "atom_motif" or any(record["chain"] != self.ligand_source.value for record in records):
                    raise ValueError("Choose a source ligand atom motif first.")
                clicked = list(dict.fromkeys(str(self.atoms.atom_name[row]) for row in all_rows))
                self.ligand_motif_atoms.value = tuple(self._toggled_atoms(self.ligand_motif_atoms.value, clicked)) if scope == "atom" else ()
                self.view.highlight = {"data": [{"struct_asym_id": self.ligand_source.value, "atoms": list(self.ligand_motif_atoms.value or self.ligand_motif_atoms.options)}]}
                self._status("Ligand motif atoms updated.")
                return
            if purpose == "motif":
                if not self.motif_enabled.value:
                    raise ValueError("Enable Use a motif first.")
                if any(record["chain"] != self.motif_source.value for record in records):
                    raise ValueError("Select the chosen Reference chain.")
            if purpose == "motif" and scope == "atom":
                if self.modality.value not in {"protein", "peptide"}:
                    raise ValueError("For RNA or DNA, choose all atoms or a supported atom group in the Atoms editor.")
                for selector, names in self._atom_targets(all_rows).items():
                    controls = self.motif_atom_controls.get(selector)
                    if controls:
                        current = controls["atoms"].value if controls["mode"].value == "custom" else ()
                        chosen = self._toggled_atoms(current, names)
                        self._syncing_controls = True
                        try:
                            controls["mode"].value = "custom"
                            controls["atoms"].value = tuple(chosen)
                        finally:
                            self._syncing_controls = False
                        self._show_motif_atom_editor(selector, highlight=False)
                        self._motif_atoms_changed(selector)
                    else:
                        row = next(row for row in all_rows if self._selector(self._record(row)) == selector)
                        self._add(self._record(row), purpose)
                        self._pending_atom_choices[selector] = self._toggled_atoms(self._pending_atom_choices.get(selector, ()), names)
                self._refresh_picks()
                self._status("Motif atoms selected. Apply residues for any newly selected motif residues.")
            else:
                basket = self._baskets[purpose]
                selected = {record["uid"] for record in basket.values()}
                remove = all(record["uid"] in selected for record in records)
                for record in records:
                    if remove:
                        for key in [key for key, value in basket.items() if value["uid"] == record["uid"]]:
                            basket.pop(key)
                        self._pending_atom_choices.pop(self._selector(record), None)
                    else:
                        self._add(record, purpose)
                self._refresh_picks()
                self._status(f"Updated {purpose} selection · {len(records)} residues. Apply it to update the task.")
        except Exception as exc:
            self._status(f"Selection not applied: {exc}")
        finally:
            self.view.click_event = {}

    def _atom_targets(self, rows):
        result = {}
        for row in rows:
            selector = self._selector(self._record(row))
            result.setdefault(selector, []).append(str(self.atoms.atom_name[row]))
        return {selector: list(dict.fromkeys(names)) for selector, names in result.items()}

    def _query(self, record):
        result = {"struct_asym_id": record["chain"]}
        if record["kind"] != "chain":
            if record["label_seq"] not in {"", ".", "?"}:
                result["residue_number"] = record["seq"]
            else:
                result["auth_residue_number"] = int(record["auth_seq"])
                if record["ins"]:
                    result["auth_ins_code_id"] = record["ins"]
            if record["atom"]:
                result["atoms"] = [record["atom"]]
        return result

    def _refresh_picks(self):
        for purpose, control in (("motif", self.motif_picks), ("hotspot", self.hotspot_picks)):
            values = tuple(control.value)
            options = []
            for key, record in self._baskets[purpose].items():
                selector = self._selector(record)
                suffix = ""
                if purpose == "motif":
                    choice = self._motif_choice(selector) if selector in self.motif_atom_controls else self._pending_atom_choices.get(selector, "full")
                    suffix = " · " + (f"{len(choice)} atoms" if isinstance(choice, list) and choice else "All atoms" if choice == "full" or not choice else str(choice))
                options.append((f"{selector} · {record['name']}{suffix}", key))
            control.options = options
            available = {value for _, value in options}
            control.value = tuple(value for value in values if value in available)
        if self.display_mode == "source":
            self.view.highlight = {"data": [self._query(record) for record in self.selection.values()]}

    def _highlight_pick_list(self, purpose, keys):
        if purpose == "motif" and keys:
            record = self._baskets[purpose].get(keys[-1])
            if record:
                selector = self._selector(record)
                if selector in self.motif_atom_controls:
                    self._show_motif_atom_editor(selector)
        if self.display_mode == "source":
            self.view.highlight = {"data": [self._query(self._baskets[purpose][key]) for key in keys if key in self._baskets[purpose]]}

    def clear_picks(self, purpose=None):
        purpose = purpose or ("hotspot" if self.steps.value == 2 else "motif")
        self._baskets[purpose].clear()
        self._refresh_picks()

    def remove_picks(self, purpose=None):
        purpose = purpose or ("hotspot" if self.steps.value == 2 else "motif")
        control = self.hotspot_picks if purpose == "hotspot" else self.motif_picks
        for key in control.value:
            self._baskets[purpose].pop(key, None)
        self._refresh_picks()

    def mark_hotspot(self):
        records = self._residues("hotspot") if self._baskets["hotspot"] else []
        if any(record["chain"] not in self._context_chains() for record in records):
            raise ValueError("Hotspots must be on Context chains.")
        self._hotspots = records
        self.hotspot_selectors.value = "/".join(self._selector(record) for record in records)
        self._refresh_draft()
        self._color_source()
        self.view.highlight = {"data": []}
        self._status(f"Applied {len(records)} hotspots." if records else "Hotspots cleared.")
        return self.hotspot_selectors.value

    def clear_hotspots(self):
        self._hotspots = []
        self.hotspot_selectors.value = ""
        self.clear_picks("hotspot")
        self._refresh_draft()
        self._color_source()

    def _refresh_contact_references(self):
        values = self._reference_chains()
        old_options = list(self.contact_references.options)
        selected = self.contact_references.value
        self.contact_references.options = values
        self.contact_references.value = tuple(chain for chain in values if chain in selected or chain not in old_options)
        self._clear_contacts()

    def _clear_contacts(self):
        self._contact_records = []
        self.contact_candidates.options = []
        self.contact_note.value = "Choose Reference chains to find nearby Context residues. Motif selection is independent."

    def _contact_options_changed(self, change):
        if not self._syncing_controls:
            self._clear_contacts()

    def _highlight_contacts(self, selected):
        if self.display_mode == "source":
            self.view.highlight = {"data": [self._query(self._record(record["row"]))
                for record in self._contact_records if record["uid"] in selected]}

    def find_hotspots(self):
        self._require_source()
        if self._source_filter_error:
            raise ValueError(self._source_filter_error)
        self._contact_records = backend.calculate_hotspots(self.atoms, self._context_chains(),
            list(self.contact_references.value), self.contact_cutoff.value)
        self.contact_candidates.options = [(f"{record['selector']} · {record['name']} · {record['minimum_distance']:.2f} Å", record["uid"])
            for record in self._contact_records]
        self.contact_candidates.value = tuple(record["uid"] for record in self._contact_records)
        self.contact_note.value = f"{len(self._contact_records)} Context residues within {self.contact_cutoff.value:g} Å. Select candidates, then merge or replace hotspots."
        self.show_source(area=self._area)
        self._highlight_contacts(self.contact_candidates.value)
        return deepcopy(self._contact_records)

    def apply_contact_hotspots(self, merge=True):
        if self._imported_task:
            raise ValueError("Choose Use form task before changing hotspots.")
        selected = set(self.contact_candidates.value)
        records = [self._record(record["row"]) for record in self._contact_records if record["uid"] in selected]
        combined = OrderedDict((record["uid"], record) for record in (getattr(self, "_hotspots", []) if merge else []))
        combined.update((record["uid"], record) for record in records)
        self._hotspots = list(combined.values())
        self._baskets["hotspot"] = OrderedDict(((record["kind"], record["uid"], record["atom"]), record) for record in self._hotspots)
        self.hotspot_selectors.value = "/".join(self._selector(record) for record in self._hotspots)
        self._refresh_picks()
        self._refresh_draft()
        self._color_source()
        self._status(f"Applied {len(self._hotspots)} hotspots.")
        return self.hotspot_selectors.value

    def _clear_motif_state(self):
        self._motif_chain = None
        self._contig_applied = ""
        self.motif_condition = {}
        self._motif_rows = []
        self.motif_atom_controls.clear()
        self._focused_motif_selector = None
        self._pending_atom_choices.clear()
        self.motif_contig.value = ""
        self.motif_atoms_box.children = ()
        self.mark_summary.value = "No motif selected."
        self.contig_summary.value = "Select motif residues first."

    def mark_motif(self):
        records = self._residues("motif")
        chain = self.motif_source.value
        if any(record["chain"] != chain for record in records):
            raise ValueError("Motif residues must belong to one Reference chain.")
        self._syncing_controls = True
        try:
            self.motif_enabled.value = True
            self.backbone.value = False
            self.design_chain.value = chain
            self._motif_chain = chain
            self._roles[chain] = "reference"
            self.motif_contig.value = "/".join(self._selector(record) for record in records)
        finally:
            self._syncing_controls = False
        self.apply_contig()
        return deepcopy(self.motif_condition)

    def apply_contig(self):
        self._require_source()
        chain = self.motif_source.value
        if not chain:
            raise ValueError("Choose a Reference chain.")
        info = backend.inspect_contig(self.atoms, chain, self.motif_contig.value.strip())
        old = {selector: self._motif_choice(selector) for selector in self.motif_atom_controls}
        old.update(self._pending_atom_choices)
        self._pending_atom_choices.clear()
        self._motif_chain = chain
        self._roles[chain] = "reference"
        self._syncing_controls = True
        try:
            self.motif_enabled.value = True
            self.backbone.value = False
            self.design_chain.value = chain
            self._contig_applied = self.motif_contig.value.strip()
            self._motif_rows = info["residues"]
            self._render_motif_atoms(old)
            self._refresh_controls()
        finally:
            self._syncing_controls = False
        self.motif_condition = self._current_motif()
        bounds = f"{info['minimum_length']}" if info['minimum_length'] == info['maximum_length'] else f"{info['minimum_length']}–{info['maximum_length']}"
        self.contig_summary.value = f"Keep {info['copied_count']} source residues; final length {bounds}."
        self.mark_summary.value = f"{chain} is the design source. Its full chain remains available; other source residues are excluded from Context."
        removed = self._prune_hotspots()
        self._render_roles()
        self._refresh_draft()
        self._color_source()
        self._status("Contig applied. Choose atoms for each residue." + (f" Removed {removed} hotspots outside Context." if removed else ""))
        return info

    def _render_motif_atoms(self, previous):
        focused = self._focused_motif_selector
        self.motif_atom_controls = {}
        for record in self._motif_rows:
            selector = record["selector"]
            choices = record["atom_choices"]
            options = [(value["label"], key) for key, value in choices.items() if value["enabled"]]
            mode = W.Dropdown(options=options, value="full", description="Keep atoms", style={"description_width": "94px"}, layout=W.Layout(width="100%", min_width="0", margin="2px 0"))
            selected = W.SelectMultiple(options=record["available_atoms"], value=(), rows=min(6, max(2, len(record["available_atoms"]))), layout=W.Layout(width="100%", min_width="0", margin="2px 0"))
            mode.description = ""
            mode.style.description_width = "0px"
            note = self._info("")
            controls = {"mode": mode, "atoms": selected, "note": note, "record": record}
            self.motif_atom_controls[selector] = controls
            prior = previous.get(selector, "full")
            if isinstance(prior, list):
                mode.value = "custom"
                selected.value = tuple(name for name in prior if name in record["available_atoms"])
            elif prior in dict(options).values():
                mode.value = prior
            for control in (mode, selected):
                control.observe(lambda ch, key=selector: self._motif_atoms_changed(key), names="value")
            self._refresh_atom_note(selector)
            controls["box"] = W.VBox([self._info(f"{selector} · {record['name']}"), mode, selected, note], layout=W.Layout(width="100%", min_width="0", grid_gap="4px"))
        self._baskets["motif"] = OrderedDict()
        for record in self._motif_rows:
            source = self._record(record["row"])
            self._baskets["motif"][(source["kind"], source["uid"], source["atom"])] = source
        self._refresh_picks()
        self._show_motif_atom_editor(focused, highlight=False)
        self.motif_atoms_box.layout.max_height = ""
        self.motif_atoms_box.layout.overflow = "visible"

    def _motif_choice(self, selector):
        controls = self.motif_atom_controls[selector]
        return list(controls["atoms"].value) or "full" if controls["mode"].value == "custom" else controls["mode"].value

    def _refresh_atom_note(self, selector):
        controls = self.motif_atom_controls[selector]
        mode = controls["mode"].value
        controls["atoms"].layout.display = "" if mode == "custom" else "none"
        value = self._motif_choice(selector)
        names = backend.motif_atom_names(self.atoms, selector, value)
        names = names if names is not None else controls["record"]["available_atoms"]
        extra = controls["record"]["atom_choices"].get(mode, {})
        missing = extra.get("missing_atoms", [])
        controls["note"].value = f"Keep {len(names)} atoms: " + ", ".join(names) + ("; absent from the input: " + ", ".join(missing) if missing else "")

    def _motif_atoms_changed(self, selector):
        if self._syncing_controls:
            return
        try:
            self._refresh_atom_note(selector)
            self.motif_condition = self._current_motif()
            self._refresh_draft()
            self._color_source()
            self._highlight_motif_atoms(selector)
            self._refresh_picks()
        except Exception as exc:
            self._status(str(exc))

    def _show_motif_atom_editor(self, selector=None, *, highlight=True):
        selector = selector if selector in self.motif_atom_controls else next(iter(self.motif_atom_controls), None)
        self._focused_motif_selector = selector
        self.motif_atoms_box.children = ((self.motif_atom_controls[selector]["box"],) if selector else
            (self._info("Apply residues to choose their atoms. Empty atom selections retain all observed atoms."),))
        if selector and highlight:
            self._highlight_motif_atoms(selector)

    def _native_motif_focus(self, selector):
        if self.atoms is None:
            return
        uids = source_selector(self.atoms, selector)
        self.view.highlight = {"data": [self._query(self._record(int(np.flatnonzero(self.atoms.residue_uid == uid)[0]))) for uid in uids]}

    def _current_motif(self):
        if not self.motif_enabled.value or self.modality.value == "ligand":
            return {}
        if not self.motif_contig.value.strip():
            return {}
        if not self._motif_chain:
            raise ValueError("Select motif residues and apply a contig.")
        if self.motif_contig.value.strip() != self._contig_applied:
            raise ValueError("Contig changed. Choose Apply contig.")
        return backend.build_motif_condition(self.atoms, self._motif_chain, self.motif_contig.value.strip(),
            {selector: self._motif_choice(selector) for selector in self.motif_atom_controls})

    def _highlight_motif_atoms(self, selector):
        if self.input_file is None:
            return
        if self.display_mode != "source":
            self.show_source()
        controls = self.motif_atom_controls[selector]
        record = self._record(controls["record"]["row"])
        query = self._query(record)
        names = backend.motif_atom_names(self.atoms, selector, self._motif_choice(selector))
        query["atoms"] = list(names or controls["record"]["available_atoms"])
        self.view.highlight = {"data": [query]}

    def _compose_form(self):
        if self._source_filter_error:
            raise ValueError(f"Correct the input filter: {self._source_filter_error}")
        name = self.task_name.value.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("Use letters, numbers, underscores, or hyphens for the task name.")
        chain = self.design_chain.value.strip()
        if not chain:
            raise ValueError("Enter a design chain ID.")
        if self.samples.value <= 0:
            raise ValueError("The sample count must be a positive integer.")
        canonical = "protein" if self.modality.value == "peptide" else self.modality.value
        active = self._active_source_chain()
        if self._source_modality(chain) is not None and chain != active and self._roles.get(chain) != "reference":
            raise ValueError("This chain already exists in the input. Use a new ID, or select a motif or source backbone.")
        condition = self._current_motif()
        bonds = yaml.load(self.bond_yaml.value or "[]", Loader=_UniqueLoader)
        if not isinstance(bonds, list):
            raise ValueError("Chemical bonds must be a YAML list.")
        if bonds:
            condition["bond"] = bonds
        if self.fixed_sequence.value.strip():
            condition.setdefault("sequence", {})[chain] = self.fixed_sequence.value.strip()
        if self.backbone.value:
            condition["backbone"] = [chain]
        if self.secondary_structure.value:
            condition.setdefault("secondary_structure", {}).setdefault("coarse", {})[chain] = self.secondary_structure.value
        token, pair = _yaml(self.token_ss_yaml.value), _yaml(self.pair_ss_yaml.value)
        if token:
            condition.setdefault("secondary_structure", {}).setdefault("token", {})[chain] = token
        if pair:
            condition.setdefault("secondary_structure", {})["pair"] = pair
        if canonical == "ligand":
            mode = self.ligand_mode.value
            if mode in {"type_bond", "backbone", "atom_motif"} and not self.ligand_source.value:
                raise ValueError("Choose a source ligand chain.")
            if mode == "backbone":
                condition["backbone"] = [chain]
            elif mode != "free":
                ligand = {"mode": mode}
                if mode in {"smiles", "ccd"}:
                    text = self.ligand_value.value.strip()
                    if not text:
                        raise ValueError("Enter ligand SMILES or a CCD ID.")
                    ligand[mode] = text
                elif mode == "atom_motif":
                    names = list(self.ligand_motif_atoms.value)
                    if not names:
                        names = list(self.ligand_motif_atoms.options)
                    if not names:
                        raise ValueError("The source ligand has no available atoms.")
                    ligand["atoms"] = names
                condition.setdefault("ligand", {})[chain] = ligand
        condition = _merge(condition, _yaml(self.condition_yaml.value))
        self._validate_form_condition(condition, chain, canonical)
        payload = {**deepcopy(self._form_extras), "input": str(self.work_dir / "input" / f"{name}.cif") if self.input_file else None, "sample_num": self.samples.value,
            "design_chains": [{chain: self.modality.value}], "score": self.score.value,
            "postprocess": self.postprocess.value}
        if self.postprocess.value:
            payload["override"] = True
        ligand = condition.get("ligand", {}).get(chain, {})
        fixed = isinstance(condition.get("sequence", {}).get(chain), str) or chain in condition.get("contig", {}) or chain in condition.get("backbone", []) or ligand.get("mode") in {"smiles", "ccd", "type_bond"}
        if not fixed:
            lengths = self._lengths()
            if ligand.get("mode") == "atom_motif" and lengths != "from_distribution":
                minimum = lengths["min"] if isinstance(lengths, dict) else min(lengths)
                if minimum < len(ligand["atoms"]):
                    raise ValueError("The total heavy-atom count cannot be smaller than the retained atom count.")
            payload["design_length"] = {chain: lengths}
        if condition:
            payload["condition"] = condition
        context = self._context_chains()
        if self.input_file:
            included = [*context, *([active] if active else [])]
            if not included:
                payload["input"] = None
        if self.hotspot_selectors.value:
            payload["hotspot"] = self.hotspot_selectors.value
        if self.cyclic_offset.value:
            payload["cyclic_offset"] = True
        return name, payload

    def _prepared_payload(self, payload, *, final=False):
        prepared = deepcopy(payload)
        if prepared.get("postprocess"):
            prepared["override"] = True
        if self._imported_task:
            source = self._task_source(prepared)
            if final and self._loaded_preset_key and source and self.input_file and self._path(source) == self.input_file:
                name = self._imported_task[0]
                saved = backend.save_input(self.atoms, self.work_dir / "input" / f"{name}.cif")
                if prepared.get("format") == "antibody":
                    prepared["antigen"]["input"] = str(saved)
                else:
                    prepared["input"] = str(saved)
            return prepared
        if not prepared.get("input"):
            return prepared
        included = [*self._context_chains()]
        active = self._active_source_chain()
        if active:
            included.append(active)
        destination = Path(prepared["input"]) if final else self._cache_directory() / "prepared.cif"
        prepared["input"] = str(backend.save_input(self.atoms[np.isin(self.atoms.chain_id, included)], destination))
        return prepared

    def _task(self):
        if self._import_pending:
            raise ValueError("The YAML has unapplied changes. Apply it or return to the form task.")
        name, payload = deepcopy(self._imported_task) if self._imported_task else self._compose_form()
        if payload.get("postprocess"):
            payload["override"] = True
        return name, payload

    def build_yaml(self):
        self._refresh_draft()
        return self._task()[1]

    def _import_changed(self, change):
        if self._building_yaml:
            return
        self._import_pending = True
        self.import_status.value = "YAML changed. Choose Apply YAML to use it."
        self._invalidate_check()

    def edit_current_yaml(self):
        name, payload = self._task()
        payload = self._prepared_payload(payload, final=True)
        self._building_yaml = True
        try:
            self.import_yaml.value = yaml.safe_dump({name: payload}, sort_keys=False, allow_unicode=True)
        finally:
            self._building_yaml = False
        self._import_pending = False

    def apply_yaml(self):
        document = _yaml(self.import_yaml.value)
        if len(document) != 1:
            raise ValueError("Provide one named task. For joint design, declare multiple design chains within that task.")
        name, payload = next(iter(document.items()))
        backend._task_document(name, payload, self.repo_root)
        # Full configuration owns the task until the user explicitly returns to forms.
        self._imported_task = (name, deepcopy(payload))
        self._import_pending = False
        self.import_status.value = "YAML applied. Your form task is saved and can be restored."
        self.refresh_controls()
        self._activate_native_form()
        self._render_roles()
        self._refresh_draft()
        self._status("YAML applied.")
        return payload

    def use_form_task(self):
        self._imported_task = None
        self._import_pending = False
        self.import_status.value = "Using the form task."
        self._activate_native_form()
        self.refresh_controls()
        self._render_roles()
        self._refresh_draft()

    def _check_advanced(self):
        self._move_viewer("advanced")
        return self.check_task()

    def _export_advanced(self):
        self._move_viewer("advanced")
        return self.export_task()

    def check_task(self):
        name, payload = self._task()
        result = backend.preview_task(name, self._prepared_payload(payload), invocation_dir=self.repo_root,
            output_dir=self._path(self.preview_dir.value), context_mode=self.context_mode.value)
        self.last_preview = result
        if result["input_preview"]:
            self._move_viewer("advanced" if self._area == "advanced" else "basic")
            self._show_file(result["input_preview"], "preview")
        else:
            self._show_empty("No supplied coordinates", "preview")
        self._update_summary()
        self._status(f"Check passed: {result['request_count']} samples planned; {result['checked_count']} representative plans checked.")
        with self.log:
            self.log.clear_output(wait=True)
            print(f"Planned samples: {result['request_count']}. Checked {result['checked_count']} representative plans.")
            print("The condition view shows Context and fixed coordinates. New design regions appear after generation.")
        return result

    def export_task(self):
        name, payload = self._task()
        # Export verifies the current configuration, never a previously rendered YAML.
        if self.last_preview is None:
            self.check_task()
        destination = self.work_dir / "tasks" / f"{name}.yaml" if self._imported_task else self._path(self.task_file.value)
        payload = self._prepared_payload(payload, final=True)
        path = backend.save_task(name, payload, destination, invocation_dir=self.repo_root)
        command = ["partidesign", "design", str(path), "--task", name, "--out", str(self._path(self.output_dir.value)),
            "--compile", str(self.runtime.compile_enabled.value).lower()]
        if self.checkpoint.value.strip():
            command += ["--ckpt", str(self._path(self.checkpoint.value.strip()))]
        if self.context_mode.value != "auto":
            command += ["--override", f"model.context_constraint_mode={self.context_mode.value}"]
        self.command.value = shlex.join(command)
        self._status(f"Task saved: {path.name}")
        return path

    def _move_viewer(self, area):
        self._area = area

    def _show_empty(self, caption="No input structure", mode="empty"):
        self.display_mode = mode
        self._display_path = None
        self.view.click_event = {}
        self.view.highlight = {"data": []}
        self.view.color_data = None
        self.view.custom_data = None
        self.view.molecule_id = ""
        self.view.alphafold_view = False
        self.viewer_note.value = caption
        self.viewer_box.children = (self.viewer_header, self.view)

    def _show_file(self, path, mode):
        path = self._path(path)
        suffix = path.suffix.lower()
        if suffix not in {".pdb", ".cif", ".mmcif"}:
            raise ValueError("Choose a PDB or CIF structure file.")
        self.display_mode = mode
        self._display_path = path
        self.view.click_event = {}
        self.view.highlight = {"data": []}
        self.view.color_data = None
        self.view.alphafold_view = mode == "refold"
        self._result_colors = self._cif_colors(path) if suffix != ".pdb" and mode != "refold" else []
        self.view.custom_data = {"data": path.read_bytes(), "format": "pdb" if suffix == ".pdb" else "cif", "binary": False}
        self.viewer_note.value = {"source": "Source input", "preview": "Applied conditions", "result": "Raw result", "refold": "Validated refold", "imported": "Task source"}[mode]
        self.viewer_box.children = (self.viewer_header, self.view)
        if mode == "source" or (mode == "imported" and path == self.input_file):
            self._color_source()
            self._refresh_picks()
        elif self._result_colors and mode != "imported":
            self.apply_category_colors()

    def show_source(self, area="basic"):
        if self._imported_task:
            source = self._task_source(self._imported_task[1])
            self._move_viewer("advanced")
            if source:
                path = self._path(source)
                if self.display_mode == "imported" and getattr(self, "_display_path", None) == path:
                    self._color_source()
                    self._refresh_picks()
                else:
                    self._show_file(path, "imported")
            else:
                self._show_empty()
            return
        if self.atoms is None:
            self._show_empty()
            return
        self._move_viewer(area)
        if self.display_mode == "source" and getattr(self, "_display_path", None) == self.input_file:
            self._color_source()
            self._refresh_picks()
        else:
            self._show_file(self.input_file, "source")

    def _color_source(self):
        if self.display_mode not in {"source", "imported"} or self.atoms is None:
            return
        data = []
        for chain, role in self._roles.items():
            context_color, reference_color = ("#d8dfe8", "#e9edf3")
            color = context_color if chain in self._context_chains() else reference_color
            data.append({"struct_asym_id": chain, "color": _rgb(color)})
        try:
            if self._imported_task:
                payload = self._imported_task[1]
                condition = payload.get("condition") or {}
                for chain, entry in (condition.get("contig") or {}).items():
                    info = self._native_form._inspect(chain, entry.get("token", ""))
                    mapping = entry.get("polymer_atom") or {}
                    for record in info["residues"]:
                        choice = "full"
                        for selector, value in mapping.items():
                            if record["uid"] in source_selector(self.atoms, selector):
                                choice = value
                        names = choice if isinstance(choice, list) else backend.motif_atom_names(self.atoms, record["selector"], choice)
                        query = self._query(self._record(record["row"]))
                        query.update(atoms=list(names or record["available_atoms"]), color=_rgb(_CATEGORY_COLORS[80][1]))
                        data.append(query)
                for chain, entry in (condition.get("ligand") or {}).items():
                    if entry.get("mode") == "atom_motif":
                        names = list(entry.get("atoms") or self.atoms.atom_name[self.atoms.chain_id == chain])
                        data.append({"struct_asym_id": chain, "atoms": names, "color": _rgb(_CATEGORY_COLORS[80][1])})
            if self.motif_enabled.value and self._motif_chain:
                for controls in self.motif_atom_controls.values():
                    record = controls["record"]
                    query = self._query(self._record(record["row"]))
                    names = backend.motif_atom_names(self.atoms, record["selector"], self._motif_choice(record["selector"]))
                    query.update(atoms=list(names or record["available_atoms"]), color=_rgb(_CATEGORY_COLORS[80][1]))
                    data.append(query)
            if self.modality.value == "ligand" and self.ligand_mode.value == "atom_motif" and self.ligand_source.value:
                data.append({"struct_asym_id": self.ligand_source.value, "atoms": list(self.ligand_motif_atoms.value or self.ligand_motif_atoms.options), "color": _rgb(_CATEGORY_COLORS[80][1])})
        except ValueError:
            pass
        for record in getattr(self, "_hotspots", []):
            data.append({**self._query(record), "color": _rgb(_CATEGORY_COLORS[100][1])})
        self.view.color_data = {"data": data, "nonSelectedColor": _rgb("#e9edf3"), "keepColors": False, "keepRepresentations": False}

    def apply_category_colors(self):
        if self.display_mode == "source":
            self._color_source()
            return 0
        if not self._result_colors:
            self._status("This file has no complete condition categories. Keeping its original colors.")
            return 0
        self.view.color_data = {"data": self._result_colors, "nonSelectedColor": _rgb("#d8dfe8"), "keepColors": False, "keepRepresentations": False}
        return len(self._result_colors)

    def load_result(self, path, *, refold=False):
        resolved = self._path(path)
        self.steps.value = 5
        self._move_viewer("runtime")
        self._show_file(resolved, "refold" if refold else "result")
        self.runtime.result_file.value = str(resolved)
        self._status("Result opened.")
        return resolved

    def load_result_upload(self):
        return self.load_result(self._uploaded(self.runtime.result_upload))

    def display(self):
        self._move_viewer("basic")
        display(self.widget)

    def display_advanced(self):
        self._move_viewer("advanced")
        self.steps.value = 3
        display(self.widget)

    def close(self):
        self.view.unobserve(self._on_click, names="click_event")
        self.runtime.close()
        self.view.close()
        self.widget.close()

    def _lengths(self):
        if self.modality.value == "ligand" and self.ligand_length_mode.value == "distribution":
            if not self._context_chains():
                raise ValueError("From distribution needs a Context pocket. Set a source chain to Context first.")
            return "from_distribution"
        text = self.length_plan.value.strip()
        if not text:
            raise ValueError("Enter a design length.")
        if text == "from_distribution":
            if self.modality.value != "ligand":
                raise ValueError("From distribution is available for ligands. For polymers, enter a residue or nucleotide count.")
            return text
        try:
            if ":" in text:
                numbers = [int(value) for value in text.split(":")]
                if len(numbers) not in {2, 3}:
                    raise ValueError
                low, high = numbers[:2]
                step = numbers[2] if len(numbers) == 3 else 1
                if low <= 0 or high < low or step <= 0 or (high - low) % step:
                    raise ValueError
                return {"min": low, "max": high, "step": step}
            values = [int(value) for value in _tokens(text)]
        except ValueError as exc:
            raise ValueError("Enter positive integer lengths, or a min:max:step range that includes both endpoints.") from exc
        if not values or any(value <= 0 for value in values):
            raise ValueError("Enter positive integer lengths. Only ligands support from_distribution.")
        return values

    def _validate_sequence(self, value, canonical, chain):
        # Use the native parser, including bracket syntax, instead of a second
        # sequence grammar. CCD chemistry/modality still needs the CPU compiler.
        from ..core.compiler import _parse_sequence_residue, _parse_sequence_string
        try:
            if isinstance(value, str):
                _parse_sequence_string(value, canonical, f"condition.sequence.{chain}")
            elif isinstance(value, list) and value:
                for row in value:
                    if not isinstance(row, dict) or set(row) != {"resid", "residue"}:
                        raise ValueError
                    _parse_sequence_residue(row["residue"], canonical, f"condition.sequence.{chain}")
            else:
                raise ValueError
        except Exception as exc:
            alphabet = {"protein": "the 20 standard amino-acid letters", "rna": "A/C/G/U", "dna": "A/C/G/T"}[canonical]
            raise ValueError(f"Invalid sequence for {chain}. Use {alphabet} (lowercase is accepted) or valid [CCD] entries, without spaces. Sparse entries need resid and residue.") from exc

    def _validate_form_condition(self, condition, chain, canonical):
        source_modality = self._source_modality(chain)
        if source_modality is not None and source_modality != canonical:
            raise ValueError(f"Source chain {chain} is {_MODALITY_NAMES.get(source_modality, source_modality)}. Choose a source that matches the design modality, or use a new chain ID.")
        unknown = set(condition) - {"sequence", "backbone", "contig", "secondary_structure", "ligand", "bond"}
        if unknown:
            raise ValueError(f"Unknown condition fields: {sorted(unknown)}. Check the task syntax.")
        mappings = {}
        for name in ("sequence", "contig", "ligand", "secondary_structure"):
            value = condition.get(name)
            value = {} if value is None else value
            if not isinstance(value, dict):
                raise ValueError(f"condition.{name} must be a YAML mapping.")
            mappings[name] = value
        for name in ("sequence", "contig", "ligand"):
            extra = set(mappings[name]) - {chain}
            if extra:
                raise ValueError(f"The form defines chain {chain}, but condition.{name} refers to {sorted(extra)}. Use full YAML for a multichain task.")
        backbone = condition.get("backbone")
        backbone = [] if backbone is None else backbone
        if not isinstance(backbone, list) or any(value != chain for value in backbone):
            raise ValueError(f"Set condition.backbone to [{chain}]. Use full YAML for multichain conditions.")
        if canonical == "ligand":
            if mappings["sequence"] or mappings["contig"] or mappings["secondary_structure"]:
                raise ValueError("Ligands do not support polymer sequence, contig, or secondary-structure conditions. Remove these fields or use full YAML.")
        elif mappings["ligand"]:
            raise ValueError("This design chain is a polymer and cannot use condition.ligand. Choose Ligand or use full YAML for a mixed task.")
        if mappings["sequence"]:
            self._validate_sequence(mappings["sequence"][chain], canonical, chain)
        contig = mappings["contig"].get(chain)
        if contig is not None:
            if not isinstance(contig, dict):
                raise ValueError("A contig condition needs token and an optional polymer_atom mapping.")
            if backbone:
                raise ValueError("Use either backbone or contig for a design chain.")
            if self._source_modality(chain) != canonical:
                raise ValueError(f"A contig needs a matching source chain named {chain}. Check the input and design chain.")
            atom_scope = contig.get("polymer_atom")
            if atom_scope is not None and not isinstance(atom_scope, dict):
                raise ValueError("polymer_atom must be null or a mapping from residue selectors to atom-name lists or tip.")
            if canonical != "protein" and atom_scope and any(value == "tip" for value in atom_scope.values()):
                raise ValueError("The tip preset is for protein or peptide residues. For RNA or DNA, use available atom names.")
        if backbone and self._source_modality(chain) != canonical:
            raise ValueError(f"Backbone needs a matching source chain named {chain}. Load the appropriate input.")
        ss = mappings["secondary_structure"]
        if set(ss) - {"coarse", "token", "pair"}:
            raise ValueError("Secondary structure supports coarse, token, and pair.")
        for channel in ("coarse", "token", "pair"):
            if ss.get(channel) is not None and not isinstance(ss[channel], dict):
                raise ValueError(f"secondary_structure.{channel} must be a mapping.")
        allowed_coarse = {"alpha", "beta"} if canonical == "protein" else {"paired", "unpaired"}
        allowed_token = {"helix", "sheet", "loop"} if canonical == "protein" else {"paired", "unpaired"}
        allowed_pair = {"adjacent", "not_adjacent"} if canonical == "protein" else {"paired", "unpaired"}
        for target, value in (ss.get("coarse") or {}).items():
            if target != chain or value not in allowed_coarse:
                raise ValueError(f"Global secondary structure for {chain} accepts {sorted(allowed_coarse)}. Use loop only for per-position protein conditions.")
        for target, classes in (ss.get("token") or {}).items():
            if target != chain or not isinstance(classes, dict) or set(classes) - allowed_token or any(not isinstance(value, str) for value in classes.values()):
                raise ValueError(f"Per-position secondary structure for {chain} maps {sorted(allowed_token)} to residue ranges.")
        if set(ss.get("pair") or {}) - allowed_pair:
            raise ValueError(f"Pair conditions accept {sorted(allowed_pair)}. Endpoints also need matching per-position conditions.")
        ligand = mappings["ligand"].get(chain)
        if ligand is not None:
            if not isinstance(ligand, dict) or ligand.get("mode") not in {"smiles", "ccd", "type_bond", "atom_motif"}:
                raise ValueError("Ligand mode must be smiles, ccd, type_bond, or atom_motif. For backbone, use condition.backbone.")
            if backbone:
                raise ValueError("Use either ligand backbone or condition.ligand.")
            mode = ligand["mode"]
            if mode in {"type_bond", "atom_motif"} and self._source_modality(chain) != "ligand":
                raise ValueError(f"{mode} needs a source ligand chain named {chain}. Load it, or use smiles or ccd with a new chain.")
            if mode == "ccd" and self._source_modality(chain) is not None:
                raise ValueError("CCD mode needs a new design chain ID. Use type_bond for an existing source ligand.")
            if mode == "atom_motif":
                names = ligand.get("atoms")
                available = set(str(x) for x in self.atoms.atom_name[self.atoms.chain_id == chain])
                if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names) or len(names) != len(set(names)) or set(names) - available:
                    raise ValueError("atom_motif needs unique atom names present in the source ligand chain. Select the atoms again.")


    def _cif_colors(self, path):
        from biotite.structure.io import pdbx
        category = pdbx.CIFFile.read(path).block.get("atom_site")
        if category is None or "B_iso_or_equiv" not in category:
            return []
        def column(name, default=""):
            return category[name].as_array(str) if name in category else [default] * category.row_count
        groups = {}
        values = column("B_iso_or_equiv")
        for chain, seq, auth, ins, comp, atom, value in zip(
            column("label_asym_id"), column("label_seq_id"), column("auth_seq_id"),
            column("pdbx_PDB_ins_code"), column("label_comp_id"), column("label_atom_id"), values,
        ):
            try:
                number = float(value)
            except ValueError:
                return []
            # A physical B-factor CIF is not interpreted as a conditioning ledger.
            if number not in _CATEGORY_COLORS:
                return []
            key = (int(number), chain, seq, auth, ins, comp)
            groups.setdefault(key, []).append(atom)
        queries = []
        for (number, chain, seq, auth, ins, comp), names in groups.items():
            query = {"struct_asym_id": chain,
                     "atoms": list(dict.fromkeys(names)), "color": _rgb(_CATEGORY_COLORS[number][1])}
            if str(seq) not in {"", ".", "?"}:
                query["residue_number"] = int(seq)
            elif str(auth) not in {"", ".", "?"}:
                query["auth_residue_number"] = int(auth)
                if ins not in {"", ".", "?"}:
                    query["auth_ins_code_id"] = ins
            else:
                # No unique positional address is available; do not broaden a
                # categorical color to every component of the same name.
                return []
            queries.append(query)
        return queries
