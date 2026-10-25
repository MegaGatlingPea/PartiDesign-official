"""Notebook controls for detached native jobs and corresponding structures."""
from __future__ import annotations

from pathlib import Path
import shlex
import asyncio
import contextlib
import io
import re
import time

import ipywidgets as W
import yaml
import anywidget
import traitlets

from . import notebook_runtime as runtime
from .notebook_action import SyncedActionButton


class CommandEditor(anywidget.AnyWidget):
    _esm = Path(__file__).with_name("notebook_command.js")
    value = traitlets.Unicode("").tag(sync=True)
    editable = traitlets.Bool(False).tag(sync=True)


class MethodSelection(W.HBox):
    """Direct-click method choices with the same tuple value as SelectMultiple."""

    value = traitlets.Tuple(default_value=("af3",)).tag(sync=True)
    options = (("AF3", "af3"), ("AF3-LA", "af3-la"), ("AF2-IG", "af2_ig"),
               ("ESMFold2-fast", "esmfold2-fast"),
               ("ESMFold2-standard", "esmfold2-standard"), ("PartiScore", "partiscore"))

    def __init__(self):
        self._updating = False
        self.controls = {
            key: W.Checkbox(description=label, value=key in self.value,
                            indent=False, layout=W.Layout(width="160px", min_width="0"))
            for label, key in self.options
        }
        for control in self.controls.values():
            control.add_class("pd-check")
        choices = W.Box(list(self.controls.values()),
                        layout=W.Layout(display="flex", flex_flow="row wrap", grid_gap="4px 12px",
                                        flex="1 1 0", min_width="0"))
        label = W.Label("Methods", layout=W.Layout(width="94px", min_width="94px"))
        super().__init__([label, choices], layout=W.Layout(width="100%", min_width="0",
                                                         align_items="flex-start", grid_gap="8px"))
        for control in self.controls.values():
            control.observe(self._checked, names="value")
        self.observe(self._sync_checks, names="value")

    @traitlets.validate("value")
    def _valid_value(self, proposal):
        values = proposal["value"]
        allowed = {key for _, key in self.options}
        if len(values) != len(set(values)) or not set(values).issubset(allowed):
            raise traitlets.TraitError("Choose unique supported prediction methods")
        return values

    def _checked(self, change):
        if not self._updating:
            self.value = tuple(key for _, key in self.options if self.controls[key].value)

    def _sync_checks(self, change):
        self._updating = True
        try:
            selected = set(change["new"])
            for key, control in self.controls.items():
                control.value = key in selected
        finally:
            self._updating = False


class NotebookRuntimePanel:
    def __init__(self, editor):
        self.editor = editor
        self.repo_root = editor.repo_root
        self._syncing = False
        self._records = []
        self._jobs = {}
        self._last_output = None
        self._suggested_output = None
        self._task_output = None
        self._monitor_task = None
        self._prepared_command = None
        text, check, actions = editor._text, editor._checkbox, editor._actions
        self.gpu_ids = text("0", "GPU IDs", placeholder="Comma-separated GPU IDs")
        self.cpu_budget = self._int(8, "CPU budget")
        self.resource_note = editor._info("")
        self.compile_enabled = W.Dropdown(
            options=[("Auto", "auto"), ("On (native)", "true"), ("Off (native)", "false")],
            value="auto", description="Compilation",
        )
        self.compile_help = editor._help("Auto: below 100 expanded candidates, use BioIR with CUDA Graph when usable, otherwise native eager. At 100 or more, use native dynamic compilation. On/Off explicitly select native compilation/eager execution.")
        self.batch_size = self._int(16, "Batch size")
        self.data_workers = self._int(4, "Data workers")
        self.output_hint = editor._info("")
        self.command = CommandEditor(layout=W.Layout(width="100%", min_width="0"))
        self.command.add_class("widget-textarea")
        self.command.add_class("pd-command")
        self.execute = check("Execute")
        self.modify_command = check("Modify command")
        self.execute_button = self._action_button("Execute command", self.execute_prepared, {
            "execute": self.execute, "modify_command": self.modify_command, "command": self.command,
            "gpu_ids": self.gpu_ids, "cpu_budget": self.cpu_budget}, "primary")
        self.execute_button.disabled = True
        self.command_note = editor._info("Prepare a command, then copy it or explicitly execute it here.")
        self.prepare_inference_button = self._action_button("Prepare inference", self.prepare_inference, {
            "task_name": editor.task_name, "output_dir": editor.output_dir, "checkpoint": editor.checkpoint,
            "context_mode": editor.context_mode, "compile_enabled": self.compile_enabled,
            "batch_size": self.batch_size, "data_workers": self.data_workers,
            "gpu_ids": self.gpu_ids, "cpu_budget": self.cpu_budget}, "primary", preparation=True)
        inference = W.VBox([
            editor.output_dir, editor.checkpoint, editor.context_mode,
            actions(self.compile_enabled, self.compile_help), self.batch_size,
            self.data_workers,
            actions(self.prepare_inference_button),
        ])
        self.validate_input = text("", "Result directory", placeholder="Task directory or its parent run directory")
        self.methods = MethodSelection()
        self.worker_gpus = self._float(0.5, "GPU / worker")
        self.worker_cpus = self._float(1.0, "CPU / worker")
        self.source = W.Dropdown(options=[("SolubleMPNN", "soluble_mpnn"),
            ("ProteinMPNN", "protein_mpnn"), ("LigandMPNN", "ligand_mpnn"),
            ("CyclicMPNN", "cyclic_mpnn"), ("NA-MPNN", "na_mpnn")],
            value="soluble_mpnn", description="Redesign model", style={"description_width": "94px"},
            layout=W.Layout(width="100%", min_width="0"))
        self.ordinary = check("Ordinary redesign", True)
        self.ordinary_count = self._int(1, "Sequences")
        self.ordinary_temperature = self._float(0.1, "Temperature")
        self.fi = check("Fixed-interface redesign")
        self.fi_count = self._int(1, "FI sequences")
        self.fi_temperature = self._float(0.1, "FI temperature")
        self.fi_cutoff = self._float(6.0, "FI cutoff (Å)")
        self.prediction_samples = self._int(1, "Predictions")
        self.recycles = self._int(10, "AF3 recycles")
        self.af2_recycles = self._int(3, "AF2 recycles")
        self.seed = self._int(101, "Seed")
        self.loops = self._int(20, "ESM loops")
        self.steps = self._int(100, "ESM steps")
        self.msa_mode = W.Dropdown(options=[("Off", "off"), ("Auto / reuse cache", "auto"),
            ("Cache only", "cache-only")], value="auto", description="Context MSA",
            style={"description_width": "94px"}, layout=W.Layout(width="100%", min_width="0"))
        self.templates = check("Use templates", True)
        self._criteria_catalog, self._criteria_errors = self._load_criteria()
        self.criteria = W.Dropdown(
            options=[("No criteria", ""), *[(item["label"], key)
                for key, item in self._criteria_catalog.items()]],
            value="", description="Preset", style={"description_width": "94px"},
            layout=W.Layout(width="100%", min_width="0"))
        self.criteria_help = editor._help("")
        criteria_row = actions(self.criteria, self.criteria_help)
        criteria_row.add_class("pd-preset-row")
        self.score_refolds = check("Calculate PartiScore")
        self.sequence_options = W.VBox([editor._info("Original · 1 sequence (always retained)"), self.source,
            self.ordinary, self.ordinary_count, self.ordinary_temperature,
            self.fi, self.fi_count, self.fi_temperature, self.fi_cutoff])
        validation_details = W.Accordion(children=[self.sequence_options,
            W.VBox([self.prediction_samples, self.recycles, self.af2_recycles, self.seed,
                    self.loops, self.steps, self.msa_mode, self.templates, self.score_refolds]),
            W.VBox([criteria_row])], selected_index=0)
        for i, name in enumerate(["Sequence branches", "Prediction options", "Success criteria (optional)"]):
            validation_details.set_title(i, name)
        self.validation_details = validation_details
        validation_bindings = {f"method:{name}": control for name, control in self.methods.controls.items()}
        validation_bindings.update({name: getattr(self, name) for name in (
            "gpu_ids", "cpu_budget", "validate_input", "worker_gpus", "worker_cpus", "source",
            "ordinary", "ordinary_count", "ordinary_temperature", "fi", "fi_count",
            "fi_temperature", "fi_cutoff", "prediction_samples", "recycles", "af2_recycles",
            "seed", "loops", "steps", "msa_mode", "templates", "criteria", "score_refolds")})
        self.prepare_validation_button = self._action_button("Prepare validation", self.prepare_validation,
            validation_bindings, "primary", preparation=True)
        validation = W.VBox([self.validate_input, self.methods, self.worker_gpus, self.worker_cpus,
            validation_details, actions(self.prepare_validation_button)])
        self.cluster_input = text("", "Result directory")
        self.cluster_method = text("", "Method", placeholder="All available validation methods if blank")
        self.cluster_workers = self._int(16, "Concurrency")
        self.cluster_threshold = self._float(0.5, "TM threshold")
        self.prepare_cluster_button = self._action_button("Prepare clustering", self.prepare_cluster,
            {name: getattr(self, name) for name in ("cluster_input", "cluster_method", "cluster_workers",
                "cluster_threshold", "gpu_ids", "cpu_budget")}, "primary", preparation=True)
        cluster = W.VBox([self.cluster_input, self.cluster_method, self.cluster_workers, self.cluster_threshold,
            editor._help("Cluster original successful design backbones. Each validation method and success cohort is handled separately. At most 16 directories run concurrently."),
            actions(self.prepare_cluster_button)])
        self.results_input = text("", "Result directory")
        self.raw_sample = self._dropdown("Raw sample")
        self.result_method = self._dropdown("Method")
        self.refold = self._dropdown("Successful refold")
        self.result_note = editor._info("Choose a result directory, then refresh samples.")
        self.result_file = text("", "Result file", placeholder="Optional PDB or CIF")
        self.result_upload = W.FileUpload(accept=".pdb,.cif,.mmcif", multiple=False,
            description="Choose result", layout=W.Layout(width="auto", min_width="0"))
        self.open_file = W.Accordion(children=[W.VBox([self.result_file,
            actions(self._button("Open result file", self.open_result_file), self.result_upload,
                    self._button("Open upload", self.open_upload))])], selected_index=None)
        self.open_file.set_title(0, "Open a result file")
        results = W.VBox([self.results_input,
            self._button("Refresh samples", self.refresh_results), self.raw_sample,
            self.result_method, self.refold,
            actions(self._button("Preview raw sample", self.preview_raw),
                    self._button("Preview refold", self.preview_refold, "primary")),
            self.result_note, self.open_file])
        self.sections = W.Accordion(children=[inference, validation, cluster, results], selected_index=0)
        for i, title in enumerate(["Inference", "Validate", "Cluster", "Results & preview"]):
            self.sections.set_title(i, title)
        self.jobs = self._dropdown("Background job")
        self.job_note = editor._info("No notebook jobs found.")
        self.stop_button = self._button("Stop selected job", self.stop_selected)
        self.stop_button.disabled = True
        self.logs = editor._area_text(rows=7, disabled=True)
        self.logs.add_class("pd-runtime-log")
        job_details = W.Accordion(children=[W.VBox([self.jobs,
            actions(self._button("Refresh jobs & logs", self.refresh_jobs), self.stop_button),
            self.job_note, self.logs])], selected_index=None)
        job_details.set_title(0, "Background jobs & logs")
        self.job_details = job_details
        self.command_box = W.Accordion(children=[W.VBox([
            actions(self.execute, self.modify_command, editor._help("Execute is off by default: copy the command to run it yourself. Enable Execute and click Execute command to launch in tmux. Modify command makes the text editable; edits never execute automatically. Enter one executable and its arguments; use an explicit shell invocation for shell scripts.")),
            self.command, self.command_note, self.execute_button])], selected_index=None)
        self.command_box.set_title(0, "Command & execution")
        self.widget = W.VBox([
            actions(editor._heading("Runtime", small=True), editor._help("Run tasks in detached tmux sessions. Jobs survive a disconnected notebook or SSH connection. Refresh jobs to reconnect; selecting or previewing a result never starts a model.")),
            self.gpu_ids, self.cpu_budget, self.resource_note, self.sections, self.command_box, job_details,
        ], layout=W.Layout(width="100%", min_width="0", grid_gap="8px"))
        self.widget.add_class("pd-runtime")
        self.execute.observe(lambda ch: self._execution_controls(), names="value")
        self.modify_command.observe(lambda ch: self._execution_controls(), names="value")
        self.raw_sample.observe(self._raw_changed, names="value")
        self.result_method.observe(self._method_changed, names="value")
        self.jobs.observe(lambda ch: self._show_job(), names="value")
        self.methods.observe(lambda ch: self._prediction_controls(), names="value")
        self.criteria.observe(lambda ch: self._criteria_controls(), names="value")
        self.ordinary.observe(lambda ch: self._branch_controls(), names="value")
        self.fi.observe(lambda ch: self._branch_controls(), names="value")
        self.source.observe(self._source_changed, names="value")
        editor.output_dir.observe(lambda ch: self.sync_task(), names="value")
        editor.task_name.observe(lambda ch: self.sync_task(), names="value")
        self._prediction_controls()
        self._branch_controls()
        self.refresh_resources()
        self.refresh_jobs()

    @staticmethod
    def _int(value, description):
        return W.IntText(value=value, description=description, style={"description_width": "94px"},
            layout=W.Layout(width="100%", min_width="0"))

    @staticmethod
    def _float(value, description):
        return W.FloatText(value=value, description=description, style={"description_width": "94px"},
            layout=W.Layout(width="100%", min_width="0"))

    @staticmethod
    def _dropdown(description):
        return W.Dropdown(options=[], description=description, style={"description_width": "94px"},
            layout=W.Layout(width="100%", min_width="0"))

    def _button(self, label, action, style=""):
        return self.editor._button(label, action, style)

    def _action_button(self, label, action, bindings, style="", *, preparation=False):
        def failed(exc):
            if preparation:
                self._clear_prepared()
            self.editor._status(f"Please check: {exc}")
        button = SyncedActionButton(description=label, button_style=style, bindings=bindings,
            on_error=failed, layout=W.Layout(width="auto", min_width="100px",
                margin="4px 6px 4px 0"))
        button.on_click(lambda _: action())
        return button

    def _clear_prepared(self):
        self._prepared_command = None
        self._prepared_design = None
        self.command.value = ""
        self.command_note.value = "Prepare a command, then copy it or explicitly execute it here."
        self._execution_controls()

    def refresh_resources(self):
        try:
            info = runtime.resources()
            cpus = int(info.get("cpu_count", info.get("cpus", 1)))
            self.cpu_budget.value = max(1, min(8, cpus))
            gpus = info.get("gpus", [])
            if gpus:
                self.gpu_ids.value = str(gpus[0].get("index", gpus[0].get("id", 0)))
            else:
                self.gpu_ids.value = ""
            self.resource_note.value = f"{len(gpus)} GPUs · {cpus} available CPUs"
        except Exception as exc:
            self.resource_note.value = f"Resource discovery: {exc}"

    def sync_task(self, reset_sources=False):
        name = self.editor.task_name.value.strip() or "my_design"
        self.output_hint.value = str(self.editor._path(self.editor.output_dir.value) / name)
        previous = self._suggested_output
        destination = self.output_hint.value
        if (reset_sources or destination != self._task_output) and hasattr(self, "execute_button"):
            self._clear_prepared()
        if destination == self._task_output and previous:
            destination = previous
        for control in (self.validate_input, self.cluster_input, self.results_input):
            if not control.value or control.value == previous:
                control.value = destination
        self._task_output = self.output_hint.value
        self._suggested_output = destination
        if reset_sources:
            modality = self.editor.modality.value
            payload = self.editor._imported_task[1] if self.editor._imported_task else {}
            cyclic = bool(payload.get("cyclic_offset", self.editor.cyclic_offset.value))
            source = "na_mpnn" if modality in {"rna", "dna"} else "cyclic_mpnn" if cyclic else "soluble_mpnn"
            self.source.value = source
            self.ordinary.disabled = self.fi.disabled = modality == "ligand"
            self.ordinary.value = modality != "ligand"
            self.fi.value = False
            self._branch_controls()

    def _source_changed(self, change):
        value = 0.0001 if change["new"] in {"protein_mpnn", "cyclic_mpnn"} else 0.1
        self.ordinary_temperature.value = self.fi_temperature.value = value

    def _branch_controls(self):
        for control in (self.ordinary_count, self.ordinary_temperature):
            control.disabled = not self.ordinary.value
        for control in (self.fi_count, self.fi_temperature, self.fi_cutoff):
            control.disabled = not self.fi.value

    def _prediction_controls(self):
        methods = set(self.methods.value)
        self.af2_recycles.layout.display = "" if "af2_ig" in methods else "none"
        self.recycles.layout.display = "" if methods.intersection({"af3", "af3-la"}) else "none"
        for control in (self.loops, self.steps):
            control.layout.display = "" if methods.intersection({"esmfold2-fast", "esmfold2-standard"}) else "none"
        for control in (self.msa_mode, self.templates):
            control.layout.display = "" if methods.intersection({"af3", "af3-la"}) else "none"
        self._criteria_controls()

    def _load_criteria(self):
        from src.validate.filters import load_presets

        catalog, errors = {}, []
        files = (("Project", self.repo_root / "configs" / "criteria.yaml"),
                 ("Custom", self.repo_root / "configs" / "examples" / "custom_presets.yaml"))
        for label, path in files:
            if not path.is_file():
                continue
            try:
                presets = load_presets(path)
            except (OSError, ValueError, yaml.YAMLError) as exc:
                errors.append(f"{path.name}: {exc}")
                continue
            for name, preset in presets.items():
                if not set(preset["methods"]).intersection(value for _, value in MethodSelection.options):
                    continue
                key = name if name not in catalog else f"{label.lower()}:{name}"
                catalog[key] = {"name": name, "label": f"{label} · {name}",
                                "definition": preset, "path": path}
        return catalog, errors

    def _criteria_controls(self):
        help_text = ("Choose an existing success preset, or leave No criteria selected. "
                     "With no criteria, correctly completed predictions enter success; "
                     "failed or incomplete predictions remain unsuccessful.")
        item = self._criteria_catalog.get(self.criteria.value)
        if item:
            definition = item["definition"]
            supported = set(definition["methods"])
            inactive = set(self.methods.value) - supported
            help_text += f" Selected: {item['name']}. Supported methods: {', '.join(definition['methods'])}."
            if inactive:
                help_text += f" This preset does not apply to: {', '.join(sorted(inactive))}."
            help_text += " Rules: " + "; ".join(
                f"{rule['metric']} {rule['op']} {rule['value']} ({rule.get('scope', 'complex')})"
                for rule in definition["rules"])
        if self._criteria_errors:
            help_text += " Unavailable preset files: " + "; ".join(self._criteria_errors)
        self.criteria_help.help_text = help_text

    def _snapshot_dir(self, kind):
        path = self.repo_root / ".cache" / "notebook_runtime" / "inputs" / f"{kind}-{time.time_ns()}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _snapshot_task(self):
        source = self.editor.export_task()
        name = self.editor._task()[0]
        path = runtime.freeze_task_inputs(self.repo_root, source, self._snapshot_dir("design"))
        return name, path

    def prepare_inference(self):
        self._clear_prepared()
        name, path = self._snapshot_task()
        argv = runtime.build_design_command(self.repo_root, path, name,
            self.editor._path(self.editor.output_dir.value), compile_enabled=self.compile_enabled.value,
            checkpoint=self.editor.checkpoint.value.strip() or None,
            batch_size=self.batch_size.value, num_workers=self.data_workers.value)
        if self.editor.context_mode.value != "auto":
            argv += ["--override", f"model.context_constraint_mode={self.editor.context_mode.value}"]
        self._show_command(argv, "inference", self.editor._path(self.editor.output_dir.value) / name, name)
        self._prepared_design = (name, argv)
        self.editor._status("Inference command prepared. Nothing is running.")
        return argv

    def _show_command(self, argv, kind, output, task_name=None):
        self.command.value = shlex.join(argv)
        self._prepared_command = {"argv": list(argv), "kind": kind, "output": str(output), "task_name": task_name}
        self.command_note.value = f"Prepared {kind} · {task_name or output}"
        self.command_box.selected_index = 0
        self._execution_controls()

    def _execution_controls(self):
        self.command.editable = self.modify_command.value
        self.execute_button.disabled = not self.execute.value or self._prepared_command is None

    def _command_context(self, argv):
        action, start = None, None
        executable = Path(argv[0]).name
        python = bool(re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", executable))
        if executable in {"partidesign", "partid"} and len(argv) > 1:
            action, start = argv[1], 2
        elif python and len(argv) > 3 and argv[1:3] == ["-m", "partidesign"]:
            action, start = argv[3], 4
        elif python and len(argv) > 2 and argv[1] == "-m" and argv[2] in {"src.design", "src.validate", "src.cluster"}:
            action, start = argv[2].split(".")[-1], 3
        kind = {"design": "inference", "validate": "validate", "cluster": "cluster"}.get(action, "custom")
        if kind == "custom":
            return kind, None, None
        if any(token in {"--show-config", "--plan", "--help", "-h"} for token in argv):
            return "inspection", None, None
        from partidesign.cli import build_parser
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                args = build_parser().parse_args([action, *argv[start:]])
        except SystemExit:
            return kind, None, None
        if kind == "inference":
            if args.out is None:
                return kind, None, None
            output = self.editor._path(args.out)
            tasks = args.task
            if not tasks:
                try:
                    tasks = list(yaml.safe_load(self.editor._path(args.task_file).read_text()))
                except (OSError, TypeError, yaml.YAMLError):
                    return kind, None, None
            task = tasks[0] if len(tasks) == 1 else None
            return kind, output / task if task else output, task
        paths = args.runs if kind == "validate" else args.paths
        maintenance = kind == "validate" and (getattr(args, "refilter", False) or getattr(args, "recompute_metrics", False))
        if len(paths) != 1:
            return "maintenance" if maintenance else kind, None, None
        if kind == "validate":
            raw = Path(paths[0]).expanduser()
            local = self.repo_root / "inference" / raw
            bare = not raw.is_absolute() and "/" not in str(paths[0]) and "\\" not in str(paths[0])
            direct = raw if raw.is_absolute() else self.repo_root / raw
            output = local if bare and local.is_dir() else direct if direct.exists() else self.repo_root / "inference" / raw
            if maintenance:
                return "maintenance", output.resolve(), None
            return kind, output.resolve(), None
        return kind, self.editor._path(paths[0]), None

    def execute_prepared(self):
        if not self.execute.value:
            self.editor._status("Execution is off. Copy the command to run it yourself.")
            return None
        if self._prepared_command is None:
            raise ValueError("Prepare a command first.")
        try:
            argv = shlex.split(self.command.value)
        except ValueError as exc:
            raise ValueError(f"Check command quoting: {exc}") from exc
        if not argv:
            raise ValueError("The command is empty.")
        if any(token in {"|", "||", "&&", ";", ">", ">>", "&"} for token in argv):
            raise ValueError("Enter one command. For a shell script, explicitly use bash -lc with a quoted script.")
        if not self.modify_command.value and argv != self._prepared_command["argv"]:
            raise ValueError("Enable Modify command before changing the prepared command.")
        kind, output, name = self._command_context(argv)
        if kind == "inference" and name is not None and output is not None and output.is_dir() and any(output.iterdir()):
            raise ValueError("This output already exists. Choose another task name or output root.")
        job = self._launch(kind, argv, output, task_name=name)
        if kind == "inference" and output is not None:
            self._last_output = str(output)
            self._task_output = str(self.editor._path(self.editor.output_dir.value) /
                                    (self.editor.task_name.value.strip() or "my_design"))
            self._suggested_output = str(output)
            for control in (self.validate_input, self.cluster_input, self.results_input):
                control.value = str(output)
        elif output is None:
            self.command_note.value = "Custom or inspection command: select its result directory manually after completion."
        return job

    def _resources(self):
        if self.cpu_budget.value < 1:
            raise ValueError("CPU budget must be positive.")
        gpu_ids = [int(value.strip()) for value in self.gpu_ids.value.split(",") if value.strip()]
        if any(value < 0 for value in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("Choose unique, nonnegative GPU IDs.")
        return gpu_ids, self.cpu_budget.value

    def _launch(self, kind, argv, output, task_name=None):
        gpu_ids, cpus = self._resources()
        if kind == "validate" and not gpu_ids:
            raise ValueError("Choose at least one GPU for model validation.")
        for previous in runtime.list_jobs(self.repo_root):
            if previous.get("state", previous.get("status")) == "running" and previous.get("kind") == kind and previous.get("output_dir") == str(output):
                raise ValueError("A matching background job is already running. Select it in Background jobs & logs.")
        job = runtime.launch_job(self.repo_root, kind, argv, gpu_ids=[] if kind in {"cluster", "maintenance", "inspection"} else gpu_ids,
            cpu_budget=cpus, output_dir=output, task_name=task_name)
        self.refresh_jobs()
        session = job.get("session", job.get("session_name"))
        if session in self._jobs:
            self.jobs.value = session
        self.job_details.selected_index = 0
        self.editor._status(f"{kind.title()} started in tmux: {session}")
        return job

    def run_inference(self):
        self.prepare_inference()
        return self.execute_prepared()

    def _sources(self):
        sources = [{"source": "gt", "count": 1, "label": "self"}]
        if set(self.methods.value) == {"partiscore"}:
            return sources
        if "partiscore" in self.methods.value and (self.ordinary.value or self.fi.value):
            raise ValueError("Run PartiScore separately when redesign branches are enabled.")
        for enabled, count, temperature, fixed in [
            (self.ordinary.value, self.ordinary_count.value, self.ordinary_temperature.value, False),
            (self.fi.value, self.fi_count.value, self.fi_temperature.value, True),
        ]:
            if not enabled:
                continue
            if count < 1 or temperature <= 0:
                raise ValueError("Sequence counts and temperatures must be positive.")
            settings = {"temperature": temperature}
            if fixed:
                if self.fi_cutoff.value <= 0:
                    raise ValueError("FI cutoff must be positive.")
                settings["fixed_interface"] = {"enabled": True, "cutoff": self.fi_cutoff.value}
            sources.append({"source": self.source.value, "count": count,
                "label": self.source.value + ("-fi" if fixed else ""), "settings": settings})
        return sources

    def prepare_validation(self):
        self._clear_prepared()
        if not self.methods.value:
            raise ValueError("Select at least one prediction method.")
        if self.worker_cpus.value > self.cpu_budget.value:
            raise ValueError("CPU per worker cannot exceed the total CPU budget.")
        options = {
            "af2": {"models": [1], "samples": self.prediction_samples.value,
                    "recycles": self.af2_recycles.value, "seed": self.seed.value},
            "af3": {"samples": self.prediction_samples.value, "recycles": self.recycles.value,
                    "seed": self.seed.value},
            "esmfold2": {"samples": self.prediction_samples.value, "loops": self.loops.value,
                    "steps": self.steps.value, "seed": self.seed.value},
            "sequence": {"seed": self.seed.value},
            "scorer": {"enabled": self.score_refolds.value},
        }
        if set(self.methods.value).intersection({"af3", "af3-la"}):
            options["af3"]["msa"] = {"mode": self.msa_mode.value,
                "templates": self.templates.value}
        selected = self._criteria_catalog.get(self.criteria.value)
        if selected and not set(self.methods.value).intersection(selected["definition"]["methods"]):
            raise ValueError(f"Success preset {selected['name']} does not apply to the selected methods.")
        snapshot = self._snapshot_dir("validate")
        path = runtime.write_validate_config(snapshot / "validate.yaml",
            list(self.methods.value), self._sources(),
            criteria=[selected["name"]] if selected else [],
            worker_gpus=self.worker_gpus.value, worker_cpus=self.worker_cpus.value,
            method_options=options)
        argv = runtime.build_validate_command(self.repo_root, self.editor._path(self.validate_input.value), path)
        if selected:
            presets_path = snapshot / "presets.yaml"
            presets_path.write_text(yaml.safe_dump({"presets": {
                selected["name"]: selected["definition"]}}, sort_keys=False))
            argv += ["--presets", str(presets_path)]
        self._show_command(argv, "validate", self.editor._path(self.validate_input.value))
        self.editor._status("Validation command prepared. Nothing is running.")
        return argv

    def run_validate(self):
        self.prepare_validation()
        return self.execute_prepared()

    def prepare_cluster(self):
        self._clear_prepared()
        if not 1 <= self.cluster_workers.value <= 16:
            raise ValueError("Clustering concurrency must be between 1 and 16.")
        argv = runtime.build_cluster_command(self.repo_root, self.editor._path(self.cluster_input.value),
            method=self.cluster_method.value.strip() or None, workers=self.cluster_workers.value,
            threshold=self.cluster_threshold.value)
        self._show_command(argv, "cluster", self.editor._path(self.cluster_input.value))
        self.editor._status("Cluster command prepared. Nothing is running.")
        return argv

    def run_cluster(self):
        self.prepare_cluster()
        return self.execute_prepared()

    def refresh_jobs(self):
        current = self.jobs.value
        jobs = runtime.list_jobs(self.repo_root)
        self._jobs = {job.get("session", job.get("session_name")): job for job in jobs}
        self.jobs.options = [(f"{job.get('kind', 'job')} · {job.get('task_name') or key}", key)
                            for key, job in self._jobs.items()]
        self.jobs.value = current if current in self._jobs else (next(iter(self._jobs)) if self._jobs else None)
        self._show_job()
        self._start_monitor()

    def _start_monitor(self):
        if self._monitor_task is not None and not self._monitor_task.done():
            return
        if not any(job.get("state", job.get("status")) == "running" for job in self._jobs.values()):
            return
        try:
            self._monitor_task = asyncio.get_running_loop().create_task(self._monitor())
        except RuntimeError:
            pass

    async def _monitor(self):
        while any(job.get("state", job.get("status")) == "running" for job in self._jobs.values()):
            await asyncio.sleep(4)
            try:
                self.refresh_jobs()
            except Exception as exc:
                self.job_note.value = f"Refresh jobs to reconnect: {exc}"
                return

    def _show_job(self):
        if not self.jobs.value:
            self.job_note.value = "No notebook jobs found."
            self.stop_button.disabled = True
            return
        job = runtime.job_status(self.repo_root, self.jobs.value)
        self._jobs[self.jobs.value] = job
        self.job_note.value = f"{job.get('state', job.get('status', 'unknown'))} · {self.jobs.value}"
        self.logs.value = runtime.tail_log(job, lines=80)
        self.stop_button.disabled = job.get("state", job.get("status")) not in {"running", "starting"}

    def stop_selected(self):
        if not self.jobs.value:
            raise ValueError("Select a background job first.")
        runtime.stop_job(self.repo_root, self.jobs.value)
        self.refresh_jobs()

    def refresh_results(self):
        self._records = runtime.discover_results(self.editor._path(self.results_input.value))
        self._syncing = True
        try:
            samples = {}
            unavailable = 0
            for record in self._records:
                path = record.get("raw_path")
                if not path or not Path(path).is_file():
                    unavailable += record.get("kind") == "raw"
                    continue
                key = str(path)
                samples.setdefault(key, f"{record.get('task', '')} · {record.get('sample_label', Path(key).stem)}")
            self.raw_sample.options = [(label, key) for key, label in samples.items()]
            self.raw_sample.value = next(iter(samples), None)
        finally:
            self._syncing = False
        self._raw_changed()
        self.editor._status(f"Loaded {len(samples)} raw samples" + (f" · {unavailable} unavailable structures" if unavailable else "") + ".")

    def _raw_changed(self, change=None):
        if self._syncing:
            return
        records = [row for row in self._records if str(row["raw_path"]) == self.raw_sample.value]
        methods = list(dict.fromkeys(row.get("method") for row in records if row.get("method")))
        self._syncing = True
        try:
            self.result_method.options = [(method, method) for method in methods]
            self.result_method.value = methods[0] if methods else None
        finally:
            self._syncing = False
        self._method_changed()

    def _method_changed(self, change=None):
        if self._syncing:
            return
        options = []
        for index, row in enumerate(self._records):
            if str(row["raw_path"]) != self.raw_sample.value or row.get("method") != self.result_method.value:
                continue
            if row.get("success") and row.get("refold_path"):
                label = row.get("sequence_label") or row.get("source") or "Original"
                options.append((f"{label} · {Path(row['refold_path']).name}", index))
        self.refold.options = options
        self.refold.value = options[0][1] if options else None
        self.result_note.value = f"{len(options)} successful refolds for this raw sample and method."

    def preview_raw(self):
        if not self.raw_sample.value:
            raise ValueError("Refresh samples and select a raw sample first.")
        self.editor.load_result(self.raw_sample.value)
        self.editor.viewer_note.value = f"Raw sample · {Path(self.raw_sample.value).name}"

    def preview_refold(self):
        if self.refold.value is None:
            raise ValueError("This raw sample has no successful refold in the selected method.")
        row = self._records[self.refold.value]
        self.editor.load_result(row["refold_path"], refold=True)
        self.editor.viewer_note.value = f"{row.get('method')} · {row.get('sequence_label') or row.get('source')} · {Path(row['refold_path']).name}"

    def open_result_file(self):
        self.editor.load_result(self.result_file.value)

    def open_upload(self):
        self.editor.load_result(self.editor._uploaded(self.result_upload))

    def close(self):
        if self._monitor_task is not None:
            self._monitor_task.cancel()
        self.widget.close()
