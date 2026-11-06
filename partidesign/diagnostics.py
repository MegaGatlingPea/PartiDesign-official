"""Read-only hardware, environment and tool inventory; no model execution."""
from __future__ import annotations

import csv
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys

from .paths import code_root, community_source, native_bin_dir, resource_root

PROC_ROOT = Path("/proc")
CGROUP_ROOT = Path("/sys/fs/cgroup")


def format_bytes(value):
    if value is None:
        return "unavailable"
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(value) < 1024 or unit == "PiB":
            return f"{value:,.1f} {unit}"
        value /= 1024


def _text(path):
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return None


def _row(label, value, warning=False):
    return {"label": str(label), "value": str(value), "level": "WARN" if warning else "INFO"}


def _ancestors(base, relative):
    base, relative = Path(base), Path(relative.lstrip("/"))
    member = base / relative if ".." not in relative.parts else base
    if not member.is_dir():
        member = base  # Container mounts may already be rooted at membership.
    result = [member]
    while member != base and base in member.parents:
        member = member.parent
        result.append(member)
    return result


def _cgroup_info(proc=PROC_ROOT, base=CGROUP_ROOT):
    result = {"cpu_quota_cores": None, "memory_limit_bytes": None,
              "memory_available_bytes": None, "cpu_observed": False,
              "memory_observed": False, "memory_usage_complete": True,
              "notes": [], "version": "not detected"}
    raw = _text(Path(proc) / "self/cgroup")
    if raw is None:
        result["notes"].append("cgroup membership unreadable; process limits are unverified")
        return result
    entries = [line.split(":", 2) for line in raw.splitlines() if line.count(":") >= 2]
    unified = next((rel for identity, controllers, rel in entries if identity == "0" and not controllers), None)
    quotas, limits, remaining = [], [], []

    def collect(paths, cpu_file=None, memory_file=None, usage_file=None, v2=False):
        for path in paths:
            if cpu_file and (value := _text(path / cpu_file)) is not None:
                result["cpu_observed"] = True
                try:
                    if v2:
                        quota, period = value.split()[:2]
                        quota = None if quota == "max" else int(quota) / int(period)
                    else:
                        quota = int(value)
                        quota = quota / int(_text(path / "cpu.cfs_period_us")) if quota > 0 else None
                    if quota is not None and quota > 0:
                        quotas.append(quota)
                except (ValueError, TypeError, ZeroDivisionError):
                    result["notes"].append(f"invalid CPU quota at {path}")
            if memory_file and (value := _text(path / memory_file)) is not None:
                result["memory_observed"] = True
                try:
                    limit = None if value.strip() == "max" else int(value)
                    if limit is not None and 0 < limit < (1 << 60):
                        limits.append(limit)
                        usage = _text(path / usage_file)
                        if usage is None:
                            result["memory_usage_complete"] = False
                            result["notes"].append(f"memory usage unreadable at {path}")
                        else:
                            remaining.append(max(0, limit - int(usage)))
                except (ValueError, TypeError):
                    result["memory_usage_complete"] = False
                    result["notes"].append(f"invalid memory accounting at {path}")

    if unified is not None:
        result["version"] = "cgroup v2"
        collect(_ancestors(base, unified), "cpu.max", "memory.max", "memory.current", True)
    else:
        result["version"] = "cgroup v1"
        for _, controllers, rel in entries:
            names = controllers.split(",")
            if "cpu" in names:
                candidates = [Path(base) / controllers, Path(base) / "cpu", Path(base) / "cpu,cpuacct"]
                root = next((p for p in candidates if p.is_dir()), candidates[0])
                collect(_ancestors(root, rel), cpu_file="cpu.cfs_quota_us")
            if "memory" in names:
                collect(_ancestors(Path(base) / "memory", rel), memory_file="memory.limit_in_bytes", usage_file="memory.usage_in_bytes")
    result.update(cpu_quota_cores=min(quotas) if quotas else None,
                  memory_limit_bytes=min(limits) if limits else None,
                  memory_available_bytes=min(remaining) if remaining else None)
    for controller in ("cpu", "memory"):
        if not result[controller + "_observed"]:
            result["notes"].append(controller + " cgroup limit was not readable")
    return result


def _cpu_report(cgroup, proc=PROC_ROOT):
    raw = _text(Path(proc) / "cpuinfo")
    blocks = [{key.strip().lower(): value.strip() for line in block.splitlines() if ":" in line
               for key, value in [line.split(":", 1)]} for block in (raw or "").split("\n\n")]
    model = next((b.get("model name") or b.get("hardware") for b in blocks if b.get("model name") or b.get("hardware")), None)
    model = model or platform.processor() or "unavailable"
    physical = {(b["physical id"], b["core id"]) for b in blocks if "physical id" in b and "core id" in b}
    logical = os.cpu_count()
    try:
        affinity_indices = sorted(os.sched_getaffinity(0))
        affinity, error = len(affinity_indices), None
    except (AttributeError, OSError) as exc:
        affinity_indices = None
        affinity, error = None, str(exc)
    quota = cgroup["cpu_quota_cores"]
    limits = [float(v) for v in (logical, affinity, quota) if v is not None]
    effective = min(limits) if limits else None
    rows = [_row("CPU model", model, model == "unavailable"),
            _row("Physical cores", len(physical) if physical else "unavailable", not physical),
            _row("Logical CPUs reported", logical if logical is not None else "unavailable", logical is None),
            _row("CPUs allowed by affinity", affinity if affinity is not None else "unavailable", affinity is None),
            _row("CPU quota", f"{quota:g} cores" if quota is not None else "no finite quota" if cgroup["cpu_observed"] else "unavailable", not cgroup["cpu_observed"]),
            _row("Usable CPU budget", f"{effective:g} cores (minimum reported/affinity/quota bound; unobserved limits are not guaranteed)" if effective is not None else "unavailable", effective is None or affinity is None or not cgroup["cpu_observed"])]
    if error:
        rows.append(_row("Affinity probe", error, True))
    try:
        rows.append(_row("Host load average", ", ".join(f"{v:.2f}" for v in os.getloadavg()) + " (1/5/15 min; not free CPU count)"))
    except (AttributeError, OSError):
        rows.append(_row("Host load average", "unavailable", True))
    return {"title": "CPU capacity", "rows": rows}, {"logical_cpus": logical, "affinity_cpus": affinity, "affinity_indices": affinity_indices,
            "quota_cores": quota, "effective_cores": effective, "model": model, "physical_cores": len(physical) or None}


def _memory_report(cgroup, proc=PROC_ROOT):
    values = {}
    for line in (_text(Path(proc) / "meminfo") or "").splitlines():
        key, separator, value = line.partition(":")
        if separator:
            try:
                values[key] = int(value.split()[0]) * 1024
            except (ValueError, IndexError):
                pass
    total, available = values.get("MemTotal"), values.get("MemAvailable")
    rows = []
    if available is None and "MemFree" in values:
        available = sum(values.get(key, 0) for key in ("MemFree", "Buffers", "Cached"))
        rows.append(_row("Available-memory method", "free+buffers+cache estimate; MemAvailable absent", True))
    if total is None:
        try:
            total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            if total <= 0 or available < 0:
                total = available = None
                raise ValueError("sysconf memory sizes are unavailable")
            rows.append(_row("Memory method", "sysconf pages; available memory is an estimate", True))
        except (AttributeError, OSError, ValueError):
            rows.append(_row("Memory probe", "no readable host memory accounting", True))
    limit, remaining = cgroup["memory_limit_bytes"], cgroup["memory_available_bytes"]
    totals = [v for v in (total, limit) if v is not None]
    effective = min(totals) if totals else None
    bounds = [v for v in (available, remaining) if v is not None]
    free = min(bounds) if bounds else None
    if limit is not None and not cgroup.get("memory_usage_complete", True):
        free = None
    if free is not None and effective is not None:
        free = min(free, effective)
    rows += [_row("Host RAM", format_bytes(total), total is None), _row("Host RAM available", format_bytes(available), available is None),
             _row("Cgroup RAM limit", format_bytes(limit) if limit is not None else "no finite limit" if cgroup["memory_observed"] else "unavailable", not cgroup["memory_observed"]),
             _row("RAM usable by this process", format_bytes(effective), effective is None),
             _row("RAM currently available", format_bytes(free), free is None)]
    meter = {"label": "RAM", "total_bytes": effective, "available_bytes": free,
             "used_bytes": max(0, effective - free) if effective is not None and free is not None else None,
             "scope": "minimum of host and visible cgroup availability; shared, not reserved"}
    return {"title": "Memory", "rows": rows}, meter, {"host_total_bytes": total, "host_available_bytes": available,
            "total_bytes": effective, "available_bytes": free}


def _run(command, timeout=12, env=None):
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env)


def _torch_gpu_probe():
    if importlib.util.find_spec("torch") is None:
        return {"error": "PyTorch not discoverable", "count": None, "available": None}
    code = """import json,torch
result={'version':torch.__version__,'cuda_build':torch.version.cuda,
        'available':torch.cuda.is_available(),'count':torch.cuda.device_count(),'devices':[]}
if result['available']:
    for index in range(result['count']):
        try:
            item=torch.cuda.get_device_properties(index)
            result['devices'].append({'ordinal':index,'name':item.name,'total_bytes':item.total_memory,
                                      'uuid':str(getattr(item,'uuid',''))})
        except Exception as error:
            result['devices'].append({'ordinal':index,'error':str(error)})
print(json.dumps(result))
"""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTORCH_NVML_BASED_CUDA_CHECK="1")
    try:
        result = _run([sys.executable, "-B", "-c", code], timeout=20, env=env)
        if result.returncode:
            return {"error": result.stderr.strip() or f"probe exit {result.returncode}", "count": None, "available": None}
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        return {"error": f"{type(error).__name__}: {error}", "count": None, "available": None}


def _visible_gpu_indices(devices, mask):
    if mask is None:
        return [d["index"] for d in devices]
    if not mask.strip() or mask.strip() == "-1":
        return []
    result = []
    for token in (value.strip() for value in mask.split(",")):
        if token.startswith("MIG-"):
            return None
        matches = [d["index"] for d in devices if d["index"] == token or
                   (token.startswith("GPU-") and d["uuid"].startswith(token))]
        if len(matches) != 1 or matches[0] in result:
            break  # CUDA stops at invalid ordinals; never invent a mapping.
        result.append(matches[0])
    return result


def _gpu_report():
    rows, devices, meters = [], [], []
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = _run([executable, "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,driver_version,utilization.gpu", "--format=csv,noheader,nounits"])
            if result.returncode:
                rows.append(_row("NVIDIA inventory", result.stderr.strip() or f"nvidia-smi exit {result.returncode}", True))
            else:
                for fields in csv.reader(io.StringIO(result.stdout)):
                    if not fields:
                        continue
                    if len(fields) not in {7, 8}:
                        rows.append(_row("NVIDIA inventory row", "unrecognized nvidia-smi fields", True)); continue
                    index, uuid, name, total, used, free, driver = [v.strip() for v in fields[:7]]
                    try:
                        utilization = float(fields[7].strip()) if len(fields) == 8 else None
                        if utilization is not None and not 0 <= utilization <= 100:
                            utilization = None
                    except ValueError:
                        utilization = None
                    def memory(value):
                        try:
                            return int(float(value) * 1024 * 1024)
                        except ValueError:
                            return None
                    devices.append({"index": index, "uuid": uuid, "name": name, "total_bytes": memory(total),
                                    "used_bytes": memory(used), "available_bytes": memory(free), "driver": driver,
                                    "utilization_percent": utilization})
        except (OSError, subprocess.SubprocessError) as error:
            rows.append(_row("NVIDIA inventory", f"{type(error).__name__}: {error}", True))
    else:
        rows.append(_row("NVIDIA inventory", "nvidia-smi not found; names and VRAM unavailable", True))
    torch = _torch_gpu_probe()
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    mapped = _visible_gpu_indices(devices, mask)
    count = torch.get("count")
    observed = torch.get("devices") or []
    observed_ids = []
    for item in observed:
        identifier = str(item.get("uuid") or "").removeprefix("GPU-")
        matches = [d["index"] for d in devices if identifier and d["uuid"].removeprefix("GPU-") == identifier]
        if len(matches) == 1:
            observed_ids.append(matches[0])
    if observed and len(observed_ids) == count:
        mapped = observed_ids
    elif mask is not None and any(token.strip().isdigit() for token in mask.split(",")):
        # CUDA ordinals need not be identical to nvidia-smi's ordering.
        mapped = None
    rows += [_row("NVIDIA GPUs reported", len(devices) if devices else "none or unavailable", not devices),
             _row("CUDA-visible GPU count", count if count is not None else "unavailable", count is None),
             _row("CUDA runtime-reported available GPUs", count if torch.get("available") else 0 if torch.get("available") is False else "unavailable", not torch.get("available")),
             _row("CUDA runtime available", torch.get("available") if torch.get("available") is not None else "unavailable", not torch.get("available")),
             _row("CUDA_VISIBLE_DEVICES", repr(mask) if mask is not None else "not set"),
             _row("NVIDIA_VISIBLE_DEVICES", os.environ.get("NVIDIA_VISIBLE_DEVICES", "not set")),
             _row("PyTorch / CUDA build", str(torch.get("version", "unavailable")) + " / " + str(torch.get("cuda_build") or "CPU or unavailable"))]
    if torch.get("error"):
        rows.append(_row("PyTorch GPU probe", torch["error"], True))
    if mapped is None:
        rows.append(_row("Device mapping", "CUDA-to-NVIDIA UUID mapping unavailable; physical VRAM is not a verified visible/MIG allocation", True))
    elif count is not None and count != len(mapped):
        rows.append(_row("Device mapping", "CUDA count differs from ordinal/UUID mapping; individual usability unverified", True))
        mapped = None
    for d in devices:
        visible = d["index"] in mapped if mapped is not None else None
        detail = (f"{d['name']}; VRAM total {format_bytes(d['total_bytes'])}, used {format_bytes(d['used_bytes'])}, "
                  f"free {format_bytes(d['available_bytes'])}; driver {d['driver']}; UUID {d['uuid']}; "
                  f"CUDA-visible: {'yes' if visible else 'no' if visible is False else 'unverified'}")
        rows.append(_row("GPU " + d["index"], detail, d["total_bytes"] is None or visible is None))
        if visible:
            meters.append({"label": f"GPU {d['index']} · {d['name']}", "used_bytes": d["used_bytes"],
                           "total_bytes": d["total_bytes"], "available_bytes": d["available_bytes"],
                           "scope": "nvidia-smi instantaneous VRAM; shared, not reserved"})
    for item in observed:
        if item.get("error"):
            rows.append(_row(f"CUDA device {item['ordinal']}", item["error"], True))
        else:
            rows.append(_row(f"CUDA device {item['ordinal']}", f"{item.get('name', 'unknown')}; total VRAM "
                             f"{format_bytes(item.get('total_bytes'))}; UUID {item.get('uuid') or 'unavailable'}"))
    rows.append(_row("GPU probe scope", "driver/runtime metadata only; no kernels, models or memory-allocation test"))
    return {"title": "GPU capacity", "rows": rows}, meters, {"reported_devices": devices,
            "cuda": torch, "cuda_visible_indices": mapped, "cuda_visible_count": count}


def _existing_parent(path):
    path = Path(path).expanduser().absolute()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _disk_report(root):
    candidates = [("working directory", Path.cwd()), ("model root", Path(root)), ("project code", code_root()),
                  ("Python environment", Path(sys.prefix)), ("native Python", Path(sys.base_prefix)),
                  ("project cache", Path(root) / ".cache"),
                  ("user cache", Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))),
                  ("temporary files", Path(os.environ.get("TMPDIR", "/tmp")))]
    candidates += [(name, Path(os.environ[name])) for name in ("HF_HOME", "TORCH_HOME") if os.environ.get(name)]
    filesystems, rows, meters = {}, [], []
    for label, requested in candidates:
        try:
            actual = _existing_parent(requested)
            device = actual.stat().st_dev
            if device not in filesystems:
                vfs = os.statvfs(actual); block = vfs.f_frsize or vfs.f_bsize
                total, free, available = vfs.f_blocks * block, vfs.f_bfree * block, vfs.f_bavail * block
                filesystems[device] = {"device": device, "inspected_path": str(actual), "total_bytes": total,
                                      "free_bytes": free, "available_bytes": available,
                                      "used_bytes": max(0, total - free), "paths": []}
            filesystems[device]["paths"].append({"label": label, "path": str(requested), "existing_parent": str(actual)})
        except (OSError, ValueError) as error:
            rows.append(_row("Disk · " + label, f"{requested}: {type(error).__name__}: {error}", True))
    for index, record in enumerate(filesystems.values(), 1):
        locations = "; ".join(item["label"] + "=" + item["path"] +
                             (" (inspected " + item["existing_parent"] + ")" if item["path"] != item["existing_parent"] else "") for item in record["paths"])
        rows.append(_row(f"Filesystem {index}", f"total {format_bytes(record['total_bytes'])}; available to current user "
                         f"{format_bytes(record['available_bytes'])}; used {format_bytes(record['used_bytes'])}; {locations}"))
        meters.append({"label": "Disk · " + record["inspected_path"], "used_bytes": record["used_bytes"],
                       "total_bytes": record["total_bytes"], "available_bytes": record["available_bytes"],
                       "scope": "filesystem availability for current user; quota/reservation not guaranteed"})
    return {"title": "Disk space", "rows": rows}, meters, list(filesystems.values())


def _conda_versions():
    versions = {}
    for path in (Path(sys.base_prefix) / "conda-meta").glob("*.json"):
        try:
            record = json.loads(path.read_text())
            versions[str(record["name"])] = str(record["version"]) + ("=" + str(record["build"]) if record.get("build") else "")
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return versions


def _version(names, conda):
    for name in names:
        try:
            return importlib.metadata.version(name)
        except (importlib.metadata.PackageNotFoundError, OSError, ValueError):
            pass
    return next((conda[name] for name in names if name in conda), None)


def _discoverable(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _environment_report():
    conda = _conda_versions()
    rows = [_row("Python", sys.version.split()[0]), _row("Interpreter", sys.executable),
            _row("Python prefix", sys.prefix), _row("Native/base prefix", sys.base_prefix),
            _row("Native binaries", native_bin_dir()), _row("Code root", code_root()), _row("Static resources", resource_root())]
    if Path(sys.prefix).resolve() != Path(sys.base_prefix).resolve():
        settings = {key.strip().lower(): value.strip() for line in (_text(Path(sys.prefix) / "pyvenv.cfg") or "").splitlines()
                    if "=" in line for key, value in [line.split("=", 1)]}
        expected = (Path(sys.base_prefix) / "bin").resolve()
        home = Path(settings["home"]).resolve() if settings.get("home") else None
        inherited = settings.get("include-system-site-packages", "").lower() == "true"
        rows += [_row("Environment layout", "virtual environment with native/base Python"),
                 _row("Native/venv pairing", f"home={home}; expected={expected}; system-site-packages={inherited}", home != expected or not inherited)]
    else:
        rows.append(_row("Environment layout", "Conda/native Python" if (Path(sys.prefix) / "conda-meta").is_dir() else "Python environment (no nested virtual environment)"))
    rows.append(_row("Native CUDA profile", conda.get("cuda-version", "no Conda cuda-version record"), "cuda-version" not in conda))
    sections = [{"title": "Python and native environment", "rows": rows}]
    core = [("torch", ["torch"], "design, pipeline, score, validate"), ("lightning", ["lightning"], "design, pipeline"),
            ("hydra", ["hydra-core"], "configuration commands"), ("omegaconf", ["omegaconf"], "configuration commands"),
            ("yaml", ["PyYAML", "pyyaml"], "task YAML"), ("numpy", ["numpy"], "numerics"),
            ("biotite", ["biotite"], "structure I/O"), ("rdkit", ["rdkit"], "chemistry")]
    backends = [("jax", ["jax"], "AF2/AF3 validation"), ("openmm", ["OpenMM", "openmm"], "relax, design postprocess, postfilter"),
                ("openmmforcefields", ["openmmforcefields"], "OpenMM force fields"), ("openff.toolkit", ["openff-toolkit"], "ligand parameterization"),
                ("pdbfixer", ["pdbfixer"], "structure repair"), ("pyrosetta", ["pyrosetta"], "postfilter"),
                ("ray", ["ray"], "parallel design/validation"), ("gemmi", ["gemmi"], "AF3 structure inputs"),
                ("esm", ["esm"], "ESMFold2"), ("transformers", ["transformers"], "private model backends")]
    for title, items in [("Core package discovery", core), ("Backend package discovery", backends)]:
        lines = []
        for module, distributions, used_by in items:
            found = _discoverable(module)
            version = _version(distributions, conda)
            lines.append(_row(module, (f"discoverable; version {version or 'unavailable'}" if found else "not discoverable") + "; used by " + used_by, not found))
        lines.append(_row("Check scope", "package discovery and metadata only; imports, backend execution and GPU numerics are untested"))
        sections.append({"title": title, "rows": lines})
    sections.append({"title": "Bundled backend sources", "rows": [_row(name + " SDK", str(community_source(name)) +
                     (" (directory present)" if community_source(name).is_dir() else " (missing)"), not community_source(name).is_dir())
                     for name in ("esmfold2", "af3", "mol1", "pipetools")]})
    return sections


def _tool_report():
    rows, native = [], native_bin_dir()
    for name in ("antechamber", "parmchk2", "sqm", "tleap", "bondtype", "atomtype"):
        path = native / name; found = path.is_file() and os.access(path, os.X_OK)
        rows.append(_row(name, str(path) + (" (executable)" if found else " (missing/not executable)") +
                         "; used by ligand parameterization in relax/design postprocess", not found))
    for name, used_by in [("dssp", "postfilter secondary structure"), ("dalphaball", "postfilter Rosetta buried-unsatisfied metric")]:
        try:
            from src.postfilter.config import resolve_tool
            path = resolve_tool(name)
            rows.append(_row(name, (path or "not found") + "; used by " + used_by, path is None))
        except (ImportError, OSError, ValueError) as error:
            rows.append(_row(name, f"lookup unavailable: {error}; used by {used_by}", True))
    path = next((p for p in (Path(sys.executable).parent / "USalign", native / "USalign") if p.is_file() and os.access(p, os.X_OK)), None)
    if path is None and (found := shutil.which("USalign")):
        path = Path(found).resolve()
    rows.append(_row("USalign", (str(path) if path else "not found") + "; used by cluster", path is None))
    rows.append(_row("Tool scope", "resolved executable paths only; no tool execution or runtime-library test"))
    return {"title": "Native tool inventory", "rows": rows}


def _cpu_utilization(cpu):
    """A short affinity-scoped sample; absence never becomes a fabricated 0%."""
    try:
        import math
        import psutil
        values = psutil.cpu_percent(interval=0.1, percpu=True)
        indices = cpu.get("affinity_indices")
        indices = list(range(len(values))) if indices is None else indices
        selected = [float(values[index]) for index in indices if 0 <= index < len(values)]
        if not selected or any(not math.isfinite(value) or not 0 <= value <= 100 for value in selected):
            return None
        return sum(selected) / len(selected)
    except (ImportError, OSError, ValueError, TypeError, IndexError, AttributeError):
        return None


def _machine_overview(details):
    """Four compact categories, reusing the full diagnostic source records."""
    cpu = details.get("cpu", {})
    percent = _cpu_utilization(cpu)
    budget = cpu.get("effective_cores")
    capacity = (f"{budget:g} usable threads" if float(budget).is_integer() else f"{budget:g} CPU budget") if budget is not None else "thread budget unavailable"
    cpu_value = (f"{percent:.1f}%" if percent is not None else "usage unavailable") + " · " + capacity
    result = [{"label": "CPU", "value": cpu_value, "used": percent, "total": 100, "unit": "%"}]

    gpu = details.get("GPU capacity", {})
    cuda = gpu.get("cuda", {})
    visible = gpu.get("cuda_visible_indices")
    selected = [device for device in gpu.get("reported_devices", [])
                if visible is not None and device["index"] in visible
                and cuda.get("available") is not False and cuda.get("count") != 0]
    usage = None
    if selected:
        samples = [device.get("utilization_percent") for device in selected]
        if all(value is not None for value in samples):
            usage = sum(samples) / len(samples)
        labels = []
        for device in selected:
            current = device.get("utilization_percent")
            activity = f"{current:.0f}%" if current is not None else "usage unavailable"
            labels.append(device["index"] + ": " + device["name"] + " " + activity + " · VRAM " +
                          format_bytes(device.get("used_bytes")) + "/" + format_bytes(device.get("total_bytes")))
        gpu_value = (f"avg {usage:.0f}% · " if len(selected) > 1 and usage is not None else "") + "; ".join(labels)
    elif cuda.get("available") is False or cuda.get("count") == 0:
        gpu_value = "CUDA unavailable" + (f" · {cuda['count']} detected" if cuda.get("count") else " · 0 CUDA GPUs")
    elif cuda.get("devices"):
        gpu_value = "; ".join(str(item.get("name") or "CUDA GPU") + " · usage unavailable · VRAM total " +
                              format_bytes(item.get("total_bytes")) for item in cuda["devices"])
    else:
        gpu_value = "usage / GPU identity unavailable"
    result.append({"label": "GPU", "value": gpu_value, "used": usage, "total": 100, "unit": "%"})

    memory = details.get("Memory", {})
    total, free = memory.get("total_bytes"), memory.get("available_bytes")
    used = max(0, total - free) if total is not None and free is not None else None
    memory_value = format_bytes(used) + "/" + format_bytes(total) + " · " + format_bytes(free) + " free"
    result.append({"label": "RAM", "value": memory_value, "used": used, "total": total, "unit": "bytes"})

    disks = details.get("Disk space", [])
    disk = next((record for record in disks if any(path.get("label") == "model root" for path in record.get("paths", []))), None)
    disk_value = (format_bytes(disk.get("available_bytes")) + " free/" + format_bytes(disk.get("total_bytes"))
                  if disk is not None else "available space unavailable")
    result.append({"label": "Disk", "value": disk_value, "used": disk.get("used_bytes") if disk else None,
                   "total": disk.get("total_bytes") if disk else None, "unit": "bytes"})
    return result


def collect_system_report(root: Path) -> dict:
    """Best-effort report that never creates files, runs models or uses the network."""
    machine = []
    for label, probe in [("Host", socket.gethostname), ("OS", platform.platform),
                         ("Architecture", platform.machine), ("Working directory", Path.cwd),
                         ("Model root", lambda: Path(root).expanduser().absolute())]:
        try:
            machine.append(_row(label, probe()))
        except Exception as error:
            machine.append(_row(label, f"{type(error).__name__}: {error}", True))
    sections = [{"title": "Machine", "rows": machine}]
    meters, details = [], {}
    try:
        cgroup = _cgroup_info(PROC_ROOT, CGROUP_ROOT)
    except Exception as error:
        cgroup = {"cpu_quota_cores": None, "memory_limit_bytes": None, "memory_available_bytes": None,
                  "cpu_observed": False, "memory_observed": False, "version": "unavailable", "notes": [f"{type(error).__name__}: {error}"]}
    for title, callback in [("CPU capacity", lambda: _cpu_report(cgroup, PROC_ROOT)),
                            ("Memory", lambda: _memory_report(cgroup, PROC_ROOT)),
                            ("GPU capacity", _gpu_report), ("Disk space", lambda: _disk_report(root))]:
        try:
            result = callback()
            sections.append(result[0])
            if title == "CPU capacity":
                details["cpu"] = result[1]
            else:
                extra = result[1]
                meters.extend([extra] if isinstance(extra, dict) else extra)
                details[title] = result[2]
        except Exception as error:
            sections.append({"title": title, "rows": [_row("Probe", f"{type(error).__name__}: {error}", True)]})
    sections.append({"title": "Process resource limits", "rows": [_row("Controller", cgroup["version"]),
                     *[_row("Limit probe", note, True) for note in cgroup["notes"]]]})
    for title, callback in [("Python and native environment", _environment_report), ("Native tool inventory", _tool_report)]:
        try:
            result = callback()
            sections.extend(result if isinstance(result, list) else [result])
        except Exception as error:
            sections.append({"title": title, "rows": [_row("Probe", f"{type(error).__name__}: {error}", True)]})
    return {"sections": sections, "meters": meters, "details": details,
            "machine_overview": _machine_overview(details),
            "scope": "read-only metadata/package discovery/tool paths; no model execution"}
