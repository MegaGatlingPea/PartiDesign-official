"""Standalone PartiDesign sampler with per-modality polymer/ligand updates.

Polymer uses the validated score-corrected update, while ligand supports the
validated score-corrected (SC) and velocity-field (VF) updates in each
product-state component. The validated 400-step component-specific grids,
checkpoint-selected flex/rigid context semantics, and detached clean
self-conditioning are retained. Training losses, guidance variants, and
trajectory collection remain outside this module.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from src.design.model.constants import MOL_TYPE_TO_ID


STATE_COMPONENTS = ("token_center", "local_latents")
STATE_DIMS = {"token_center": 3, "local_latents": 8}
CONTEXT_CONSTRAINT_MODES = ("flex", "rigid")
TOKEN_MODALITIES = ("polymer", "ligand")
POLYMER_MOL_TYPE_IDS = tuple(
    MOL_TYPE_TO_ID[name] for name in ("protein", "rna", "dna")
)
MOTIF_MASK_KEYS = (
    "protein_motif_atom37_mask",
    "rna_motif_atom27_mask",
    "dna_motif_atom27_mask",
    "ligand_motif_mask",
)
_EXPECTED_STEPS = 400
_EPS = 1e-5


@dataclass(frozen=True)
class SimulationState:
    """Immutable cursor and tensors for an in-memory segmented simulation."""

    step_index: int
    x: Mapping[str, Tensor]
    previous_clean: Mapping[str, Tensor] | None


def validate_context_constraint_mode(value: Any) -> str:
    """Return one run-wide context mode or reject a mixed/unknown spelling."""

    if not isinstance(value, str) or value not in CONTEXT_CONSTRAINT_MODES:
        raise ValueError(
            "context_constraint_mode must be exactly 'flex' or 'rigid', got "
            f"{value!r}"
        )
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_exact_keys(
    value: Any,
    expected: set[str] | frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    mapping = _mapping(value, label)
    keys = {str(key) for key in mapping}
    if keys != set(expected) or len(mapping) != len(expected):
        raise ValueError(
            f"{label} keys must be exactly {sorted(expected)}, got {sorted(keys)}"
        )
    return mapping


def _require_exact_value(value: Any, expected: Any, label: str) -> None:
    """Require both the frozen scalar type and value.

    Python considers ``True == 1.0`` and ``False == 0``.  The public sampling
    contract must not inherit that coercion because those spellings denote
    different YAML types and can silently alter downstream numerics.
    """

    if type(value) is not type(expected) or value != expected:
        raise ValueError(
            f"unsupported {label}: expected {expected!r} "
            f"({type(expected).__name__})"
        )


def default_sampling_config() -> dict[str, Any]:
    """Return a fresh copy of the confirmed standalone sampling configuration."""

    common_step = {
        "sampling_mode": {"polymer": "sc", "ligand": "sc"},
        "sc_scale_noise": 0.1,
        "sc_scale_score": 1.0,
        "t_lim_ode": 0.98,
        "t_lim_ode_below": 0.02,
        "tsr_k": 1.0,
        "tsr_sigma": 1.0,
        "center_every_step": False,
    }
    return {
        "args": {"nsteps": _EXPECTED_STEPS, "self_cond": True},
        "model": {
            "token_center": {
                "schedule": {"mode": "log", "p": 2.0},
                "gt": {"mode": "1/t", "p": 1.0, "clamp_val": None},
                "simulation_step_params": {
                    **common_step,
                    "sampling_mode": {"polymer": "sc", "ligand": "sc"},
                },
            },
            "local_latents": {
                "schedule": {"mode": "power", "p": 2.0},
                "gt": {"mode": "tan", "p": 1.0, "clamp_val": None},
                "simulation_step_params": {
                    **common_step,
                    "sampling_mode": {"polymer": "sc", "ligand": "sc"},
                },
            },
        },
    }


def validate_sampling_config(config: Mapping[str, Any]) -> None:
    """Validate the closed numerical sampling contract.

    ``ligand_graph`` is accepted when the full public sampling group is passed, but it
    is not consumed by this numerical solver.
    """

    top = _mapping(config, "sampling")
    keys = set(top)
    if keys not in (
        {"args", "model"},
        {"args", "model", "ligand_graph"},
    ):
        raise ValueError(
            "sampling keys must be exactly ['args', 'model'] "
            "with optional 'ligand_graph'"
        )
    args = _require_exact_keys(top["args"], {"nsteps", "self_cond"}, "sampling.args")
    if type(args["nsteps"]) is not int or args["nsteps"] != _EXPECTED_STEPS:
        raise ValueError("sampling.args.nsteps must be exactly 400")
    if args["self_cond"] is not True:
        raise ValueError("sampling.args.self_cond must be true")
    if "ligand_graph" in top:
        graph = _require_exact_keys(
            top["ligand_graph"], {"method"}, "sampling.ligand_graph"
        )
        if type(graph["method"]) is not str or graph["method"] not in {
            "model",
            "obabel",
        }:
            raise ValueError("sampling.ligand_graph.method must be model or obabel")
    validate_sampling_model_args(_mapping(top["model"], "sampling.model"))


def validate_sampling_model_args(config: Mapping[str, Any]) -> None:
    components = _require_exact_keys(
        config,
        set(STATE_COMPONENTS),
        "sampling.model",
    )
    expected = default_sampling_config()["model"]
    for component in STATE_COMPONENTS:
        component_cfg = _require_exact_keys(
            components[component],
            {"schedule", "gt", "simulation_step_params"},
            f"sampling.model.{component}",
        )
        schedule = _require_exact_keys(
            component_cfg["schedule"],
            {"mode", "p"},
            f"sampling.model.{component}.schedule",
        )
        gt = _require_exact_keys(
            component_cfg["gt"],
            {"mode", "p", "clamp_val"},
            f"sampling.model.{component}.gt",
        )
        step = _require_exact_keys(
            component_cfg["simulation_step_params"],
            {
                "sampling_mode",
                "sc_scale_noise",
                "sc_scale_score",
                "t_lim_ode",
                "t_lim_ode_below",
                "tsr_k",
                "tsr_sigma",
                "center_every_step",
            },
            f"sampling.model.{component}.simulation_step_params",
        )
        modes = _require_exact_keys(
            step["sampling_mode"],
            set(TOKEN_MODALITIES),
            f"sampling.model.{component}.simulation_step_params.sampling_mode",
        )

        expected_component = expected[component]
        for key, expected_value in expected_component["schedule"].items():
            _require_exact_value(
                schedule[key], expected_value, f"{component} schedule.{key}"
            )
        for key, expected_value in expected_component["gt"].items():
            _require_exact_value(
                gt[key], expected_value, f"{component} g(t).{key}"
            )
        if type(modes["polymer"]) is not str or modes["polymer"] != "sc":
            raise ValueError(f"{component} requires polymer=sc sampling")
        if type(modes["ligand"]) is not str or modes["ligand"] not in {"sc", "vf"}:
            raise ValueError(f"{component} requires ligand=vf or ligand=sc sampling")
        for key in (
            "sc_scale_noise",
            "sc_scale_score",
            "t_lim_ode",
            "t_lim_ode_below",
            "tsr_k",
            "tsr_sigma",
            "center_every_step",
        ):
            _require_exact_value(
                step[key],
                expected_component["simulation_step_params"][key],
                f"{component} {key}",
            )


def get_schedule(mode: str, nsteps: int, *, p1: float | None = None) -> Tensor:
    """Return the validation-dump time grid byte-for-byte in float32."""

    if nsteps <= 0:
        raise ValueError("nsteps must be positive")
    base = torch.linspace(0.0, 1.0, nsteps + 1)
    if mode == "power":
        if p1 is None:
            raise ValueError("power schedule requires p")
        return base.pow(float(p1))
    if mode == "log":
        if p1 is None or float(p1) <= 0:
            raise ValueError("log schedule requires p > 0")
        schedule = 1.0 - torch.logspace(-float(p1), 0.0, nsteps + 1).flip(0)
        schedule = schedule - schedule.min()
        return schedule / schedule.max()
    raise ValueError(f"unsupported inference schedule {mode!r}")


def get_gt(
    t: Tensor,
    mode: str,
    param: float,
    clamp_val: float | None = None,
    eps: float = 1e-2,
) -> Tensor:
    """Return the validation-dump g(t) values."""

    def transform_gt(value: Tensor, power: float) -> Tensor:
        if power == 1.0:
            return value
        log_value = torch.log(value)
        mean_log_value = torch.mean(log_value)
        normalized = torch.sigmoid(log_value - mean_log_value).pow(power)
        return torch.exp(torch.logit(normalized, eps=1e-6) + mean_log_value)

    t = torch.clamp(t, 0.0, 1.0 - 1e-5)
    if mode == "tan":
        angle = (1.0 - t) * torch.pi / 2.0
        value = (torch.pi / 2.0) * torch.sin(angle) / (torch.cos(angle) + eps)
    elif mode == "1/t":
        value = 1.0 / (t + eps)
    else:
        raise ValueError(f"unsupported inference g(t) mode {mode!r}")
    value = transform_gt(value, param)
    return torch.clamp(value, min=0.0, max=clamp_val)


def _time_view(t: Tensor, x: Tensor) -> Tensor:
    expected = x.shape[:-2]
    if t.shape != expected:
        raise ValueError(
            f"time shape {tuple(t.shape)} must equal batch shape {tuple(expected)}"
        )
    return t.reshape(t.shape + (1, 1))


def _mean_w_mask(value: Tensor, mask: Tensor) -> Tensor:
    mask_f = mask.to(dtype=value.dtype).unsqueeze(-1)
    count = mask_f.sum(dim=-2, keepdim=True).clamp_min(1.0)
    return (value * mask_f).sum(dim=-2, keepdim=True) / count


def vf_to_score(x_t: Tensor, velocity: Tensor, t: Tensor) -> Tensor:
    if x_t.shape != velocity.shape:
        raise ValueError("x_t and velocity must have identical shapes")
    if not torch.all(t < 1.0):
        raise ValueError("vf_to_score requires t < 1")
    t_view = _time_view(t.to(x_t), x_t)
    return (t_view * velocity - x_t) / (1.0 - t_view)


def score_to_vf(x_t: Tensor, score: Tensor, t: Tensor) -> Tensor:
    if x_t.shape != score.shape:
        raise ValueError("x_t and score must have identical shapes")
    if not torch.all(t > 0.0):
        raise ValueError("score_to_vf requires t > 0")
    t_view = _time_view(t.to(x_t), x_t)
    return (x_t + (1.0 - t_view) * score) / t_view


def _vf_to_score_trusted(x_t: Tensor, velocity: Tensor, t: Tensor) -> Tensor:
    """The checked ``vf_to_score`` arithmetic for prevalidated solver inputs."""

    t_view = _time_view(t.to(x_t), x_t)
    return (t_view * velocity - x_t) / (1.0 - t_view)


def _score_to_vf_trusted(x_t: Tensor, score: Tensor, t: Tensor) -> Tensor:
    """The checked ``score_to_vf`` arithmetic for prevalidated solver inputs."""

    t_view = _time_view(t.to(x_t), x_t)
    return (x_t + (1.0 - t_view) * score) / t_view


class _RDNInferenceFlow:
    def __init__(self, dim: int) -> None:
        self.dim = int(dim)

    def _validate(self, value: Tensor, mask: Tensor) -> None:
        if value.shape != mask.shape + (self.dim,):
            raise ValueError(
                f"expected state shape {mask.shape + (self.dim,)}, got {value.shape}"
            )
        if mask.dtype is not torch.bool:
            raise TypeError("flow mask must be bool")

    def apply_mask(self, value: Tensor, mask: Tensor) -> Tensor:
        self._validate(value, mask)
        return value * mask[..., None]

    def sample_noise(self, mask: Tensor) -> Tensor:
        noise = torch.randn(mask.shape + (self.dim,), device=mask.device)
        return self.apply_mask(noise, mask)

    def clean_prediction(
        self,
        x_t: Tensor,
        t: Tensor,
        mask: Tensor,
        network_output: Mapping[str, Tensor],
    ) -> Tensor:
        if "v" not in network_output:
            raise KeyError("FM network output must contain velocity 'v'")
        t_view = _time_view(t.to(x_t), x_t)
        clean = x_t + (1.0 - t_view) * network_output["v"]
        return self.apply_mask(clean, mask)

    def step(
        self,
        *,
        x_t: Tensor,
        velocity: Tensor,
        t: Tensor,
        dt: Tensor,
        gt: Tensor,
        mode: str,
        mask: Tensor,
        params: Mapping[str, Any],
    ) -> Tensor:
        self._validate(x_t, mask)
        if t.shape != x_t.shape[:-2]:
            raise ValueError("time must match the sample batch shape")
        t_flat = t.flatten()
        if t_flat.numel() == 0 or not torch.all(t_flat == t_flat[0]):
            raise ValueError("sampling requires one common time across the batch")
        dt_t = torch.as_tensor(dt, dtype=x_t.dtype, device=x_t.device)
        gt_t = torch.as_tensor(gt, dtype=x_t.dtype, device=x_t.device)
        if dt_t.numel() != 1 or gt_t.numel() != 1:
            raise ValueError("dt and gt must be scalar")
        if dt_t.item() < 0 or gt_t.item() < 0:
            raise ValueError("dt and gt must be non-negative")

        velocity = self.apply_mask(velocity, mask)
        if mode == "vf":
            delta = velocity * dt_t
        elif mode == "sc":
            score = vf_to_score(x_t, velocity, t)
            if float(t_flat[0].item()) > float(params["t_lim_ode"]):
                delta = score_to_vf(x_t, score * 1.5, t) * dt_t
            else:
                noise = torch.randn_like(x_t)
                std = torch.sqrt(
                    2.0 * gt_t * float(params["sc_scale_noise"]) * dt_t
                )
                delta = (velocity + gt_t * score) * dt_t + std * noise
        else:
            raise ValueError(f"unsupported inference solver {mode!r}")
        result = self.apply_mask(x_t + delta, mask)
        if not torch.isfinite(result).all():
            raise FloatingPointError("non-finite inference simulation state")
        return result

    def _step_trusted(
        self,
        *,
        x_t: Tensor,
        velocity: Tensor,
        t: Tensor,
        dt: Tensor,
        gt: Tensor,
        mode: str,
        mask: Tensor,
        params: Mapping[str, Any],
        ode_threshold_exceeded: bool,
    ) -> tuple[Tensor, Tensor]:
        """Run one prevalidated step and return its state plus a finite flag.

        The public ``step`` method intentionally retains all direct-call checks.  The
        optimized solver calls this only after validating the invariant schedule,
        masks, and scalar step facts once per partial simulation.  Arithmetic and
        operation order mirror ``step`` exactly. The caller checks the finite
        result before advancing to the next component so failed trajectories
        consume exactly the same RNG draws as the original path.
        """

        self._validate(x_t, mask)
        dt_t = torch.as_tensor(dt, dtype=x_t.dtype, device=x_t.device)
        gt_t = torch.as_tensor(gt, dtype=x_t.dtype, device=x_t.device)
        velocity = self.apply_mask(velocity, mask)
        if mode == "vf":
            delta = velocity * dt_t
        elif mode == "sc":
            score = _vf_to_score_trusted(x_t, velocity, t)
            if ode_threshold_exceeded:
                delta = _score_to_vf_trusted(x_t, score * 1.5, t) * dt_t
            else:
                noise = torch.randn_like(x_t)
                std = torch.sqrt(
                    2.0 * gt_t * float(params["sc_scale_noise"]) * dt_t
                )
                delta = (velocity + gt_t * score) * dt_t + std * noise
        else:
            raise ValueError(f"unsupported inference solver {mode!r}")
        result = self.apply_mask(x_t + delta, mask)
        return result, torch.isfinite(result).all()


class ProductSpaceSolver:
    """Two-component sampler matching the training flex/rigid reference semantics."""

    def __init__(self, context_constraint_mode: str = "rigid") -> None:
        self._context_constraint_mode = validate_context_constraint_mode(
            context_constraint_mode
        )
        # The main inference dispatcher opts into the trusted host-sync-reduced path
        # explicitly.  Keep the baseline numerical route as the default.
        self.optimized = False
        self._flows = {
            component: _RDNInferenceFlow(STATE_DIMS[component])
            for component in STATE_COMPONENTS
        }

    @property
    def context_constraint_mode(self) -> str:
        return self._context_constraint_mode

    @property
    def rigid_context(self) -> bool:
        return self._context_constraint_mode == "rigid"

    @staticmethod
    def _validate_component_keys(value: Mapping[str, Any], label: str) -> None:
        keys = set(value)
        if keys != set(STATE_COMPONENTS) or len(value) != len(STATE_COMPONENTS):
            raise ValueError(f"{label} keys must be exactly {STATE_COMPONENTS}")

    def _validate_state(
        self,
        state: Mapping[str, Tensor],
        mask: Tensor,
        label: str,
    ) -> None:
        self._validate_component_keys(state, label)
        for component in STATE_COMPONENTS:
            value = state[component]
            expected = mask.shape + (STATE_DIMS[component],)
            if not isinstance(value, Tensor) or value.shape != expected:
                raise ValueError(f"{label}.{component} must have shape {expected}")
            if value.device != mask.device or not value.is_floating_point():
                raise TypeError(f"{label}.{component} has invalid device or dtype")

    @staticmethod
    def _token_mask(batch: Mapping[str, Any]) -> Tensor:
        mask = batch.get("token_mask", batch.get("mask"))
        if not isinstance(mask, Tensor) or mask.dtype is not torch.bool or mask.ndim != 2:
            raise TypeError("sampling mask must be a [B, N] bool tensor")
        if not bool(mask.any(dim=-1).all().item()):
            raise ValueError("every sampling item needs at least one valid token")
        return mask

    def _denoising_mask(
        self,
        batch: Mapping[str, Any],
        token_mask: Tensor,
    ) -> Tensor:
        if not self.rigid_context:
            return token_mask
        design_mask = batch.get("design_token_mask")
        if (
            not isinstance(design_mask, Tensor)
            or design_mask.dtype is not torch.bool
            or design_mask.shape != token_mask.shape
        ):
            raise TypeError("rigid flow requires design_token_mask as [B, N] bool")
        if bool((design_mask & ~token_mask).any().item()):
            raise ValueError("design_token_mask contains padded tokens")
        if not bool(design_mask.any(dim=-1).all().item()):
            raise ValueError("every rigid sample needs at least one design token")
        return design_mask

    def _fixed_context(
        self,
        batch: Mapping[str, Any],
        token_mask: Tensor,
        denoising_mask: Tensor,
    ) -> dict[str, Tensor] | None:
        if not self.rigid_context:
            return None
        value = _mapping(batch.get("x_context"), "x_context")
        fixed = {component: value[component] for component in STATE_COMPONENTS}
        self._validate_state(fixed, token_mask, "x_context")
        context_mask = token_mask & ~denoising_mask
        for component in STATE_COMPONENTS:
            if bool((fixed[component][~context_mask] != 0).any().item()):
                raise ValueError(
                    f"x_context.{component} must be zero outside context tokens"
                )
        return fixed

    def clamp_context(
        self,
        state: Mapping[str, Tensor],
        *,
        token_mask: Tensor,
        denoising_mask: Tensor,
        fixed_context: Mapping[str, Tensor] | None,
    ) -> dict[str, Tensor]:
        self._validate_state(state, token_mask, "state")
        if not self.rigid_context:
            return {
                component: state[component]
                * token_mask[..., None].to(state[component].dtype)
                for component in STATE_COMPONENTS
            }
        if fixed_context is None:
            raise ValueError("rigid sampling requires fixed_context")
        return {
            component: torch.where(
                denoising_mask[..., None],
                state[component],
                fixed_context[component],
            )
            * token_mask[..., None].to(state[component].dtype)
            for component in STATE_COMPONENTS
        }

    @staticmethod
    def _modality_masks(mol_type_id: Tensor, token_mask: Tensor) -> dict[str, Tensor]:
        if mol_type_id.dtype is not torch.long or mol_type_id.shape != token_mask.shape:
            raise TypeError("mol_type_id must be a [B, N] long tensor")
        polymer = torch.zeros_like(token_mask)
        for mol_type_id_value in POLYMER_MOL_TYPE_IDS:
            polymer |= mol_type_id == mol_type_id_value
        ligand = mol_type_id == MOL_TYPE_TO_ID["ligand"]
        recognized = polymer | ligand
        if bool((token_mask & ~recognized).any().item()):
            raise ValueError("valid tokens contain an unsupported modality")
        return {
            "polymer": polymer & token_mask,
            "ligand": ligand & token_mask,
        }

    def sample_schedule(
        self,
        sampling_model_args: Mapping[str, Any],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        validate_sampling_model_args(sampling_model_args)
        times: dict[str, Tensor] = {}
        gt_values: dict[str, Tensor] = {}
        for component in STATE_COMPONENTS:
            component_cfg = sampling_model_args[component]
            schedule_cfg = component_cfg["schedule"]
            times[component] = get_schedule(
                str(schedule_cfg["mode"]),
                _EXPECTED_STEPS,
                p1=float(schedule_cfg["p"]),
            )
            gt_cfg = component_cfg["gt"]
            gt_values[component] = get_gt(
                times[component][:-1],
                str(gt_cfg["mode"]),
                float(gt_cfg["p"]),
                gt_cfg["clamp_val"],
            )
        return times, gt_values

    def _clean_prediction(
        self,
        batch: Mapping[str, Any],
        network_output: Mapping[str, Mapping[str, Tensor]],
        token_mask: Tensor,
        denoising_mask: Tensor,
        fixed_context: Mapping[str, Tensor] | None,
    ) -> dict[str, Tensor]:
        self._validate_component_keys(network_output, "network output")
        clean = {
            component: self._flows[component].clean_prediction(
                batch["x_t"][component],
                batch["t"][component],
                token_mask,
                network_output[component],
            )
            for component in STATE_COMPONENTS
        }
        return self.clamp_context(
            clean,
            token_mask=token_mask,
            denoising_mask=denoising_mask,
            fixed_context=fixed_context,
        )

    def _simulation_step(
        self,
        *,
        state: Mapping[str, Tensor],
        network_output: Mapping[str, Mapping[str, Tensor]],
        times: Mapping[str, Tensor],
        dt: Mapping[str, Tensor],
        gt: Mapping[str, Tensor],
        token_mask: Tensor,
        denoising_mask: Tensor,
        mol_type_id: Tensor,
        fixed_context: Mapping[str, Tensor] | None,
        sampling_model_args: Mapping[str, Any],
    ) -> dict[str, Tensor]:
        modality_masks = self._modality_masks(mol_type_id, token_mask)
        updated: dict[str, Tensor] = {}
        for component in STATE_COMPONENTS:
            component_updated = torch.zeros_like(state[component])
            params = sampling_model_args[component]["simulation_step_params"]
            for modality in TOKEN_MODALITIES:
                active = denoising_mask & modality_masks[modality]
                if not bool(active.any().item()):
                    continue
                component_updated = component_updated + self._flows[component].step(
                    x_t=state[component],
                    velocity=network_output[component]["v"],
                    t=times[component],
                    dt=dt[component],
                    gt=gt[component],
                    mode=params["sampling_mode"][modality],
                    mask=active,
                    params=params,
                )
            updated[component] = component_updated
        return self.clamp_context(
            updated,
            token_mask=token_mask,
            denoising_mask=denoising_mask,
            fixed_context=fixed_context,
        )

    def _validate_optimized_schedule(
        self,
        *,
        times_cpu: Mapping[str, Tensor],
        gt_cpu: Mapping[str, Tensor],
        times: Mapping[str, Tensor],
        gt_values: Mapping[str, Tensor],
        device: torch.device,
    ) -> None:
        """Validate the CPU schedule without changing the caller's default dtype."""

        for component in STATE_COMPONENTS:
            cpu_time = times_cpu[component]
            cpu_gt = gt_cpu[component]
            device_time = times[component]
            device_gt = gt_values[component]
            if (
                not isinstance(cpu_time, Tensor)
                or cpu_time.device.type != "cpu"
                or not cpu_time.is_floating_point()
                or cpu_time.shape != (_EXPECTED_STEPS + 1,)
            ):
                raise TypeError(
                    f"optimized {component} time schedule must be floating CPU [401]"
                )
            if (
                not isinstance(cpu_gt, Tensor)
                or cpu_gt.device.type != "cpu"
                or not cpu_gt.is_floating_point()
                or cpu_gt.shape != (_EXPECTED_STEPS,)
            ):
                raise TypeError(
                    f"optimized {component} g(t) schedule must be floating CPU [400]"
                )
            if device_time.device != device or device_gt.device != device:
                raise TypeError(f"optimized {component} schedule device mismatch")
            if (
                device_time.dtype != cpu_time.dtype
                or device_gt.dtype != cpu_gt.dtype
            ):
                raise TypeError(f"optimized {component} schedule dtype mismatch")
            # ``torch.equal`` is deliberately done once per schedule, before the
            # trusted loop, so a device cast cannot silently change solver times.
            if not torch.equal(cpu_time.to(device), device_time):
                raise ValueError(f"optimized {component} time schedule cast mismatch")
            if not torch.equal(cpu_gt.to(device), device_gt):
                raise ValueError(f"optimized {component} g(t) schedule cast mismatch")

    def _simulation_step_optimized(
        self,
        *,
        state: Mapping[str, Tensor],
        network_output: Mapping[str, Mapping[str, Tensor]],
        times: Mapping[str, Tensor],
        dt: Mapping[str, Tensor],
        gt: Mapping[str, Tensor],
        token_mask: Tensor,
        denoising_mask: Tensor,
        fixed_context: Mapping[str, Tensor] | None,
        active_masks: Mapping[str, Tensor],
        active_any: Mapping[str, bool],
        ode_threshold_exceeded: Mapping[str, bool],
        sampling_model_args: Mapping[str, Any],
    ) -> dict[str, Tensor]:
        """Run a step with invariant masks/times already checked by the caller."""

        updated: dict[str, Tensor] = {}
        for component in STATE_COMPONENTS:
            component_updated = torch.zeros_like(state[component])
            params = sampling_model_args[component]["simulation_step_params"]
            for modality in TOKEN_MODALITIES:
                if not active_any[modality]:
                    continue
                result, result_finite = self._flows[component]._step_trusted(
                    x_t=state[component],
                    velocity=network_output[component]["v"],
                    t=times[component],
                    dt=dt[component],
                    gt=gt[component],
                    mode=params["sampling_mode"][modality],
                    mask=active_masks[modality],
                    params=params,
                    ode_threshold_exceeded=ode_threshold_exceeded[component],
                )
                if not bool(result_finite.item()):
                    raise FloatingPointError("non-finite inference simulation state")
                component_updated = component_updated + result
            updated[component] = component_updated
        return updated

    @torch.no_grad()
    def _partial_simulation_optimized(
        self,
        *,
        batch: Mapping[str, Any],
        predict_for_sampling: Callable[
            [Mapping[str, Any]], Mapping[str, Mapping[str, Tensor]]
        ],
        sampling_config: Mapping[str, Any],
        state: SimulationState,
        start_step: int,
        end_step: int,
        token_mask: Tensor,
        denoising_mask: Tensor,
        fixed_context: Mapping[str, Tensor] | None,
    ) -> SimulationState:
        """Advance the explicitly enabled host-sync-reduced solver path."""

        model_args = sampling_config["model"]
        times_cpu, gt_cpu = self.sample_schedule(model_args)
        times = {
            component: times_cpu[component].to(token_mask.device)
            for component in STATE_COMPONENTS
        }
        gt_values = {
            component: gt_cpu[component].to(token_mask.device)
            for component in STATE_COMPONENTS
        }
        self._validate_optimized_schedule(
            times_cpu=times_cpu,
            gt_cpu=gt_cpu,
            times=times,
            gt_values=gt_values,
            device=token_mask.device,
        )

        modality_masks = self._modality_masks(batch["mol_type_id"], token_mask)
        active_masks = {
            modality: denoising_mask & modality_masks[modality]
            for modality in TOKEN_MODALITIES
        }
        active_any = {
            modality: bool(active_masks[modality].any().item())
            for modality in TOKEN_MODALITIES
        }
        threshold = {
            component: float(
                model_args[component]["simulation_step_params"]["t_lim_ode"]
            )
            for component in STATE_COMPONENTS
        }
        ode_thresholds = {
            component: tuple(
                bool(float(times_cpu[component][step_index]) > threshold[component])
                for step_index in range(start_step, end_step)
            )
            for component in STATE_COMPONENTS
        }
        dt_values = {
            component: (
                times[component][start_step + 1 : end_step + 1]
                - times[component][start_step:end_step]
            )
            for component in STATE_COMPONENTS
        }

        current = dict(state.x)
        simulation_batch = dict(batch)
        simulation_batch["mask"] = token_mask
        simulation_batch["denoising_mask"] = denoising_mask
        if not self.rigid_context:
            simulation_batch.pop("x_context", None)
        simulation_batch.pop("x_sc", None)
        previous_clean = (
            None if state.previous_clean is None else dict(state.previous_clean)
        )
        batch_size = token_mask.shape[0]
        ones = torch.ones(batch_size, device=token_mask.device)

        for offset, step_index in enumerate(range(start_step, end_step)):
            step_time = {
                component: times[component][step_index] * ones
                for component in STATE_COMPONENTS
            }
            dt = {
                component: dt_values[component][offset]
                for component in STATE_COMPONENTS
            }
            gt_step = {
                component: gt_values[component][step_index]
                for component in STATE_COMPONENTS
            }
            simulation_batch["x_t"] = current
            simulation_batch["t"] = step_time
            if previous_clean is not None:
                simulation_batch["x_sc"] = self.clamp_context(
                    {
                        component: previous_clean[component].detach()
                        for component in STATE_COMPONENTS
                    },
                    token_mask=token_mask,
                    denoising_mask=denoising_mask,
                    fixed_context=fixed_context,
                )
            network_output = predict_for_sampling(simulation_batch)
            previous_clean = self._clean_prediction(
                simulation_batch,
                network_output,
                token_mask,
                denoising_mask,
                fixed_context,
            )
            updated = self._simulation_step_optimized(
                state=current,
                network_output=network_output,
                times=step_time,
                dt=dt,
                gt=gt_step,
                token_mask=token_mask,
                denoising_mask=denoising_mask,
                fixed_context=fixed_context,
                active_masks=active_masks,
                active_any=active_any,
                ode_threshold_exceeded={
                    component: ode_thresholds[component][offset]
                    for component in STATE_COMPONENTS
                },
                sampling_model_args=model_args,
            )
            current = self.clamp_context(
                updated,
                token_mask=token_mask,
                denoising_mask=denoising_mask,
                fixed_context=fixed_context,
            )

        return SimulationState(
            step_index=end_step,
            x=current,
            previous_clean=previous_clean,
        )

    @torch.no_grad()
    def initialize_simulation(
        self,
        *,
        batch: Mapping[str, Any],
        sampling_config: Mapping[str, Any],
    ) -> SimulationState:
        """Initialize the fixed solver at step zero without retaining a trajectory."""

        validate_sampling_config(sampling_config)
        token_mask = self._token_mask(batch)
        denoising_mask = self._denoising_mask(batch, token_mask)
        fixed_context = self._fixed_context(batch, token_mask, denoising_mask)
        state = {
            component: self._flows[component].sample_noise(denoising_mask)
            for component in STATE_COMPONENTS
        }
        state = self.clamp_context(
            state,
            token_mask=token_mask,
            denoising_mask=denoising_mask,
            fixed_context=fixed_context,
        )
        return SimulationState(step_index=0, x=state, previous_clean=None)

    @torch.no_grad()
    def partial_simulation(
        self,
        *,
        batch: Mapping[str, Any],
        predict_for_sampling: Callable[
            [Mapping[str, Any]], Mapping[str, Mapping[str, Tensor]]
        ],
        sampling_config: Mapping[str, Any],
        state: SimulationState,
        start_step: int,
        end_step: int,
    ) -> SimulationState:
        """Advance ``state`` over the half-open interval ``[start_step, end_step)``."""

        validate_sampling_config(sampling_config)
        if type(start_step) is not int or type(end_step) is not int:
            raise TypeError("simulation step bounds must be integers")
        if not 0 <= start_step <= end_step <= _EXPECTED_STEPS:
            raise ValueError("simulation steps must satisfy 0 <= start <= end <= 400")
        if not isinstance(state, SimulationState):
            raise TypeError("state must be a SimulationState")
        if type(state.step_index) is not int or state.step_index != start_step:
            raise ValueError("state.step_index must equal start_step")

        token_mask = self._token_mask(batch)
        denoising_mask = self._denoising_mask(batch, token_mask)
        fixed_context = self._fixed_context(batch, token_mask, denoising_mask)
        self._validate_state(state.x, token_mask, "state.x")
        if state.previous_clean is None:
            if start_step != 0:
                raise ValueError("a simulation after step zero requires previous_clean")
        else:
            if start_step == 0:
                raise ValueError("a step-zero simulation cannot have previous_clean")
            self._validate_state(
                state.previous_clean,
                token_mask,
                "state.previous_clean",
            )

        if self.optimized:
            return self._partial_simulation_optimized(
                batch=batch,
                predict_for_sampling=predict_for_sampling,
                sampling_config=sampling_config,
                state=state,
                start_step=start_step,
                end_step=end_step,
                token_mask=token_mask,
                denoising_mask=denoising_mask,
                fixed_context=fixed_context,
            )

        model_args = sampling_config["model"]
        times_cpu, gt_cpu = self.sample_schedule(model_args)
        times = {
            component: times_cpu[component].to(token_mask.device)
            for component in STATE_COMPONENTS
        }
        gt_values = {
            component: gt_cpu[component].to(token_mask.device)
            for component in STATE_COMPONENTS
        }
        current = dict(state.x)

        simulation_batch = dict(batch)
        simulation_batch["mask"] = token_mask
        simulation_batch["denoising_mask"] = denoising_mask
        if not self.rigid_context:
            simulation_batch.pop("x_context", None)
        simulation_batch.pop("x_sc", None)
        previous_clean = (
            None if state.previous_clean is None else dict(state.previous_clean)
        )
        batch_size = token_mask.shape[0]

        for step_index in range(start_step, end_step):
            step_time = {
                component: times[component][step_index]
                * torch.ones(batch_size, device=token_mask.device)
                for component in STATE_COMPONENTS
            }
            dt = {
                component: times[component][step_index + 1]
                - times[component][step_index]
                for component in STATE_COMPONENTS
            }
            gt_step = {
                component: gt_values[component][step_index]
                for component in STATE_COMPONENTS
            }
            simulation_batch["x_t"] = current
            simulation_batch["t"] = step_time
            if previous_clean is not None:
                simulation_batch["x_sc"] = self.clamp_context(
                    {
                        component: previous_clean[component].detach()
                        for component in STATE_COMPONENTS
                    },
                    token_mask=token_mask,
                    denoising_mask=denoising_mask,
                    fixed_context=fixed_context,
                )
            network_output = predict_for_sampling(simulation_batch)
            previous_clean = self._clean_prediction(
                simulation_batch,
                network_output,
                token_mask,
                denoising_mask,
                fixed_context,
            )
            current = self._simulation_step(
                state=current,
                network_output=network_output,
                times=step_time,
                dt=dt,
                gt=gt_step,
                token_mask=token_mask,
                denoising_mask=denoising_mask,
                mol_type_id=batch["mol_type_id"],
                fixed_context=fixed_context,
                sampling_model_args=model_args,
            )

        return SimulationState(
            step_index=end_step,
            x=current,
            previous_clean=previous_clean,
        )

    @torch.no_grad()
    def full_simulation(
        self,
        *,
        batch: Mapping[str, Any],
        predict_for_sampling: Callable[
            [Mapping[str, Any]], Mapping[str, Mapping[str, Tensor]]
        ],
        sampling_config: Mapping[str, Any],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Run the fixed 400-step sampling loop without trajectory retention."""

        initial = self.initialize_simulation(
            batch=batch,
            sampling_config=sampling_config,
        )
        final = self.partial_simulation(
            batch=batch,
            predict_for_sampling=predict_for_sampling,
            sampling_config=sampling_config,
            state=initial,
            start_step=0,
            end_step=_EXPECTED_STEPS,
        )
        token_mask = self._token_mask(batch)
        denoising_mask = self._denoising_mask(batch, token_mask)
        return dict(final.x), {
            "mask": token_mask,
            "denoising_mask": denoising_mask,
        }
