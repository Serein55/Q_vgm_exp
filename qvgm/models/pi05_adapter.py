"""Phase-0 adapter using OpenPI directly and the external few-shot checkpoint."""

from pathlib import Path

import numpy as np
import torch


def load_sft_policy(cfg):
    import safetensors.torch
    from openpi import transforms
    from openpi.models import pi0_config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.policies.libero_policy import LiberoInputs, LiberoOutputs
    from openpi.policies.policy import Policy
    from openpi.shared.normalize import load
    from openpi.training.config import ModelTransformFactory

    device = cfg["runtime"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; attach a GPU or use --cpu-smoke for interface checks")
    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    mc = cfg["model"]
    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=mc["action_horizon"],
        action_dim=mc["action_dim"],
        discrete_state_input=mc["discrete_state_input"],
    )
    model = PI0Pytorch(model_config)
    weight_path = Path(cfg["paths"]["checkpoint"]) / "model.safetensors"
    # Validate all missing/unexpected keys, including tied weights via load_model.
    safetensors.torch.load_model(model, str(weight_path), strict=True)
    if device.startswith("cuda"):
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model.float()
    # Upstream compiles this method in __init__; keep initial validation eager.
    model.sample_actions = PI0Pytorch.sample_actions.__get__(model)
    model.requires_grad_(False)
    stats = load(Path(cfg["paths"]["norm_stats"]).parent)
    mt = ModelTransformFactory()(model_config)
    return Policy(
        model,
        transforms=[
            LiberoInputs(model_config.model_type),
            transforms.Normalize(stats, use_quantiles=True),
            *mt.inputs,
        ],
        output_transforms=[
            *mt.outputs,
            transforms.Unnormalize(stats, use_quantiles=True),
            LiberoOutputs(),
        ],
        sample_kwargs={"num_steps": mc["denoising_steps"]},
        is_pytorch=True,
        pytorch_device=device,
    )


def infer_actions(policy, observation, cfg, rng):
    m = cfg["model"]
    noise = rng.standard_normal((m["action_horizon"], m["action_dim"])).astype(np.float32)
    if hasattr(policy, "_qvgm_flow"):
        import time

        start = time.monotonic()
        flow = policy._qvgm_flow
        context = flow.encode_context([observation])
        normalized, _ = flow.sample_with_intermediates(context, torch.from_numpy(noise)[None])
        result = {
            "actions": flow.unnormalize(normalized, context)[0],
            "policy_timing": {"infer_ms": (time.monotonic() - start) * 1000},
        }
    else:
        result = policy.infer(observation, noise=noise)
    actions = result["actions"]
    if actions.shape != (m["action_horizon"], cfg["env"]["action_dim"]):
        raise ValueError(f"Unexpected action shape {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Policy returned non-finite actions")
    return result


class Pi05Flow:
    """Expose OpenPI in the paper's convention: tau=0 noise, tau=1 clean."""

    def __init__(self, policy, cfg):
        self.policy, self.cfg = policy, cfg
        self.model = policy._model
        self.device = cfg["runtime"]["device"]

    @torch.no_grad()
    def encode_context(self, observations):
        import jax
        from openpi.models.model import Observation
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        items = [self.policy._input_transform(dict(obs)) for obs in observations]
        inputs = jax.tree.map(lambda *xs: torch.as_tensor(np.stack(xs), device=self.device), *items)
        observation = Observation.from_dict(inputs)
        images, imasks, tokens, tmasks, state = self.model._preprocess_observation(
            observation, train=False
        )
        emb, pad, att = self.model.embed_prefix(images, imasks, tokens, tmasks)
        mask = self.model._prepare_attention_masks_4d(make_att_2d_masks(pad, att))
        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
            "eager"
        )
        (prefix, _), cache = self.model.paligemma_with_expert.forward(
            attention_mask=mask,
            position_ids=pad.cumsum(1) - 1,
            past_key_values=None,
            inputs_embeds=[emb, None],
            use_cache=True,
        )
        return {
            "state": state.detach(),
            "pad": pad.detach(),
            "cache": cache,
            "prefix": prefix.detach(),
        }

    def predict_velocity(self, x, tau, context):
        time = torch.as_tensor(1.0 - tau, device=x.device, dtype=torch.float32).expand(x.shape[0])
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=str(self.device).startswith("cuda") and getattr(self, "actor_training", False),
        ):
            return -self.model.denoise_step(
                context["state"], context["pad"], context["cache"], x, time
            )

    @torch.no_grad()
    def sample_with_intermediates(self, context, noise=None, temperature=1.0, sigma=0.0):
        m = self.cfg["model"]
        if noise is None:
            noise = torch.randn(
                context["state"].shape[0], m["action_horizon"], m["action_dim"], device=self.device
            )
        x = noise.to(self.device).float() * temperature
        trajectory = []
        for k in range(m["denoising_steps"]):
            tau = k / m["denoising_steps"]
            trajectory.append((tau, x.detach()))
            x = x + self.predict_velocity(x, tau, context) / m["denoising_steps"]
            if sigma:
                x = x + sigma / m["denoising_steps"] ** 0.5 * torch.randn_like(x)
        return x.detach(), trajectory

    def unnormalize(self, actions, context):
        results = []
        for i in range(actions.shape[0]):
            output = {
                "state": context["state"][i].float().cpu().numpy(),
                "actions": actions[i].detach().float().cpu().numpy(),
            }
            results.append(self.policy._output_transform(output)["actions"])
        return np.stack(results)

    def enable_actor_training(self, master_dtype="float32"):
        if master_dtype not in ("float32", "checkpoint"):
            raise ValueError("master_dtype must be float32 or checkpoint")
        self.model.requires_grad_(False)
        self.actor_training = True
        names = (
            "paligemma_with_expert.gemma_expert.",
            "action_in_proj.",
            "action_out_proj.",
            "time_mlp_in.",
            "time_mlp_out.",
        )
        for name, parameter in self.model.named_parameters():
            if name.startswith(names):
                if master_dtype == "float32":
                    parameter.data = parameter.data.float()
                parameter.requires_grad_(True)
        self.model.eval()  # Local gradients work in eval mode; no dropout drift.
        return [p for p in self.model.parameters() if p.requires_grad]
