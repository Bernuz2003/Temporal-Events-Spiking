"""Fit stage of the preregistered future-sensory screen; never reads validation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from etsr.config import save_config
from etsr.data.common import DatasetSubset, build_loader
from etsr.data.factory import build_dataset_bundle
from etsr.evaluation.future_sensory import (
    FutureDecoder,
    SensoryDataset,
    build_causal_d,
    sensory_loss_parts,
)
from etsr.profiling.hardware import profile_model
from etsr.reproducibility import git_commit, seed_everything
from etsr.utils.io import ensure_dir, sha256_file, write_json

SPLIT_SEED = 20261001
FIRST_BUDGET = 40
EXTENDED_BUDGET = 80


def screen_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    if result["dataset"]["name"] != "dvslip":
        raise ValueError("The future sensory screen is DVS-Lip only")
    if result["representation"]["name"] != "count_frames_e0":
        raise ValueError("The screen requires the E0 representation")
    if result["representation"]["bin_width_us"] != 50_000 or result["representation"]["window_us"] != 2_000_000:
        raise ValueError("Screen physical clocks are fixed at 40 x 50 ms")
    result["augmentation"] = {"horizontal_flip_probability": 0.0}
    return result


def stratified_split(targets: tuple[int, ...] | list[int], sample_ids: list[str]) -> dict:
    """One class-stratified 80/20 split, independent of model and target seed."""
    if len(targets) != len(sample_ids):
        raise ValueError("Missing sample identities for train-only split")
    rng = np.random.default_rng(SPLIT_SEED)
    fit: list[int] = []
    holdout: list[int] = []
    for label in sorted(set(map(int, targets))):
        indices = np.flatnonzero(np.asarray(targets) == label)
        rng.shuffle(indices)
        cut = int(round(0.8 * len(indices)))
        fit.extend(int(i) for i in indices[:cut])
        holdout.extend(int(i) for i in indices[cut:])
    fit.sort()
    holdout.sort()
    if len(set(fit) & set(holdout)) or len(fit) + len(holdout) != len(targets):
        raise AssertionError("Invalid train-only split")
    identity = "\n".join(f"{index}\t{sample_ids[index]}\t{int(targets[index])}" for index in range(len(targets)))
    return {
        "seed": SPLIT_SEED,
        "fit_indices": fit,
        "holdout_indices": holdout,
        "identities_sha256": hashlib.sha256(identity.encode()).hexdigest(),
        "fit_id_sha256": hashlib.sha256("\n".join(sample_ids[i] for i in fit).encode()).hexdigest(),
        "holdout_id_sha256": hashlib.sha256("\n".join(sample_ids[i] for i in holdout).encode()).hexdigest(),
        "speaker_disjoint": False,
        "official_test_used": False,
    }


def _loader(dataset: SensoryDataset, config: dict, *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(config["dataset"].get("batch_size", 16)),
        shuffle=shuffle,
        generator=generator,
        num_workers=int(config["dataset"].get("num_workers", 0)),
        pin_memory=bool(config["dataset"].get("pin_memory", False)),
    )


def _router_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for name, parameter in model.named_parameters() if ".content_router." in name]


def _screen_parameters(model: nn.Module, decoder: FutureDecoder) -> list[nn.Parameter]:
    # The supervised classifier exists for topology compatibility but has no
    # role in this pretraining objective and must not inflate its parameter cost.
    return [
        *(parameter for name, parameter in model.named_parameters() if not name.startswith("head.")),
        *decoder.parameters(),
    ]


def train_causality_preflight(
    model: nn.Module, decoder: FutureDecoder, device: torch.device, *, amp: bool = False
) -> dict:
    """Check representation, output and gradient future-independence in train and eval."""
    torch_state = torch.get_rng_state()
    results = {}
    for training in (True, False):
        model.train(training)
        decoder.train(training)
        source = torch.rand(2, 6, 2, 128, 128, device=device)
        future = source.clone()
        future[:, 4:] = torch.rand_like(future[:, 4:]) * 9

        def run(frames: torch.Tensor):
            with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                stage = model._encode(frames)
                count, q = decoder(stage)
            return stage[:4], count[:, :4], q[:, :4]

        reference = run(source)
        changed = run(future)
        prefix = run(source[:, :4])
        value_tolerance = 2e-3 if amp else 2e-5
        for left, right in (*zip(reference, changed, strict=True), *zip(reference, prefix, strict=True)):
            torch.testing.assert_close(left, right, atol=value_tolerance, rtol=value_tolerance)
        a = source.detach().requires_grad_()
        b = future.detach().requires_grad_()
        # Scale before the AMP backward so tiny surrogate gradients do not
        # underflow and make the causality check vacuous.
        loss_a = run(a)[1].float().sum() * (1024 if amp else 1)
        loss_b = run(b)[1].float().sum() * (1024 if amp else 1)
        grad_a = torch.autograd.grad(loss_a, a)[0]
        grad_b = torch.autograd.grad(loss_b, b)[0]
        prefix_gradient_norm = float(grad_a[:, :4].norm().item())
        if prefix_gradient_norm <= 1e-8:
            raise RuntimeError("Causality test is vacuous: prefix prediction has no input gradient")
        gradient_tolerance = 3e-3 if amp else 3e-5
        torch.testing.assert_close(
            grad_a[:, :4], grad_b[:, :4], atol=gradient_tolerance, rtol=gradient_tolerance
        )
        torch.testing.assert_close(grad_a[:, 4:], torch.zeros_like(grad_a[:, 4:]), atol=0, rtol=0)
        torch.testing.assert_close(grad_b[:, 4:], torch.zeros_like(grad_b[:, 4:]), atol=0, rtol=0)
        results["train" if training else "eval"] = {
            "representation_max_abs": float((reference[0] - changed[0]).abs().max().item()),
            "prediction_max_abs": float((reference[1] - changed[1]).abs().max().item()),
            "gradient_max_abs": float((grad_a[:, :4] - grad_b[:, :4]).abs().max().item()),
            "prefix_gradient_norm": prefix_gradient_norm,
        }
    torch.set_rng_state(torch_state)
    return results


@torch.no_grad()
def fit_prior(loader: DataLoader, components: int, device: torch.device) -> dict[str, torch.Tensor | float]:
    total = torch.zeros(40, 2, components, 16, 16, dtype=torch.float64)
    q_sum = total.clone()
    cutoff_support = torch.zeros(40, 1, 1, 1, 1, dtype=torch.float64)
    cell_support = torch.zeros(40, 2, 1, 16, 16, dtype=torch.float64)
    occupied = 0
    all_cells = 0
    for batch_index, (_, field, valid, _, _, _, _) in enumerate(loader):
        field = field.double()
        count = field.sum(dim=3)
        mask = valid[:, :, None, None, None]
        total += (field * mask.unsqueeze(3)).sum(dim=0)
        q_sum += ((field / count.unsqueeze(3).clamp_min(1e-9)) * (count > 0).unsqueeze(3) * mask.unsqueeze(3)).sum(dim=0)
        cutoff_support += valid.double().sum(dim=0)[:, None, None, None, None]
        cell_support += (((count > 0) & mask).double()).sum(dim=0).unsqueeze(2)
        occupied += int(((count > 0) & mask).sum().item())
        all_cells += int(mask.sum().item()) * 2 * 16 * 16
        if (batch_index + 1) % 100 == 0:
            print(f"prior statistics: {batch_index + 1}/{len(loader)} batches", flush=True)
    mean_count = (total.sum(dim=2, keepdim=True) / cutoff_support.clamp_min(1)).float().squeeze(2)
    # A fixed alpha=1 pseudo-observation per component and cell, not per event.
    alpha = 1.0
    q_prior = ((q_sum + alpha) / (cell_support + components * alpha)).float()
    if int(cutoff_support[0]) == 0:
        raise ValueError("No valid onset cutoffs in fit")
    n_total = 0.0
    q_total = 0.0
    sample_total = 0
    for batch_index, (_, field, valid, _, _, _, _) in enumerate(loader):
        field = field.to(device)
        valid = valid.to(device)
        count_hat = mean_count.to(device)[None].expand(field.shape[0], -1, -1, -1, -1).clamp_min(1e-6)
        q_hat = q_prior.to(device)[None].expand(field.shape[0], -1, -1, -1, -1, -1)
        parts = sensory_loss_parts(count_hat, q_hat, field, valid)
        n_total += float(parts.count.sum().item())
        q_total += float(parts.timing.sum().item())
        sample_total += field.shape[0]
        if (batch_index + 1) % 100 == 0:
            print(f"prior scales: {batch_index + 1}/{len(loader)} batches", flush=True)
    scale_n = n_total / sample_total
    scale_q = q_total / sample_total
    if scale_n < 1e-6 or scale_q < 1e-6:
        raise ValueError("Mean-field loss scale is degenerate")
    return {
        "mean_count": mean_count,
        "q_prior": q_prior,
        "scale_n": scale_n,
        "scale_q": scale_q,
        "occupied_fraction": occupied / max(all_cells, 1),
        "valid_cutoffs_per_utterance": float(cutoff_support.sum().item()) / sample_total,
    }


def _batch_loss(model, decoder, batch, prior, device):
    frames, field, valid, _, _, _, _ = batch
    frames = frames.to(device, non_blocking=True)
    field = field.to(device, non_blocking=True)
    valid = valid.to(device, non_blocking=True)
    stage2 = model._encode(frames)
    count, q = decoder(stage2)
    parts = sensory_loss_parts(count.float(), q.float(), field.float(), valid)
    loss = 0.5 * parts.count.mean() / prior["scale_n"] + 0.5 * parts.timing.mean() / prior["scale_q"]
    return loss, parts


def _bounded_overfit(model, decoder, dataset, prior, config, device):
    # Isolated copy: the main run starts again from the original initial state.
    indices = list(range(min(8, len(dataset))))
    mini = SensoryDataset(dataset.encoded, [dataset.indices[i] for i in indices], dataset.mode, dataset.past)
    batch = next(iter(_loader(mini, config, shuffle=False, seed=0)))
    model.train()
    decoder.train()
    optimizer = torch.optim.AdamW(_screen_parameters(model, decoder), lr=1e-3)
    amp = bool(config["training"].get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    router = _router_parameters(model)
    router_initial = [parameter.detach().clone() for parameter in router]
    router_gradient = 0.0
    with torch.no_grad():
        initial = float(_batch_loss(model, decoder, batch, prior, device)[0].item())
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss, _ = _batch_loss(model, decoder, batch, prior, device)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite bounded sensory loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        router_gradient += sum(
            float(parameter.grad.detach().norm().item())
            for parameter in router if parameter.grad is not None
        )
        scaler.step(optimizer)
        scaler.update()
    with torch.no_grad():
        final = float(_batch_loss(model, decoder, batch, prior, device)[0].item())
    if not np.isfinite(final) or final >= initial * 0.9:
        raise RuntimeError(f"Bounded sensory overfit did not improve 10%: {initial:.4g} -> {final:.4g}")
    router_delta = sum(
        float((parameter.detach() - original).norm().item())
        for parameter, original in zip(router, router_initial, strict=True)
    )
    # TCAP tap weights start at zero, so a zero router gradient on step one is
    # expected; the bounded fit must verify that it becomes reachable later.
    if not np.isfinite(router_gradient) or not np.isfinite(router_delta) or router_gradient <= 0 or router_delta <= 0:
        raise RuntimeError("Bounded sensory fit did not train the stage2 content router")
    return {
        "initial": initial, "final": final, "relative_drop": 1 - final / initial,
        "router_gradient_norm_sum": router_gradient,
        "router_parameter_delta_norm": router_delta,
    }


def fit_future_sensory(
    config: dict,
    output_dir: str | Path,
    *,
    mode: str,
    seed: int = 42,
    past: bool = False,
    resume: bool = False,
) -> dict:
    """Complete 40-epoch fit or continuation to 80; no holdout samples are read."""
    from etsr.evaluation.future_sensory import MODES

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    config = screen_config(config)
    seed_everything(seed, deterministic=bool(config["experiment"].get("deterministic", True)))
    output = ensure_dir(output_dir)
    resolved_config_path = output / "config_resolved.yaml"
    if not resume:
        if any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing fit: {output}")
        save_config(config, resolved_config_path)
    elif not resolved_config_path.exists():
        raise ValueError("Resume requires the original resolved configuration")
    elif yaml.safe_load(resolved_config_path.read_text()) != {
        key: value for key, value in config.items() if not key.startswith("_")
    }:
        raise ValueError("Resume configuration differs from the original fit")
    bundle = build_dataset_bundle(config)
    split = stratified_split(bundle.train.targets, bundle.train.sample_ids)
    split_path = output / "train_only_split.json"
    if resume:
        if not split_path.exists() or json.loads(split_path.read_text()) != split:
            raise ValueError("Resume split differs from the saved train-only split")
    else:
        write_json(split, split_path)
    dataset = SensoryDataset(bundle.train, split["fit_indices"], mode, past)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_causal_d(config, len(bundle.classes)).to(device)
    components = 3 if mode == "fepf2" else 4
    decoder = FutureDecoder(components).to(device)
    loader = _loader(dataset, config, shuffle=True, seed=seed)
    if resume:
        state = torch.load(output / "last_state.pt", map_location="cpu", weights_only=False)
        if state["mode"] != mode or state["seed"] != seed or state["past"] != past:
            raise ValueError("Resume mode/seed does not match saved state")
        model.load_state_dict(state["model"], strict=True)
        decoder.load_state_dict(state["decoder"], strict=True)
        prior = torch.load(output / "fit_prior.pt", map_location="cpu", weights_only=False)
        preflight = json.loads((output / "preflight.json").read_text())
        start = state["epoch"] + 1
        if start != FIRST_BUDGET + 1:
            raise ValueError("Only the preregistered 40-to-80 extension may resume")
    else:
        print(json.dumps({"stage": "causality_preflight", "mode": mode, "seed": seed}), flush=True)
        causality = train_causality_preflight(
            model, decoder, device, amp=bool(config["training"].get("amp", False))
        )
        print(json.dumps({"stage": "fit_only_prior", "mode": mode, "seed": seed}), flush=True)
        prior = fit_prior(_loader(dataset, config, shuffle=False, seed=seed), components, device)
        torch.save(prior, output / "fit_prior.pt")
        init_model = copy.deepcopy(model.state_dict())
        init_decoder = copy.deepcopy(decoder.state_dict())
        print(json.dumps({"stage": "bounded_overfit", "mode": mode, "seed": seed}), flush=True)
        bounded = _bounded_overfit(model, decoder, dataset, prior, config, device)
        model.load_state_dict(init_model, strict=True)
        decoder.load_state_dict(init_decoder, strict=True)
        preflight = {"causality": causality, "bounded_overfit": bounded, "passed": True}
        write_json(preflight, output / "preflight.json")
        start = 1
    screen_parameters = _screen_parameters(model, decoder)
    optimizer = torch.optim.AdamW(
        screen_parameters, lr=3e-4, weight_decay=5e-4
    )
    amp = bool(config["training"].get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    if resume:
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        loader.generator.set_state(state["loader_rng"])
    rows = json.loads((output / "fit_history.json").read_text()) if resume else []
    total_epochs = EXTENDED_BUDGET if resume else FIRST_BUDGET
    router_params = _router_parameters(model)
    if not router_params:
        raise RuntimeError("Stage2 content router missing from the screen trunk")
    accumulation = int(config["training"].get("gradient_accumulation_steps", 2))
    for epoch in range(start, total_epochs + 1):
        model.train()
        decoder.train()
        optimizer.zero_grad(set_to_none=True)
        total_n = total_q = total_loss = 0.0
        router_grad = router_delta = 0.0
        old_router = [parameter.detach().clone() for parameter in router_params]
        for batch_idx, batch in enumerate(loader):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss, parts = _batch_loss(model, decoder, batch, prior, device)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite sensory loss in epoch {epoch} batch {batch_idx}")
            scaler.scale(loss / accumulation).backward()
            total_n += float(parts.count.detach().mean().item())
            total_q += float(parts.timing.detach().mean().item())
            total_loss += float(loss.detach().item())
            if (batch_idx + 1) % accumulation == 0 or batch_idx + 1 == len(loader):
                scaler.unscale_(optimizer)
                router_grad += sum(
                    float(parameter.grad.detach().norm().item())
                    for parameter in router_params if parameter.grad is not None
                )
                torch.nn.utils.clip_grad_norm_(screen_parameters, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        router_delta = sum(
            float((parameter.detach() - old).norm().item())
            for parameter, old in zip(router_params, old_router, strict=True)
        )
        row = {
            "epoch": epoch,
            "count_loss": total_n / len(loader),
            "timing_loss": total_q / len(loader),
            "normalized_loss": total_loss / len(loader),
            "router_gradient_norm_sum": router_grad,
            "router_parameter_delta_norm": router_delta,
        }
        rows.append(row)
        write_json(rows, output / "fit_history.json")
        torch.save(
            {
                "model": model.state_dict(), "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "loader_rng": loader.generator.get_state(),
                "mode": mode, "past": past, "seed": seed, "epoch": epoch,
            }, output / "last_state.pt"
        )
        print(json.dumps(row), flush=True)
    def falling(key):
        first = rows[FIRST_BUDGET - 9][key]
        last = rows[FIRST_BUDGET - 1][key]
        return first > 0 and (first - last) / first > 0.05

    extension_needed = any(falling(key) for key in ("count_loss", "timing_loss"))
    profile_indices = split["fit_indices"][:8]
    profile = profile_model(
        model,
        build_loader(DatasetSubset(bundle.train, profile_indices), config["dataset"], False),
        device,
        max_samples=len(profile_indices),
    )
    profile["screen_context"] = {
        "normalization": "per-step GroupNorm",
        "classification_head": "random and not optimized during SSL",
        "training_only_decoder_excluded": True,
        "groupnorm_arithmetic": "not separately counted by hardware profiler",
        "sample_partition": "fit",
    }
    write_json(profile, output / "hardware_profile_v4.json")
    report = {
        "mode": mode, "past": past, "seed": seed, "epochs": total_epochs,
        "extension_requested_by_this_arm": extension_needed,
        "split_sha256": sha256_file(split_path),
        "config_resolved_sha256": sha256_file(resolved_config_path),
        "config_source": config.get("_source_path"),
        "git_commit": git_commit(), "preflight": preflight,
        "fit_count_scale": prior["scale_n"], "fit_timing_scale": prior["scale_q"],
        "fit_occupied_fraction": prior["occupied_fraction"],
        "fit_valid_cutoffs_per_utterance": prior["valid_cutoffs_per_utterance"],
        "optimized_parameters": sum(p.numel() for p in screen_parameters),
        "training_only_decoder_parameters": sum(p.numel() for p in decoder.parameters()),
        "training_only_decoder_dense_macs_per_utterance": 40 * (
            8 * 8 * 128 * 128 * 2 * 2
            + 16 * 16 * 128 * 128 * 3 * 3
            + 16 * 16 * 128 * (2 * (components + 1))
        ),
        "fit_utterances": len(split["fit_indices"]),
        "sealed_internal_holdout_utterances": len(split["holdout_indices"]),
        "optimizer": {"name": "AdamW", "learning_rate": 3e-4, "weight_decay": 5e-4,
                      "gradient_accumulation_steps": accumulation, "gradient_clip_norm": 1.0,
                      "amp_fp16": amp},
        "official_test_used": False, "development_validation_used": False,
    }
    write_json(report, output / "fit_report.json")
    return report
