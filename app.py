"""
Gradio Space for TinyWorlds — Autoregressive World Model demo.
Deploy at: https://huggingface.co/spaces
"""
import os
import sys
import random
import re
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from einops import repeat
from huggingface_hub import hf_hub_download, list_repo_files

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_CACHE_DIR = BASE_DIR / "model_cache"
DATA_DIR.mkdir(exist_ok=True)
MODEL_CACHE_DIR.mkdir(exist_ok=True)

MODELS_REPO_ID = "AlmondGod/tinyworlds-models"
DATASET_REPO_ID = "AlmondGod/tinyworlds"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SUITE = "sonic"
DATASET = "SONIC"

# ---------------------------------------------------------------------------
# Asset download helpers
# ---------------------------------------------------------------------------

def _download_suite_checkpoints(suite: str = SUITE) -> Path:
    """Download all .pt checkpoint files for a given suite from HF model repo."""
    suite_dir = MODEL_CACHE_DIR / suite
    suite_dir.mkdir(exist_ok=True)

    all_files = list(list_repo_files(MODELS_REPO_ID, repo_type="model"))
    suite_files = [f for f in all_files if f.startswith(f"{suite}/") and f.endswith(".pt")]

    if not suite_files:
        raise FileNotFoundError(
            f"No checkpoints found for suite '{suite}' in {MODELS_REPO_ID}. "
            "Make sure the HF model repo has files under the suite folder."
        )

    for fname in suite_files:
        local = MODEL_CACHE_DIR / fname
        if not local.exists():
            hf_hub_download(
                repo_id=MODELS_REPO_ID,
                filename=fname,
                repo_type="model",
                local_dir=str(MODEL_CACHE_DIR),
                local_dir_use_symlinks=False,
            )
    return suite_dir


def _find_ckpt_dir(suite_dir: Path, model_name: str) -> Path:
    """Return the checkpoint directory with the highest step number for a model."""
    aliases = {
        "video_tokenizer": ["video_tokenizer"],
        "latent_actions": ["latent_actions", "lam", "actions", "action_tokenizer"],
        "dynamics": ["dynamics"],
    }
    names = aliases.get(model_name, [model_name])

    candidates = []
    for state_file in suite_dir.rglob("model_state_dict.pt"):
        parent = state_file.parent
        if any(n in str(parent) for n in names):
            candidates.append(parent)

    if not candidates:
        raise FileNotFoundError(f"No checkpoint directory for '{model_name}' in {suite_dir}")

    def _step(p: Path) -> int:
        m = re.search(r"_step_(\d+)", str(p))
        return int(m.group(1)) if m else 0

    return max(candidates, key=_step)


def _ensure_dataset_h5(h5_filename: str) -> None:
    """Download dataset h5 file from HF if not already present."""
    dest = DATA_DIR / h5_filename
    if dest.exists():
        return
    print(f"Downloading {h5_filename} from HF dataset repo…")
    hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=h5_filename,
        repo_type="dataset",
        local_dir=str(DATA_DIR),
        local_dir_use_symlinks=False,
    )


# ---------------------------------------------------------------------------
# Model cache (loaded once, reused across requests)
# ---------------------------------------------------------------------------

_model_cache: dict = {}


def get_models():
    if SUITE in _model_cache:
        return _model_cache[SUITE]

    print("Downloading / loading model checkpoints…")
    suite_dir = _download_suite_checkpoints(SUITE)

    vt_path = _find_ckpt_dir(suite_dir, "video_tokenizer")
    lam_path = _find_ckpt_dir(suite_dir, "latent_actions")
    dyn_path = _find_ckpt_dir(suite_dir, "dynamics")

    from utils.inference_utils import load_models  # noqa: PLC0415
    vt, lam, dyn = load_models(str(vt_path), str(lam_path), str(dyn_path), DEVICE, use_actions=True)
    _model_cache[SUITE] = (vt, lam, dyn)
    print(f"Models loaded on {DEVICE}.")
    return _model_cache[SUITE]


# ---------------------------------------------------------------------------
# Dataset cache
# ---------------------------------------------------------------------------

_dataset_cache: dict = {}


def get_val_dataset(num_frames: int):
    if (DATASET, num_frames) in _dataset_cache:
        return _dataset_cache[(DATASET, num_frames)]

    _ensure_dataset_h5("sonic_frames.h5")

    from datasets.data_utils import load_data_and_data_loaders  # noqa: PLC0415
    _, val_data, _, _, _ = load_data_and_data_loaders(
        dataset=DATASET, batch_size=1, num_frames=num_frames, preload_ratio=0.15
    )
    _dataset_cache[(DATASET, num_frames)] = val_data
    return val_data


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _frames_to_mp4(frames_01: torch.Tensor, path: str, fps: int = 4) -> None:
    """Save [1, T, C, H, W] float tensor (values in [0,1]) as an MP4 file."""
    B, T, C, H, W = frames_01.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
    for t in range(T):
        frame = (frames_01[0, t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        if frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def run_inference(
    n_steps: int,
    temperature: float,
    use_random_actions: bool,
    progress=gr.Progress(),
) -> str:
    context_window = 2
    prediction_horizon = 1
    frames_to_load = context_window + n_steps * prediction_horizon

    progress(0.0, desc="Loading models…")
    video_tokenizer, latent_action_model, dynamics_model = get_models()

    progress(0.15, desc="Loading dataset sample…")
    val_data = get_val_dataset(frames_to_load)

    idx = random.randint(0, len(val_data) - 1)
    gt_frames = val_data[idx][0].unsqueeze(0).to(DEVICE)  # [1, T, C, H, W]

    context_frames = gt_frames[:, :context_window]
    generated_frames = context_frames.clone()

    n_actions = latent_action_model.quantizer.codebook_size
    inferred_actions: list = []

    with torch.no_grad():
        for i in range(n_steps):
            progress(0.15 + 0.75 * (i / n_steps), desc=f"Generating frame {i + 1}/{n_steps}…")

            ctx = generated_frames[:, -context_window:]
            video_indices = video_tokenizer.tokenize(ctx)
            video_latents = video_tokenizer.quantizer.get_latents_from_indices(video_indices)

            action_latent = None
            if use_random_actions:
                from utils.inference_utils import sample_random_action  # noqa: PLC0415

                action_idx = sample_random_action(n_actions)
                inferred_actions.append(action_idx)
                recent = inferred_actions[-context_window:]
                recent_t = repeat(torch.tensor(recent, device=DEVICE), "i -> 1 i")
                action_latent = latent_action_model.quantizer.get_latents_from_indices(recent_t)

                if len(recent) < context_window:
                    pad_len = context_window - len(recent) + 1
                    pad_enc = latent_action_model.encode(ctx[:, :pad_len])
                    pad_q = latent_action_model.quantizer(pad_enc)
                    action_latent = torch.cat([pad_q, action_latent], dim=1)

            def idx_to_latents(idx):
                return video_tokenizer.quantizer.get_latents_from_indices(idx, dim=-1)

            next_latents = dynamics_model.forward_inference(
                context_latents=video_latents,
                prediction_horizon=prediction_horizon,
                num_steps=10,
                index_to_latents_fn=idx_to_latents,
                conditioning=action_latent,
                temperature=temperature,
            )

            next_frames = video_tokenizer.detokenize(next_latents)
            generated_frames = torch.cat(
                [generated_frames, next_frames[:, -prediction_horizon:]], dim=1
            )

    progress(0.92, desc="Rendering output…")

    gen = torch.clamp((generated_frames.detach().cpu() + 1) / 2, 0, 1)
    gt = torch.clamp((gt_frames[:, : gen.shape[1]].detach().cpu() + 1) / 2, 0, 1)

    B, T, C, H, W = gen.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = str(BASE_DIR / "demo_output.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, 4, (W * 2, H))
    for t in range(T):
        gt_f = (gt[0, t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gn_f = (gen[0, t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        combined = np.concatenate([gt_f, gn_f], axis=1)
        writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    writer.release()

    progress(1.0, desc="Done!")
    return out_path


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="TinyWorlds — World Model Demo") as demo:
    gr.Markdown(
        """
# TinyWorlds — Autoregressive World Model

A minimal implementation of [Google DeepMind's Genie architecture](https://arxiv.org/pdf/2402.15391).
The model learns to predict future game frames **without any action labels** — actions are inferred unsupervised.

**Left half**: ground truth &nbsp;|&nbsp; **Right half**: model predictions

> First generation takes longer — models and dataset are downloaded on first run.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            n_steps = gr.Slider(2, 15, value=8, step=1, label="Frames to generate")
            temperature = gr.Slider(
                0.0, 1.0, value=0.5, step=0.05,
                label="Temperature  (0 = deterministic, 1 = max diversity)"
            )
            use_actions = gr.Checkbox(value=True, label="Condition on random actions")
            btn = gr.Button("Generate", variant="primary", size="lg")

        with gr.Column(scale=2):
            video_out = gr.Video(
                label="Ground Truth (left)  ·  Predicted (right)",
                autoplay=True,
            )

    btn.click(
        fn=run_inference,
        inputs=[n_steps, temperature, use_actions],
        outputs=[video_out],
    )

if __name__ == "__main__":
    demo.launch()
