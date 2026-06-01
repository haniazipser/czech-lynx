import random
import json
import traceback
from pathlib import Path
import torch
from config.presets.megadesc import get_config
from data.dataloader import CzechLynxDataModule
from run_reid import run_experiment
from evaluation.retrieval import RetrievalEvaluator
from models.reid import MegaDescriptorModel

SEARCH_SPACE = {
    "arcface_m":  [0.1, 0.2, 0.3, 0.4, 0.5],
    "arcface_s":  [16.0, 32.0, 48.0, 64.0],
    "lr":         [1e-5, 3e-5, 1e-4],
    "arcface_lr": [1e-4, 3e-4, 1e-3],
}

REGISTRY_PATH = Path("tune_registry.json")
N_TRIALS      = 10
SEED          = 42


def sample_params(trial: int) -> dict:
    rng = random.Random(SEED + trial)
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def run_name(params: dict) -> str:
    return f"tune_m{params['arcface_m']}_s{params['arcface_s']}_lr{params['lr']}_alr{params['arcface_lr']}"


def main():
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registry = json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else {}

    for trial in range(N_TRIALS):
        params = sample_params(trial)
        name   = run_name(params)

        if registry.get(name, {}).get("status") == "done":
            print(f"SKIP {name}")
            continue

        print(f"\n[{trial+1}/{N_TRIALS}] {name}")
        print(f"  params: {params}")

        cfg            = get_config()
        cfg.arcface_m  = params["arcface_m"]
        cfg.arcface_s  = params["arcface_s"]
        cfg.lr         = params["lr"]
        cfg.arcface_lr = params["arcface_lr"]
        cfg.epochs     = 10

        try:
            run_id = run_experiment(cfg, device)

            # Eval on result
            model = MegaDescriptorModel().to(device)
            dm = CzechLynxDataModule(cfg)

            model.load_state_dict(torch.load(
                f"run/{run_id}/checkpoints/best.pt", map_location=device
            ))
            evaluator = RetrievalEvaluator(device)
            metrics   = evaluator.evaluate(
                model, dm.val_query_loader(), dm.val_gallery_loader()
            )
            metrics["wandb_run_id"] = run_id

            registry[name] = {"status": "done", "params": params, "metrics": metrics}
            print(f"  Rank-1: {metrics['rank1']:.4f}  mAP: {metrics['map']:.4f}")

        except Exception as e:
            registry[name] = {
                "status": "failed", "params": params,
                "error":  traceback.format_exc()
            }
            print(f"  FAILED: {e}")

        REGISTRY_PATH.write_text(json.dumps(registry, indent=2))

    # Best run
    done = {k: v for k, v in registry.items() if v["status"] == "done"}
    if done:
        best = max(done.items(), key=lambda x: x[1]["metrics"]["rank1"])
        print(f"\nBEST: {best[0]}")
        print(f"  params:  {best[1]['params']}")
        print(f"  Rank-1:  {best[1]['metrics']['rank1']:.4f}")
        print(f"  mAP:     {best[1]['metrics']['map']:.4f}")
        print(f"  wandb:   {best[1]['metrics']['wandb_run_id']}")


if __name__ == "__main__":
    main()