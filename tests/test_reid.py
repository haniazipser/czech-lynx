import argparse
import torch

from config.presets.megadesc import get_config
from data.dataloader import CzechLynxDataModule
from data.transforms import get_transforms
from evaluation.retrieval import RetrievalEvaluator
from models.reid import MegaDescriptorModel
from evaluation.visualize import run_tsne_analysis
from evaluation.visualize_retrieval import run_all_visualizations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True,
                        help="WandB run ID z checkpointem do ewaluacji.")
    parser.add_argument("--no-viz", action="store_true",
                        help="Pomiń wizualizacje (t-SNE, UMAP, retrieval).")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dm = CzechLynxDataModule(cfg)
    model = MegaDescriptorModel()

    checkpoint = f"run/{args.run_id}/checkpoints/best.pt"
    print(f"Ładowanie checkpointu: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    _, val_transform = get_transforms(cfg.experiment_type)

    test_query_loader = dm.test_query_loader()
    test_gallery_loader = dm.test_gallery_loader()

    print("\n=== Ewaluacja na zbiorze testowym ===")
    evaluator = RetrievalEvaluator(device)
    metrics = evaluator.evaluate(model, test_query_loader, test_gallery_loader)

    print(f"  Rank-1  : {metrics['rank1']:.4f}")
    print(f"  Rank-5  : {metrics['rank5']:.4f}")
    print(f"  mAP     : {metrics['map']:.4f}")

    if not args.no_viz:

        print("\nRunning t-SNE...")
        run_tsne_analysis(
            model=model,
            dataset=dm.test_ds,
            device=device,
            val_transform=val_transform,
            color_by=["identity", "trap_id"],
            save_path=f"run/{args.run_id}/test_tsne.png",
        )

        run_all_visualizations(
            model=model,
            query_dataset=dm.test_query_ds,
            gallery_dataset=dm.test_gallery_ds,
            device=device,
            val_transform=val_transform,
            save_dir=f"run/{args.run_id}/test_retrieval",
        )


if __name__ == "__main__":
    main()