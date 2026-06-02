from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from datasets import tqdm
from lightglue.utils import load_image

from config.presets.baseline import get_config
from data.dataloader import CzechLynxDataModule
from wildfusion.wildfusion import LocalMatcher, get_bbox_from_rle


def load_and_crop_via_rle(img_path, mask_str, device, target_size=512):
    """
    Pancerna wersja ładowania – odporna na uszkodzone/puste maski RLE
    oraz błędy PIL DecompressionBombError.
    """
    try:
        # 1. Pobieramy boks z maski
        bbox = get_bbox_from_rle(mask_str)
        xmin, ymin, xmax, ymax = bbox

        # OCHRONA: Sprawdzamy, czy boks nie ma absurdalnych wymiarów (np. ujemne lub gigantyczne)
        width = xmax - xmin
        height = ymax - ymin
        if width <= 0 or height <= 0 or width > 5000 or height > 5000:
            # Ponad 5000 px dla samego rysia to anomalia w masce, bezpieczniej pominąć
            return None

        # 2. Otwieramy obraz i sprawdzamy jego rzeczywiste wymiary
        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size

        # OCHRONA: Przycinamy boks do fizycznych granic obrazu, gdyby wykraczał na krawędziach
        xmin = max(0, min(xmin, img_w))
        ymin = max(0, min(ymin, img_h))
        xmax = max(0, min(xmax, img_w))
        ymax = max(0, min(ymax, img_h))

        # Ponowna weryfikacja czy po przycięciu cokolwiek zostało z wycinka
        if (xmax - xmin) <= 5 or (ymax - ymin) <= 5:
            return None

        # 3. Bezpieczne wycięcie
        cropped_img = img.crop((xmin, ymin, xmax, ymax))

        # Zapisujemy tymczasowo plik dla LightGlue
        tmp_path = Path("tmp_crop.jpg")
        cropped_img.save(tmp_path)

        # Ładujemy i skalujemy za pomocą load_image z LightGlue
        tensor = load_image(tmp_path, resize=target_size).to(device)
        return tensor

    except Exception:
        # W razie jakiegokolwiek innego błędu (np. brak pliku) zwracamy None,
        # co pozwoli pętli głównej bezpiecznie przejść do kolejnego zdjęcia.
        print("SKIPPP")
        return None

def evaluate_reid(query_df, gallery_df, data_root, matcher, device):
    """Przechodzi pętlą Query vs Gallery na surowych ramkach danych."""
    results = []
    base_path = Path(data_root)

    print(f"Rozpoczynam ewaluację Re-ID. Query: {len(query_df)} | Gallery: {len(gallery_df)}")

    for q_idx, q_row in tqdm(query_df.iterrows(), total=len(query_df), desc="Query loop"):
        # Używamy poprawnej nazwy kolumny: "path"
        q_path = base_path / q_row['path']
        q_tensor = load_and_crop_via_rle(q_path, q_row['mask'], device)
        if q_tensor is None:
            continue

        for g_idx, g_row in gallery_df.iterrows():
            # Jeśli to dokładnie to samo zdjęcie, pomiń
            if q_row['path'] == g_row['path']:
                continue

            g_path = base_path / g_row['path']
            g_tensor = load_and_crop_via_rle(g_path, g_row['mask'], device)
            if g_tensor is None:
                continue

            # Dopasowanie cech przez LightGlue
            matches_count = matcher.get_matches_count_from_tensors(q_tensor, g_tensor, conf_threshold=0.1)

            # Używamy poprawnej nazwy kolumny: "identity"
            is_same_lynx = (q_row['identity'] == g_row['identity'])

            results.append({
                'query_img': q_row['path'],
                'gallery_img': g_row['path'],
                'query_identity': q_row['identity'],
                'gallery_identity': g_row['identity'],
                'is_same_ground_truth': is_same_lynx,
                'lightglue_matches': matches_count
            })

    return pd.DataFrame(results)


def main():
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = LocalMatcher(device)

    # 1. Inicjalizacja Twojego DataModule (użyj swojego configu)
    data_module = CzechLynxDataModule(cfg=cfg)

    # 2. Dobieramy się bezpośrednio do surowego pola .df wewnątrz datasetów
    # Przechodzi to całkowicie OBOK transformacji w __getitem__!
    val_query_df = data_module.val_query_ds.df
    val_gallery_df = data_module.val_gallery_ds.df

    # 3. Odpalenie ewaluacji
    results_df = evaluate_reid(
        query_df=val_query_df,
        gallery_df=val_gallery_df,
        data_root=cfg.data_root,
        matcher=matcher,
        device=device
    )

    # 4. Zapis wyników do CSV
    output_path = "val_reid_lightglue_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n[Sukces] Pipeline ukończony bez ingerencji w transformacje! Wyniki: {output_path}")

if __name__ == "__main__":
    main()