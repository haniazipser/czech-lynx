import json
from pathlib import Path
from pycocotools import _mask as coco_mask
import torch
from PIL.Image import Image
from lightglue import ALIKED, LightGlue

class LocalMatcher:
    def __init__(self, device: torch.device):

        self.device = device
        # Inicjalizacja modeli z domyślnymi wagami
        self.extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
        self.matcher = LightGlue(features='aliked').eval().to(device)

    def get_matches_count(self, img_path_a: str, img_path_b: str, conf_threshold: float = 0.85) -> int:
        from lightglue.utils import load_image
        import torch

        # 1. Wczytanie obrazów
        image0 = load_image(str(img_path_a)).to(self.device)
        image1 = load_image(str(img_path_b)).to(self.device)

        # 2. Ekstrakcja i dopasowanie
        with torch.no_grad():
            feats0 = self.extractor.extract(image0)
            feats1 = self.extractor.extract(image1)
            matches = self.matcher({'image0': feats0, 'image1': feats1})

        # 3. FILTROWANIE: Wyciągamy indeksy dopasowań i ich stopień pewności (scores)
        # matches['matches'][0] zawiera pary indeksów (N, 2)
        # matches['scores'][0] zawiera pewność dla każdego dopasowania (N,)
        valid_indices = matches['scores'][0] > conf_threshold

        # Liczba dopasowań, które przeszły próg pewności
        filtered_matches_count = torch.sum(valid_indices).item()

        return filtered_matches_count

    def get_matches_count_from_tensors(self, tensor0: torch.Tensor, tensor1: torch.Tensor,
                                       conf_threshold: float = 0.0) -> int:
        with torch.no_grad():
            # LightGlue oczekuje wymiaru batcha [1, C, H, W]
            if len(tensor0.shape) == 3:
                tensor0 = tensor0.unsqueeze(0)
            if len(tensor1.shape) == 3:
                tensor1 = tensor1.unsqueeze(0)

            feats0 = self.extractor.extract(tensor0)
            feats1 = self.extractor.extract(tensor1)
            matches = self.matcher({'image0': feats0, 'image1': feats1})

        if conf_threshold > 0:
            valid_indices = matches['scores'][0] > conf_threshold
            return torch.sum(valid_indices).item()

        return len(matches['matches'][0])

def get_bbox_from_rle(mask_string: str) -> list:
    """
    Dekoduje maskę COCO RLE zapisaną jako string i zwraca bounding box [xmin, ymin, xmax, ymax].
    Bezpieczna dla surowych danych z Windows/CSV ze znakami escape.
    """
    if isinstance(mask_string, str):
        # 1. Zamieniamy podwójne cudzysłowy na pojedyncze
        cleaned_str = mask_string.replace('""', '"')
        # 2. KLUCZOWE: Naprawiamy znaki ucieczki (slasze), żeby json.loads się nie wywalił
        cleaned_str = cleaned_str.replace('\\', '\\\\')

        mask_dict = json.loads(cleaned_str)
    else:
        mask_dict = mask_string

    # 3. pycocotools wymaga, aby string w 'counts' był zakodowany jako bajty (bytes)
    if isinstance(mask_dict['counts'], str):
        mask_dict['counts'] = bytes(mask_dict['counts'], 'utf-8')

    # 4. Wyciągamy boks w formacie COCO: [x_min, y_min, width, height]
    coco_bbox = coco_mask.toBbox([mask_dict])[0]

    # 5. Przeliczamy na format PIL [xmin, ymin, xmax, ymax]
    xmin = int(coco_bbox[0])
    ymin = int(coco_bbox[1])
    xmax = int(xmin + coco_bbox[2])
    ymax = int(ymin + coco_bbox[3])

    return [xmin, ymin, xmax, ymax]


from sklearn.isotonic import IsotonicRegression

class ScoresCalibrator:
    def __init__(self):
        # out_of_bounds='clip' zabezpiecza przed wartościami spoza zakresu treningowego
        self.global_regressor = IsotonicRegression(out_of_bounds='clip')
        self.local_regressor = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False

    def fit(self, global_scores: list, local_scores: list, labels: list):
        """
        Trenuje kalibratory na zbiorze par.
        labels: lista 1 (ten sam rys) lub 0 (różne rysie)
        """
        self.global_regressor.fit(global_scores, labels)
        self.local_regressor.fit(local_scores, labels)
        self.is_fitted = True

    def predict_probability(self, global_score: float, local_score: float) -> float:
        """Zwraca ostateczne, połączone prawdopodobieństwo (średnia z obu)"""
        if not self.is_fitted:
            # Jeśli brak kalibracji, zwracamy prostą kombinację heurystyczną
            return float(global_score)

        p_global = self.global_regressor.predict([global_score])[0]
        p_local = self.local_regressor.predict([local_score])[0]

        # Fuzja późna (Late Fusion) - średnia arytmetyczna prawdopodobieństw
        return float((p_global + p_local) / 2.0)


class WildFusionEvaluator:
    def __init__(self, global_model, device: torch.device, top_k: int = 20):
        self.global_model = global_model.eval().to(device)
        self.device = device
        self.top_k = top_k

        self.local_matcher = LocalMatcher(device)
        self.calibrator = ScoresCalibrator()

    def prepare_calibration_data(self, query_loader, gallery_loader):
        """
        KROK A: Przechodzimy przez część danych, generujemy surowe wyniki
        i karmimy self.calibrator.fit()
        """
        pass

    def evaluate(self, query_loader, gallery_loader) -> dict:
        """
        KROK B: Docelowa ewaluacja.
        1. Wyciągnij embeddingi globalne dla całego Query i Gallery.
        2. Oblicz macierz cosinusową.
        3. Dla każdego query weź Top-K z gallery.
        4. Dla tych par Top-K odpal self.local_matcher.get_matches_count().
        5. Przepuść wyniki przez self.calibrator.predict_probability().
        6. Przesortuj na nowo i oblicz końcowy mAP / Rank-1.
        """
        pass