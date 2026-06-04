
import torch
from lightglue import ALIKED, LightGlue
import numpy as np
from sklearn.isotonic import IsotonicRegression

class LocalMatcher:
    def __init__(self, device: torch.device):

        self.device = device
        self.extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
        self.matcher = LightGlue(features='aliked').eval().to(device)

    def get_matches_count_from_tensors(self, tensor0: torch.Tensor, tensor1: torch.Tensor,
                                       conf_threshold: float = 0.0) -> int:
        with torch.no_grad():
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

class ScoresCalibrator:

    def __init__(self):
        self.global_regressor = IsotonicRegression(out_of_bounds="clip")
        self.local_regressor = IsotonicRegression(out_of_bounds="clip")
        self.is_fitted = False
        self.epsilon = 1e-5

    def fit(self, global_scores: list, local_scores: list, labels: list):
        global_scores = np.asarray(global_scores)
        local_scores = np.asarray(local_scores)
        labels = np.asarray(labels).astype(int)

        self.global_regressor.fit(global_scores, labels)
        self.local_regressor.fit(local_scores, labels)
        self.is_fitted = True
        return self

    def predict_probability(self, global_scores, local_scores):
        if not self.is_fitted:
            return global_scores

        g_arr = np.asarray(global_scores)
        l_arr = np.asarray(local_scores)

        p_global = self.global_regressor.predict(g_arr)
        p_local = self.local_regressor.predict(l_arr)

        p_global = np.clip(p_global, self.epsilon, 1.0 - self.epsilon)
        p_local = np.clip(p_local, self.epsilon, 1.0 - self.epsilon)

        # Late Fusion - Product Rule
        numerator = p_global * p_local
        denominator = numerator + (1.0 - p_global) * (1.0 - p_local)
        combined_prob = numerator / denominator

        if np.isscalar(global_scores) and np.isscalar(local_scores):
            return float(combined_prob[0])

        return combined_prob

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