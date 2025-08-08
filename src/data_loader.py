import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.utils import extract_all_labels
from tqdm import tqdm

class WsiReportDataset(Dataset):
    """
    Dataset capable de charger des features soit depuis des fichiers .npy (pré-chargés en RAM),
    soit depuis des fichiers .parquet (chargés à la volée avec échantillonnage de patchs).
    """
    def __init__(self, data_df: pd.DataFrame, features_path: str, feature_type: str = 'parquet', n_patches: int = 100):
        super().__init__()
        self.data_info = data_df
        self.features_path = features_path
        self.feature_type = feature_type
        self.n_patches = n_patches

        if self.feature_type == 'npy':
            print("Mode NPY: Pré-chargement de toutes les features en RAM...")
            self.all_features = []
            for idx, row in tqdm(self.data_info.iterrows(), total=len(self.data_info)):
                features_array = self._load_features_npy(row['id'])
                self.all_features.append(features_array)
            print(f"Dataset initialisé. {len(self.all_features)} tenseurs de features chargés en RAM.")
        
        elif self.feature_type == 'parquet':
            print("Mode Parquet: Les features seront chargées et échantillonnées à la volée.")
        
        else:
            raise ValueError("Type de feature non supporté. Choisissez 'npy' ou 'parquet'.")

    def __len__(self):
        return len(self.data_info)
    
    def _load_features_npy(self, slide_filename: str):
        slide_id = slide_filename.replace('.tiff', '')
        feature_filename = f"features_{slide_id}_top64_downsampled2x.npy"
        filepath = os.path.join(self.features_path, feature_filename)
        return np.load(filepath)

    def _load_features_parquet(self, slide_filename: str):
        slide_id = slide_filename.replace('.tiff', '')
        feature_filename = f"{slide_id}.parquet"
        filepath = os.path.join(self.features_path, feature_filename)
        try:
            patch_df = pd.read_parquet(filepath)
            feature_columns = [f'feat{i}' for i in range(1, 513)]
            return patch_df[feature_columns].values
        except FileNotFoundError:
            return None

    def __getitem__(self, idx):
        sample_info = self.data_info.iloc[idx]
        slide_filename = sample_info['id']
        report_text = sample_info['report']
        labels = {
            "gleason_score": sample_info['gleason_score'],
            "grade_group": sample_info['grade_group'],
            "tumor_volume": sample_info['tumor_volume']
        }

        features_array = None
        if self.feature_type == 'npy':
            features_array = self.all_features[idx]
        
        elif self.feature_type == 'parquet':
            all_patches_features = self._load_features_parquet(slide_filename)
            
            if all_patches_features is not None:
                # --- CORRECTION DE LA LOGIQUE D'ÉCHANTILLONNAGE ---
                # On s'assure que chaque tenseur a TOUJOURS la même taille (`n_patches`).
                num_available_patches = len(all_patches_features)
                
                # On détermine si on a besoin de tirer avec remise (replace=True)
                replace = num_available_patches < self.n_patches
                
                # On tire toujours `n_patches` indices.
                indices = np.random.choice(num_available_patches, self.n_patches, replace=replace)
                features_array = all_patches_features[indices]
                # --- FIN DE LA CORRECTION ---

        if features_array is None:
            print(f"Features non trouvées pour {slide_filename}, renvoi de l'échantillon 0 par défaut.")
            return self.__getitem__(0)

        features_tensor = torch.from_numpy(features_array).float()
        
        return {
            'id': slide_filename,
            'features': features_tensor,
            'report': report_text,
            'gleason_label': torch.tensor(labels["gleason_score"], dtype=torch.long),
            'grade_group_label': torch.tensor(labels["grade_group"], dtype=torch.long),
            'tumor_volume_label': torch.tensor(labels["tumor_volume"], dtype=torch.float)
        }