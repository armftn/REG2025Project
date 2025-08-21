import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.cluster import KMeans # <-- NOUVEL IMPORT

class WsiReportDataset(Dataset):
    """
    Dataset simple qui travaille avec des données déjà chargées en mémoire RAM.
    Il utilise un échantillonnage intelligent par clustering K-Means.
    """
    def __init__(self, data_df: pd.DataFrame, features_list: list, n_patches: int = 100, n_clusters: int = 4):
        super().__init__()
        self.data_info = data_df
        # La liste des features est passée directement à l'initialisation
        self.all_features = features_list
        self.n_patches = n_patches
        self.n_clusters = n_clusters # Nombre de "types de tissus" à identifier
        print(f"Dataset initialisé avec {len(self.all_features)} features pré-chargées.")

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # On récupère les features pré-chargées directement depuis la RAM (très rapide)
        all_patches_features = self.all_features[idx]
        
        # --- NOUVELLE LOGIQUE D'ÉCHANTILLONNAGE STRATIFIÉ PAR CLUSTERING ---
        
        # On s'assure qu'il y a assez de patchs pour le clustering
        if len(all_patches_features) < self.n_clusters:
            # Si pas assez de patchs, on revient à l'échantillonnage simple
            replace = len(all_patches_features) < self.n_patches
            indices = np.random.choice(len(all_patches_features), self.n_patches, replace=replace)
            features_array = all_patches_features[indices]
        else:
            # 1. On applique K-Means pour trouver des "types" de patchs
            # n_init='auto' est nécessaire pour les versions récentes de scikit-learn
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto').fit(all_patches_features)
            cluster_labels = kmeans.labels_

            # 2. On échantillonne un nombre égal de patchs dans chaque cluster
            samples_per_cluster = self.n_patches // self.n_clusters
            sampled_indices = []
            
            for i in range(self.n_clusters):
                # On récupère les indices des patchs appartenant à ce cluster
                indices_in_cluster = np.where(cluster_labels == i)[0]
                
                # Si le cluster est vide (cas rare), on passe au suivant
                if len(indices_in_cluster) == 0:
                    continue
                
                # On détermine si on a besoin de tirer avec remise pour ce cluster spécifique
                replace = len(indices_in_cluster) < samples_per_cluster
                
                # On tire `samples_per_cluster` patchs de ce cluster
                chosen_indices = np.random.choice(indices_in_cluster, samples_per_cluster, replace=replace)
                sampled_indices.extend(chosen_indices)
            
            # On s'assure d'avoir exactement `n_patches` au total, au cas où un cluster était vide
            while len(sampled_indices) < self.n_patches:
                # On complète avec des patchs tirés au hasard parmi tous les patchs
                sampled_indices.append(np.random.randint(0, len(all_patches_features)))

            features_array = all_patches_features[sampled_indices]
        # --- FIN DE LA NOUVELLE LOGIQUE ---

        # On récupère les métadonnées correspondantes.
        sample_info = self.data_info.iloc[idx]
        
        features_tensor = torch.from_numpy(features_array).float()
        
        return {
            'id': sample_info['id'],
            'features': features_tensor,
            'report': sample_info['report'],
            'gleason_label': torch.tensor(sample_info["gleason_score"], dtype=torch.long),
            'grade_group_label': torch.tensor(sample_info["grade_group"], dtype=torch.long),
            'tumor_volume_label': torch.tensor(sample_info["tumor_volume"], dtype=torch.float)
        }