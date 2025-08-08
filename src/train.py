import torch
import numpy as np
import pandas as pd
import re
import os
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler

# On importe nos fonctions et classes depuis les autres fichiers
from src.data_loader import WsiReportDataset
from src.models.transmil import TransMIL
from src.utils import load_config, extract_all_labels # On utilise la nouvelle fonction
from src.trainer import train_one_epoch, validate_one_epoch

def main():
    # --- Étape 1 : CHARGEMENT DE LA CONFIGURATION ---
    # On revient à la lecture du fichier YAML local pour un entraînement simple
    config = load_config()
    print("Configuration chargée :", config)

    # --- Étape 2 : PRÉPARATION ET NETTOYAGE DES DONNÉES ---
    print("Chargement et filtrage des données...")
    full_df = pd.read_json(config['data_paths']['json_path'])
    
    # --- FILTRAGE CRUCIAL : Ne garder que les biopsies de la prostate ---
    prostate_df = full_df[full_df['report'].str.contains("Prostate, biopsy", case=False, na=False)].reset_index(drop=True)
    print(f"{len(prostate_df)} rapports de prostate trouvés.")
    
    # --- MODIFICATION : On construit le chemin complet vers les données d'entraînement .parquet ---
    base_features_path = config['data_paths']['features_path']
    features_path = os.path.join(base_features_path, "train")
    print(f"Utilisation des features depuis : {features_path}")
    
    # --- MODIFICATION : On vérifie l'existence des fichiers .parquet ---
    # On adapte la fonction pour chercher les fichiers .parquet avec le bon nom.
    expected_filenames = prostate_df['id'].apply(lambda x: f"{x.replace('.tiff', '')}.parquet")
    file_exists_mask = expected_filenames.apply(lambda x: os.path.exists(os.path.join(features_path, x)))
    cleaned_df = prostate_df[file_exists_mask].reset_index(drop=True)
    print(f"{len(cleaned_df)}/{len(prostate_df)} rapports de prostate ont un fichier .parquet correspondant.")
    
    # On utilise la nouvelle fonction pour extraire tous les labels
    print("Extraction et filtrage des étiquettes multiples...")
    labels_series = cleaned_df['report'].apply(extract_all_labels)
    labels_df = pd.json_normalize(labels_series.tolist())
    cleaned_df = cleaned_df.join(labels_df)

    # On filtre pour ne garder que les lignes où le score de Gleason et le Grade Group sont valides
    final_df = cleaned_df[(cleaned_df['gleason_score'] != -1) & (cleaned_df['grade_group'] != -1)].reset_index(drop=True)
    
    # La liste `valid_labels` pour la stratification reste basée sur le score de Gleason pour la cohérence
    valid_labels = final_df['gleason_score'].tolist()
    print(f"{len(final_df)} échantillons de prostate valides pour l'entraînement trouvés.")
    
    if not valid_labels:
        print("ERREUR : Aucune donnée valide pour l'entraînement.")
        return

    # --- Étape 3 : CRÉATION DU DATASET PYTORCH ---
    # --- MODIFICATION : On spécifie le type de features à 'parquet' ---
    full_dataset = WsiReportDataset(
        data_df=final_df, 
        features_path=features_path,
        feature_type='parquet'  # <-- On indique d'utiliser la nouvelle logique
    )

    # --- Étape 4 : MISE EN PLACE DE LA VALIDATION CROISÉE ---
    skf = StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=42)
    X_indices = np.arange(len(full_dataset))

    # --- Étape 5 : BOUCLE PRINCIPALE SUR LES PLIS (FOLDS) ---
    for fold, (train_indices, val_indices) in enumerate(skf.split(X=X_indices, y=valid_labels)):
        print(f"\n=============== DÉMARRAGE DU FOLD {fold+1}/{config['n_splits']} ================")
        
        run = wandb.init(project=config['project_name'], config=config, name=f"HistoEncoder_multitask_fold_{fold+1}", reinit=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        
        # --- Logique du Sampler et des DataLoaders ---
        train_labels_for_fold = [valid_labels[i] for i in train_indices]
        class_counts = np.bincount(train_labels_for_fold, minlength=config['model_params']['n_classes'])
        weights_per_class = 1.0 / (class_counts + 1e-6)
        sample_weights = np.array([weights_per_class[label] for label in train_labels_for_fold])
        sampler = WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_subset, batch_size=config['batch_size'], sampler=sampler, num_workers=config['num_workers'], pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)
        
        # --- CORRECTION : On accède aux paramètres du modèle via la clé 'model_params' ---
        model = TransMIL(**config['model_params']).to(device)
        
        # --- CORRECTION : On initialise les 3 fonctions de coût ET les poids ---
        # On calcule les poids pour la loss pondérée du Grade Group
        grade_group_labels_for_fold = [full_dataset[i]['grade_group_label'].item() for i in train_indices]
        gg_counts = np.bincount(grade_group_labels_for_fold, minlength=5)
        gg_weights = 1.0 / (gg_counts + 1e-6)
        gg_weights = torch.tensor(gg_weights, dtype=torch.float).to(device)

        # On crée le dictionnaire de 'criteria' en y ajoutant la clé 'weights'
        criteria = {
            'gleason': torch.nn.CrossEntropyLoss(), # Le sampler gère l'équilibre pour le Gleason
            'grade_group': torch.nn.CrossEntropyLoss(weight=gg_weights),
            'volume': torch.nn.MSELoss(),
            # On ajoute les poids pour la loss combinée
            'weights': config.get('loss_weights', {'gleason': 1.0, 'grade_group': 1.0, 'volume': 1.0})
        }
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 1e-4))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-7)
        scaler = GradScaler()
        best_val_loss = float('inf')
        patience_counter = 0

        # --- BOUCLE D'ENTRAÎNEMENT SUR LES ÉPOQUES ---
        for epoch in range(config['epochs']):
            print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
            
            train_loss = train_one_epoch(model, train_loader, criteria, optimizer, device, scaler, scheduler)
            val_metrics = validate_one_epoch(model, val_loader, criteria, device)

            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Gleason Acc: {val_metrics['val_accuracy_gleason']:.4f} | "
                f"Grade Group Acc: {val_metrics['val_accuracy_grade_group']:.4f} | "
                f"Volume MSE: {val_metrics['val_mse_volume']:.4f}"
            )
            
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({
                "epoch": epoch+1, 
                "train_loss": train_loss, 
                "val_loss": val_metrics['val_loss'], 
                "val_accuracy_gleason": val_metrics['val_accuracy_gleason'],
                "val_accuracy_grade_group": val_metrics['val_accuracy_grade_group'],
                "val_mse_volume": val_metrics['val_mse_volume'],
                "learning_rate": current_lr
            })

            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                torch.save(model.state_dict(), f"transmil_multitask_HistoEncoder_fold_{fold+1}_best.pth")
                print("Modèle sauvegardé (meilleure performance).")
            else:
                patience_counter += 1
                print(f"Pas d'amélioration. Patience : {patience_counter}/{config['patience']}")

            if patience_counter >= config['patience']:
                print("Early stopping déclenché.")
                break
        
        run.finish()

if __name__ == '__main__':
    main()