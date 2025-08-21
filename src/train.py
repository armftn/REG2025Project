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
from src.models.clam import CLAM
from src.utils import load_config, extract_all_labels
from src.trainer import train_one_epoch, validate_one_epoch

def main():
    # --- Étape 1 : INITIALISATION DE WANDB POUR LE SWEEP ---
    run = wandb.init() 
    # `wandb.config` contient les paramètres du sweep.
    # On charge base_config pour les paramètres fixes (chemins, epochs).
    base_config = load_config()
    config = wandb.config
    
    print("--- Démarrage d'une nouvelle run du Sweep ---")
    print("Configuration pour cette run :", dict(config))

    # --- Étape 2 : PRÉPARATION ET NETTOYAGE DES DONNÉES ---
    full_df = pd.read_json(base_config['data_paths']['json_path'])
    features_path = os.path.join(base_config['data_paths']['features_path'], "train")
    
    prostate_df = full_df[full_df['report'].str.contains("Prostate, biopsy", case=False, na=False)].reset_index(drop=True)
    expected_filenames = prostate_df['id'].apply(lambda x: f"{x.replace('.tiff', '')}.parquet")
    file_exists_mask = expected_filenames.apply(lambda x: os.path.exists(os.path.join(features_path, x)))
    cleaned_df = prostate_df[file_exists_mask].reset_index(drop=True)
    
    labels_series = cleaned_df['report'].apply(extract_all_labels)
    labels_df = pd.json_normalize(labels_series.tolist())
    cleaned_df = cleaned_df.join(labels_df)

    final_df = cleaned_df[(cleaned_df['gleason_score'] != -1) & (cleaned_df['grade_group'] != -1)].reset_index(drop=True)
    valid_labels = final_df['gleason_score'].tolist()
    
    if not valid_labels:
        print("ERREUR : Aucune donnée valide pour l'entraînement.")
        return
    print(f"{len(final_df)} échantillons de prostate valides trouvés.")

    # --- NOUVELLE ÉTAPE : PRÉ-CHARGEMENT DES FEATURES .PARQUET ---
    print("Pré-chargement des features .parquet en RAM (cela peut prendre un moment)...")
    preloaded_features = []
    for slide_id in tqdm(final_df['id']):
        filename = f"{slide_id.replace('.tiff', '')}.parquet"
        filepath = os.path.join(features_path, filename)
        patch_df = pd.read_parquet(filepath)
        feature_columns = [f'feat{i}' for i in range(1, 513)]
        preloaded_features.append(patch_df[feature_columns].values)
    print(f"{len(preloaded_features)} ensembles de features ont été chargés en RAM.")

    # --- Étape 3 : CRÉATION DU DATASET PYTORCH ---
    full_dataset = WsiReportDataset(data_df=final_df, features_list=preloaded_features)
    
    # --- Étape 4 : MISE EN PLACE DE LA VALIDATION CROISÉE ---
    skf = StratifiedKFold(n_splits=base_config['n_splits'], shuffle=True, random_state=42)
    X_indices = np.arange(len(full_dataset))

    # --- Étape 5 : BOUCLE PRINCIPALE SUR LES PLIS (FOLDS) ---
    fold_accuracies = [] 
    for fold, (train_indices, val_indices) in enumerate(skf.split(X=X_indices, y=valid_labels)):
        print(f"\n=============== DÉMARRAGE DU FOLD {fold+1}/{base_config['n_splits']} ================")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        
        train_labels_for_fold = [valid_labels[i] for i in train_indices]
        class_counts = np.bincount(train_labels_for_fold, minlength=base_config['model_params']['n_classes'])
        weights_per_class = 1.0 / (class_counts + 1e-6)
        sample_weights = np.array([weights_per_class[label] for label in train_labels_for_fold])
        sampler = WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_subset, batch_size=base_config['batch_size'], sampler=sampler, num_workers=base_config['num_workers'], pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=base_config['batch_size'], shuffle=False, num_workers=base_config['num_workers'], pin_memory=True)
        
        # --- MODIFICATION : Initialisation dynamique du modèle ---
        if config.model_name == 'transmil':
            model = TransMIL(
                input_dim=base_config['model_params']['input_dim'],
                n_classes=base_config['model_params']['n_classes'],
                hidden_dim=config.hidden_dim,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                dropout=config.dropout
            ).to(device)
        elif config.model_name == 'clam':
            model = CLAM(
                input_dim=base_config['clam_params']['input_dim'],
                n_gleason_classes=base_config['clam_params']['n_gleason_classes'],
                n_gg_classes=base_config['clam_params']['n_gg_classes'],
                dropout=config.dropout # Le dropout vient du sweep
            ).to(device)
        else:
            raise ValueError(f"Modèle non supporté : {config.model_name}")
        
        # --- CORRECTION : On ajoute la clé 'weights' au dictionnaire 'criteria' ---
        criteria = {
            'gleason': torch.nn.CrossEntropyLoss(),
            'grade_group': torch.nn.CrossEntropyLoss(),
            'volume': torch.nn.MSELoss(),
            'weights': base_config.get('loss_weights', {'gleason': 1.0, 'grade_group': 1.0, 'volume': 1.0})
        }
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=base_config['epochs'], eta_min=1e-7)
        scaler = GradScaler()
        best_val_loss = float('inf')
        best_val_acc_for_fold = 0
        patience_counter = 0

        for epoch in range(base_config['epochs']):
            train_loss = train_one_epoch(model, train_loader, criteria, optimizer, device, scaler, scheduler)
            val_metrics = validate_one_epoch(model, val_loader, criteria, device)
            
            # ... (le reste du logging et de la boucle ne change pas)
            wandb.log({
                "train_loss": train_loss, 
                "val_loss": val_metrics['val_loss'], 
                "val_accuracy_gleason": val_metrics['val_accuracy_gleason'],
                "val_accuracy_grade_group": val_metrics['val_accuracy_grade_group'],
                "val_mse_volume": val_metrics['val_mse_volume'],
                "learning_rate": optimizer.param_groups[0]['lr']
            })

            if val_metrics['val_accuracy_grade_group'] > best_val_acc_for_fold:
                best_val_acc_for_fold = val_metrics['val_accuracy_grade_group']
            
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= base_config['patience']:
                print("Early stopping déclenché.")
                break
        
        fold_accuracies.append(best_val_acc_for_fold)
        print(f"Meilleure accuracy (Grade Group) pour le Fold {fold+1}: {best_val_acc_for_fold:.4f}")

    avg_accuracy = np.mean(fold_accuracies)
    print(f"\nPrécision moyenne (Grade Group) sur 5 plis : {avg_accuracy:.4f}")
    
    wandb.log({"avg_val_accuracy": avg_accuracy})

if __name__ == '__main__':
    main()