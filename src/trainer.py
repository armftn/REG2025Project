import torch
from tqdm import tqdm
import wandb
import numpy as np
# On importe les outils pour la précision mixte
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import mean_squared_error, accuracy_score

def train_one_epoch(model, train_loader, criteria, optimizer, device, scaler, scheduler):
    """
    Effectue une époque d'entraînement pour un modèle multi-tâches.
    """
    model.train()
    total_loss = 0
    
    # On récupère les trois fonctions de coût depuis le dictionnaire
    criterion_gleason, criterion_grade_group, criterion_volume = criteria['gleason'], criteria['grade_group'], criteria['volume']

    for batch in tqdm(train_loader, desc="Training"):
        features = batch['features'].to(device)
        # On récupère les trois étiquettes pour ce batch
        gleason_labels = batch['gleason_label'].to(device)
        grade_group_labels = batch['grade_group_label'].to(device)
        volume_labels = batch['tumor_volume_label'].to(device)

        # --- Mixup Augmentation ---
        alpha = 0.4  # Hyperparamètre du Mixup
        lam = np.random.beta(alpha, alpha) # Coefficient de mélange
        
        # On mélange les indices à l'intérieur du batch
        indices_melanges = torch.randperm(features.size(0))
        
        # On crée les features et labels "mixés"
        mixed_features = lam * features + (1 - lam) * features[indices_melanges, :]
        gleason_labels_a, gleason_labels_b = gleason_labels, gleason_labels[indices_melanges]
        gg_labels_a, gg_labels_b = grade_group_labels, grade_group_labels[indices_melanges]
        volume_labels_a, volume_labels_b = volume_labels, volume_labels[indices_melanges]

        with autocast():
            # Le modèle retourne maintenant un dictionnaire de prédictions
            outputs = model(mixed_features)
            
            # La loss est maintenant une interpolation des loss des deux labels originaux
            loss_gleason = lam * criterion_gleason(outputs['gleason'], gleason_labels_a) + (1 - lam) * criterion_gleason(outputs['gleason'], gleason_labels_b)
            loss_gg = lam * criterion_grade_group(outputs['grade_group'], gg_labels_a) + (1 - lam) * criterion_grade_group(outputs['grade_group'], gg_labels_b)
            loss_volume = lam * criterion_volume(outputs['tumor_volume'], volume_labels_a) + (1 - lam) * criterion_volume(outputs['tumor_volume'], volume_labels_b)

            lw = criteria['weights'] # On récupère les poids (on les passera depuis train.py)

            # On pondère chaque loss avant de les sommer
            combined_loss = (lw['gleason'] * loss_gleason + 
                        lw['grade_group'] * loss_gg + 
                        lw['volume'] * loss_volume)

        optimizer.zero_grad()
        scaler.scale(combined_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += combined_loss.item()
    
    scheduler.step()
    return total_loss / len(train_loader)

def validate_one_epoch(model, val_loader, criteria, device):
    """
    Effectue une époque de validation pour un modèle multi-tâches.
    """
    model.eval()
    total_loss = 0
    # Listes pour stocker toutes les prédictions et étiquettes de l'époque
    all_preds_gleason, all_labels_gleason = [], []
    all_preds_grade_group, all_labels_grade_group = [], []
    all_preds_volume, all_labels_volume = [], []

    criterion_gleason, criterion_grade_group, criterion_volume = criteria['gleason'], criteria['grade_group'], criteria['volume']

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            features = batch['features'].to(device)
            gleason_labels = batch['gleason_label'].to(device)
            grade_group_labels = batch['grade_group_label'].to(device)
            volume_labels = batch['tumor_volume_label'].to(device)
            
            with autocast():
                outputs = model(features)
                loss_gleason = criterion_gleason(outputs['gleason'], gleason_labels)
                loss_grade_group = criterion_grade_group(outputs['grade_group'], grade_group_labels)
                loss_volume = criterion_volume(outputs['tumor_volume'], volume_labels)
                combined_loss = loss_gleason + loss_grade_group + loss_volume

            total_loss += combined_loss.item()
            
            # --- MODIFICATION : Récupérer les prédictions pour chaque tâche ---
            preds_gleason = torch.argmax(outputs['gleason'], dim=1)
            all_preds_gleason.extend(preds_gleason.cpu().numpy())
            all_labels_gleason.extend(gleason_labels.cpu().numpy())

            preds_grade_group = torch.argmax(outputs['grade_group'], dim=1)
            all_preds_grade_group.extend(preds_grade_group.cpu().numpy())
            all_labels_grade_group.extend(grade_group_labels.cpu().numpy())

            all_preds_volume.extend(outputs['tumor_volume'].cpu().numpy())
            all_labels_volume.extend(volume_labels.cpu().numpy())
            
    # --- MODIFICATION : Calculer les métriques pour chaque tâche ---
    metrics = {
        'val_loss': total_loss / len(val_loader),
        'val_accuracy_gleason': accuracy_score(all_labels_gleason, all_preds_gleason),
        'val_accuracy_grade_group': accuracy_score(all_labels_grade_group, all_preds_grade_group),
        # Pour la régression, on utilise l'erreur quadratique moyenne (MSE)
        'val_mse_volume': mean_squared_error(all_labels_volume, all_preds_volume)
    }
    return metrics