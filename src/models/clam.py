# Fichier : src/models/clam.py
import torch
import torch.nn as nn

class GatedAttention(nn.Module):
    """Couche d'attention avec un mécanisme de porte (gating)."""
    def __init__(self, input_dim):
        super(GatedAttention, self).__init__()
        self.attention_V = nn.Linear(input_dim, 128)
        self.attention_U = nn.Linear(input_dim, 128)
        self.attention_weights = nn.Linear(128, 1)

    def forward(self, x):
        A_V = torch.tanh(self.attention_V(x))
        A_U = torch.sigmoid(self.attention_U(x))
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 2, 1)
        A = nn.functional.softmax(A, dim=1)
        return A

class CLAM(nn.Module):
    """
    Architecture CLAM (Clustering-constrained Attention Multiple Instance Learning).
    Version simplifiée et adaptée pour une approche multi-tâches.
    """
    def __init__(self, input_dim=512, n_gleason_classes=9, n_gg_classes=5, dropout=0.25):
        super(CLAM, self).__init__()
        self.input_dim = input_dim
        
        # Réseau partagé pour réduire la dimension des features des patchs
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # --- Tête pour le Score de Gleason (Classification) ---
        # On crée une branche d'attention par classe de Gleason
        self.attention_gleason = GatedAttention(256)
        self.classifier_gleason = nn.Linear(256, n_gleason_classes)

        # --- Tête pour le Grade Group (Classification) ---
        # On crée une branche d'attention par classe de Grade Group
        self.attention_gg = GatedAttention(256)
        self.classifier_gg = nn.Linear(256, n_gg_classes)
        
        # --- Tête pour le Volume Tumoral (Régression) ---
        # Pour la régression, une attention simple suffit
        self.attention_volume = GatedAttention(256)
        self.regressor_volume = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid() # Pour borner la sortie entre 0 et 1
        )

    def forward(self, x):
        # x.shape: [batch_size, n_patches, input_dim]
        
        # On passe tous les patchs dans l'extracteur de features
        H = self.feature_extractor(x) # H.shape: [batch_size, n_patches, 256]

        # --- Calcul pour la Tête Gleason ---
        A_gleason = self.attention_gleason(H) # A_gleason.shape: [batch_size, 1, n_patches]
        M_gleason = torch.bmm(A_gleason, H) # M_gleason.shape: [batch_size, 1, 256]
        M_gleason = M_gleason.squeeze(1) # [batch_size, 256]
        gleason_logits = self.classifier_gleason(M_gleason)

        # --- Calcul pour la Tête Grade Group ---
        A_gg = self.attention_gg(H)
        M_gg = torch.bmm(A_gg, H)
        M_gg = M_gg.squeeze(1)
        grade_group_logits = self.classifier_gg(M_gg)
        
        # --- Calcul pour la Tête Volume ---
        A_volume = self.attention_volume(H)
        M_volume = torch.bmm(A_volume, H)
        M_volume = M_volume.squeeze(1)
        tumor_volume_pred = self.regressor_volume(M_volume)

        return {
            'gleason': gleason_logits,
            'grade_group': grade_group_logits,
            'tumor_volume': tumor_volume_pred.squeeze(1)
        }