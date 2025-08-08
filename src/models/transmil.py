import torch
import torch.nn as nn

class TransMIL(nn.Module):
    """
    Architecture du modèle TransMIL.
    Ce modèle utilise un Transformer pour analyser les relations entre les patchs.
    """
    def __init__(self, input_dim=1536, n_classes=9, hidden_dim=512, n_heads=8, n_layers=4, dropout=0.25):
        super(TransMIL, self).__init__()
        
        # 1. Couche d'embedding : Projette les features des patchs dans un espace de travail.
        self.patch_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        # 2. Token de Classe [CLS] : Un "jeton" spécial et apprenable qui va agréger l'information de tous les patchs.
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # 3. Encodeur Transformer : Le cœur du modèle, qui traite les patchs.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout,
            batch_first=True  # Important pour la forme de nos tenseurs
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- MODIFICATION : Remplacer le classifieur unique par 3 têtes distinctes ---
        
        # Tête 1 : Pour le score de Gleason (Classification, 9 classes)
        self.gleason_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 9)
        )
        
        # Tête 2 : Pour le Grade Group (Classification, 5 classes)
        self.grade_group_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 5)
        )

        # Tête 3 : Pour le Volume de la Tumeur (Régression, 1 sortie)
        self.tumor_volume_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Passe avant dans le modèle.
        Args:
            x (torch.Tensor): Un batch de features de patchs. Shape: [batch_size, n_patches, input_dim]
        """
        # Applique l'embedding à chaque patch
        x = self.patch_embedding(x)  # Shape: [batch_size, n_patches, hidden_dim]

        # Prépare et ajoute le token [CLS] au début de la séquence de chaque lame
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # Shape: [batch_size, 1, hidden_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: [batch_size, n_patches + 1, hidden_dim]

        # Passe les données dans le Transformer
        transformer_output = self.transformer_encoder(x)  # Shape: [batch_size, n_patches + 1, hidden_dim]

        # On ne garde que la sortie du token [CLS] (le premier token) qui contient l'information agrégée
        cls_output = transformer_output[:, 0, :]  # Shape: [batch_size, hidden_dim]

        # --- MODIFICATION : On calcule les 3 sorties ---
        # La sortie du Transformer (`cls_output`) est partagée et envoyée à chaque tête.
        gleason_logits = self.gleason_head(cls_output)
        grade_group_logits = self.grade_group_head(cls_output)
        tumor_volume_pred = self.tumor_volume_head(cls_output)
        
        # On retourne un dictionnaire de prédictions
        return {
            'gleason': gleason_logits,
            'grade_group': grade_group_logits,
            'tumor_volume': tumor_volume_pred.squeeze(1) # On enlève la dimension superflue
        }