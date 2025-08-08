import re
import yaml # Librairie pour lire les fichiers .yaml

def load_config(config_path="configs/base_config.yaml"):
    """Charge la configuration depuis un fichier YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_all_labels(report_text: str) -> dict:
    """
    Analyse le texte d'un rapport de prostate pour extraire le score de Gleason,
    le Grade Group, et le volume de la tumeur.
    Retourne un dictionnaire avec les valeurs, ou des valeurs par défaut si non trouvées.
    """
    # Dictionnaire pour mapper le score de Gleason à une classe entière (0-8)
    gleason_map = {
        "3+3": 0, "3+4": 1, "3+5": 2,
        "4+3": 3, "4+4": 4, "4+5": 5,
        "5+3": 6, "5+4": 7, "5+5": 8
    }
    
    # Initialisation des valeurs par défaut
    labels = {
        "gleason_score": -1,
        "grade_group": -1,
        "tumor_volume": -1.0
    }

    # 1. Extraction du score de Gleason
    gleason_match = re.search(r"Gleason's score \d\s*\((\d\+\d)\)", report_text)
    if gleason_match:
        score_str = gleason_match.group(1)
        labels["gleason_score"] = gleason_map.get(score_str, -1)

    # 2. Extraction du Grade Group
    grade_group_match = re.search(r"grade group (\d)", report_text)
    if grade_group_match:
        # Les Grade Groups vont de 1 à 5. On les mappe de 0 à 4 pour la classification.
        labels["grade_group"] = int(grade_group_match.group(1)) - 1
    
    # 3. Extraction du Volume de la Tumeur
    tumor_volume_match = re.search(r"tumor volume: (\d+)%", report_text)
    if tumor_volume_match:
        # On normalise le volume entre 0.0 et 1.0 pour la régression.
        labels["tumor_volume"] = float(tumor_volume_match.group(1)) / 100.0
        
    return labels