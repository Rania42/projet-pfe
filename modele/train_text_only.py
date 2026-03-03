"""
Script pour entraîner uniquement le modèle texte (TF-IDF + SVM)
en utilisant le modèle vision déjà entraîné.
"""

import os
import sys
import pickle
import logging
from pathlib import Path
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Copiez toute la configuration et les fonctions depuis votre script original
# jusqu'à la ligne 338 (avant le main)

# Ajoutez cette fonction spécifique
def load_existing_vision_model():
    """Charge le modèle EfficientNet sauvegardé."""
    model_path = os.path.join(CFG["model_dir"], "efficientnet_best.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle vision introuvable : {model_path}")
    
    model = build_efficientnet(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    log.info(f"✅ Modèle vision chargé : {model_path}")
    return model

def main_text_only():
    log.info("\n" + "╔" + "═" * 53 + "╗")
    log.info("║  Entraînement TEXTE uniquement (TF-IDF + SVM)   ║")
    log.info("╚" + "═" * 53 + "╝\n")

    # ── Charger les données augmentées ──
    log.info("📁 Chargement du dataset augmenté...")
    all_paths = {}
    for cls in CFG["classes"]:
        cls_dir = os.path.join(CFG["output_dir"], "augmented", cls)
        paths = sorted([
            os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
            if f.lower().endswith(EXTENSIONS)
        ])
        all_paths[cls] = paths
        log.info(f"  [{cls}] : {len(paths)} images")

    # ── Préparer les données ──
    paths = []
    labels = []
    for cls_idx, cls in enumerate(CFG["classes"]):
        paths += all_paths[cls]
        labels += [cls_idx] * len(all_paths[cls])

    # ── Split Train/Test ──
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels,
        test_size=0.2,
        stratify=labels,
        random_state=CFG["seed"]
    )
    log.info(f"📊 Split : Train {len(train_paths)} | Test {len(test_paths)}")

    # ── Charger modèle vision (optionnel, pour stats) ──
    vision_model = load_existing_vision_model()
    
    # ── Entraînement TEXTE uniquement ──
    log.info("\n📝 Entraînement du modèle Texte (TF-IDF + SVM)...")
    text_model, train_texts, val_texts = train_text_model(
        train_paths, train_labels,
        test_paths, test_labels
    )
    
    # ── Évaluation texte seul ──
    log.info("\n📊 Évaluation du modèle TEXTE seul...")
    test_preds_text = text_model.predict(val_texts)
    text_acc = accuracy_score(test_labels, test_preds_text)
    log.info(f"✅ Accuracy texte seul : {text_acc:.4f}")
    
    # ── Évaluation ensemble (vision + texte) ──
    log.info("\n🔀 Évaluation de l'ENSEMBLE (vision + texte)...")
    preds, combined_probs = evaluate_ensemble(
        vision_model, text_model, test_paths, test_labels, "TEST"
    )
    
    # ── Sauvegarder les résultats ──
    log.info("\n💾 Sauvegarde des résultats...")
    
    # Sauvegarder les prédictions pour analyse
    results = {
        "test_paths": test_paths,
        "true_labels": [int(l) for l in test_labels],
        "pred_labels": [int(p) for p in preds],
        "probabilities": combined_probs.tolist(),
        "text_accuracy": float(text_acc),
        "ensemble_accuracy": float(accuracy_score(test_labels, preds))
    }
    
    with open(os.path.join(CFG["output_dir"], "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Matrice de confusion
    plot_confusion_matrix(test_labels, preds)
    
    log.info(f"\n✅ Terminé ! Résultats dans : {CFG['output_dir']}/")

if __name__ == "__main__":
    main_text_only()