"""
Testeur simple pour une image unique
"""

import os
import sys
import json
import pickle
import torch
from PIL import Image
import numpy as np

# Copiez ici toutes les fonctions nécessaires depuis le script original :
# - preprocess_image
# - extract_text_from_image
# - EVAL_TRANSFORMS
# - build_efficientnet
# - get_vision_probs
# - get_text_probs
# - ensemble_predict

def load_models():
    """Charge les deux modèles entraînés."""
    # Charger config
    with open(os.path.join(CFG["output_dir"], "inference_config.json"), "r") as f:
        config = json.load(f)
    
    # Charger modèle vision
    vision_model = build_efficientnet(num_classes=2).to(DEVICE)
    vision_model.load_state_dict(
        torch.load(os.path.join(CFG["model_dir"], "efficientnet_best.pth"), 
                  map_location=DEVICE)
    )
    vision_model.eval()
    
    # Charger modèle texte
    with open(os.path.join(CFG["model_dir"], "text_pipeline.pkl"), "rb") as f:
        text_model = pickle.load(f)
    
    return vision_model, text_model, config

def predict_image(image_path, vision_model, text_model, config):
    """Prédit la classe d'une image."""
    # Charger image
    img = Image.open(image_path).convert("RGB")
    
    # Prédiction vision
    vision_probs = get_vision_probs(vision_model, [image_path])[0]
    
    # Prédiction texte (OCR)
    text_probs = get_text_probs(text_model, [image_path])[0]
    
    # Ensemble
    combined = (
        config["vision_weight"] * vision_probs +
        config["text_weight"] * text_probs
    )
    
    # Décision avec seuil
    max_prob = np.max(combined)
    pred_class = config["classes"][np.argmax(combined)]
    
    if max_prob < config["confidence_threshold"]:
        pred_class = config["other_label"]
    
    return {
        "prediction": pred_class,
        "confidence": float(max_prob),
        "vision_probs": {config["classes"][i]: float(vision_probs[i]) 
                        for i in range(2)},
        "text_probs": {config["classes"][i]: float(text_probs[i]) 
                      for i in range(2)},
        "ensemble_probs": {config["classes"][i]: float(combined[i]) 
                          for i in range(2)}
    }

def main():
    # Charger les modèles
    print("Chargement des modèles...")
    vision_model, text_model, config = load_models()
    
    # Tester sur une image
    image_path = input("Entrez le chemin de l'image à tester : ").strip()
    
    if not os.path.exists(image_path):
        print(f"❌ Fichier introuvable : {image_path}")
        return
    
    print(f"\n🔍 Analyse de : {image_path}")
    result = predict_image(image_path, vision_model, text_model, config)
    
    print("\n" + "="*50)
    print(f"📌 PRÉDICTION : {result['prediction']}")
    print(f"🎯 Confiance : {result['confidence']:.2%}")
    print("="*50)
    print("\nDétails par modèle :")
    for model in ["vision", "text"]:
        probs = result[f"{model}_probs"]
        print(f"\n{model.upper()}:")
        for cls, prob in probs.items():
            print(f"  {cls}: {prob:.2%}")
    print("="*50)

if __name__ == "__main__":
    main()