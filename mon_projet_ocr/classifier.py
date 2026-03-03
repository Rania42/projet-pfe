import cv2
import pytesseract
import numpy as np
import re
import unicodedata
from rapidfuzz import fuzz
from dictionary import KEYWORDS, DOC_FAMILIES

# CONFIG
FUZZY_THRESHOLD = 75   # 🔥 Légèrement augmenté pour les phrases
MIN_SCORE = 5          # 🔥 Augmenté car les scores sont plus élevés maintenant


# 🔹 1. PREPROCESSING IMAGE AMÉLIORÉ
def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image introuvable")

    # Conversion en gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥 CLAHE pour améliorer le contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 🔥 BILATERAL FILTER (meilleur que fastNlMeansDenoising pour les chèques)
    # Lisse le fond tout en préservant les bords des caractères
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # Seuillage adaptatif
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # 🔥 Optionnel: Augmenter la résolution si l'image est trop petite
    h, w = thresh.shape
    if h < 1000:  # Si l'image est petite
        scale = 1500 / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        thresh = cv2.resize(thresh, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return thresh


# 🔹 2. NORMALISATION TEXTE AMÉLIORÉE
def normalize_text(text):
    # 🔥 Préserver la casse pour les mots-clés comme "IBAN" mais normaliser pour la recherche
    text_lower = text.lower()

    # Supprimer les accents (é → e)
    text_lower = unicodedata.normalize('NFD', text_lower)
    text_lower = ''.join(c for c in text_lower if unicodedata.category(c) != 'Mn')

    # Garder arabe + latin + chiffres
    text_lower = re.sub(r'[^a-z0-9\u0600-\u06FF\s]', ' ', text_lower)
    text_lower = re.sub(r'\s+', ' ', text_lower)

    return text_lower.strip()


# 🔹 3. FAMILY
def get_family(doc_type):
    for family, types in DOC_FAMILIES.items():
        if doc_type in types:
            return family
    return "Inconnue"


# 🔹 4. SCORE AVEC RECHERCHE DE PHRASES COMPLÈTES (CORRECTION MAJEURE)
def compute_score(text, keywords):
    """
    Calcule le score en recherchant des phrases complètes plutôt que des mots isolés.
    Compatible avec la structure du nouveau dictionary.py qui contient:
    - Des phrases composées (ex: "attestation de solde")
    - Des mots-clés simples
    """
    score = 0
    
    # Parcourir tous les mots-clés (peuvent être des phrases ou des mots simples)
    for keyword, weight in keywords.items():
        # 🔥 Ignorer les clés spéciales qui commencent par '_'
        if keyword.startswith('_'):
            continue
            
        # Vérifier si la phrase exacte existe dans le texte
        if keyword in text:
            # Correspondance exacte = poids complet
            score += weight
        else:
            # 🔥 Recherche floue sur la phrase complète (pas mot par mot)
            # partial_ratio cherche si la phrase est présente approximativement
            similarity = fuzz.partial_ratio(keyword, text)
            
            if similarity > FUZZY_THRESHOLD:
                # Score pondéré par la similarité
                score += weight * (similarity / 100)
    
    return score


# 🔹 5. FONCTION DE DEBUG OPTIONNELLE
def print_top_scores(scores, top_n=5):
    """Affiche les top scores pour le débogage"""
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print("\n🔍 Top scores:")
    for doc_type, score in sorted_scores:
        if score > 0:
            print(f"   {doc_type}: {score:.2f}")


# 🔹 6. CLASSIFICATION PRINCIPALE
def classify_image(image_path):
    try:
        # Prétraitement
        processed_img = preprocess_image(image_path)
        
        # OCR
        text = pytesseract.image_to_string(processed_img, lang='fra+ara')
        text_clean = normalize_text(text)
        
        # 🔥 DEBUG: Afficher le texte extrait (utile pour comprendre)
        print("\n📄 Texte extrait (début):", text_clean[:200] + "...")
        
        # Calcul des scores
        scores = {}
        for doc_type, keywords in KEYWORDS.items():
            scores[doc_type] = compute_score(text_clean, keywords)
        
        # Meilleur document
        best_doc = max(scores, key=scores.get)
        best_score = scores[best_doc]
        
        # 🔥 DEBUG: Afficher les top scores
        print_top_scores(scores)
        print(f"\n🏆 Meilleur document: {best_doc} (score: {best_score:.2f})")
        
        # Vérification du score minimum
        if best_score < MIN_SCORE:
            print("⚠️ Score trop faible - document non identifié")
            return "NON_IDENTIFIE", "Inconnue", text
        
        return best_doc, get_family(best_doc), text
        
    except Exception as e:
        print(f"❌ Erreur lors de la classification: {e}")
        return "ERREUR", "Inconnue", ""


# 🔹 7. POUR TEST EN LIGNE DE COMMANDE
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        doc_type, family, text = classify_image(image_path)
        print(f"\n📋 Résultat final:")
        print(f"Type: {doc_type}")
        print(f"Famille: {family}")
        print(f"Texte: {text[:200]}...")
    else:
        print("Usage: python classifier.py <chemin_image>")