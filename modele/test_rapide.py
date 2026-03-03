import torch
from test import DetecteurChequeResNet
import os

# 1. Charger le meilleur modèle (sauvegardé automatiquement)
detecteur = DetecteurChequeResNet("meilleur_modele_resnet.pth")

# 2. Liste de vos images à tester
images_test = [
    "C:/Users/lenovo/Desktop/z.png",
    "C:/Users/lenovo/Desktop/8.png", 
    "C:/Users/lenovo/Desktop/3.png",
    "C:/Users/lenovo/Desktop/9.jpg"
]

# 3. Tester chaque image
print("\n" + "="*60)
print("🧪 TEST DU MODÈLE SUR VOS IMAGES")
print("="*60)

for img in images_test:
    if os.path.exists(img):
        detecteur.predire(img)
    else:
        print(f"\n❌ Fichier non trouvé: {img}")

# 4. Mode interactif pour tester d'autres images
print("\n" + "="*60)
print("⌨️ MODE INTERACTIF")
print("="*60)

while True:
    chemin = input("\n📝 Chemin de l'image (ou 'q' pour quitter): ").strip()
    if chemin.lower() == 'q':
        break
    if not os.path.exists(chemin):
        print("❌ Fichier non trouvé")
        continue
    detecteur.predire(chemin)