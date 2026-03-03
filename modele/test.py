import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Utilisation de: {device}")

class DatasetCheque(Dataset):
    """
    Dataset avec prétraitement adapté aux scans de documents
    """
    def __init__(self, chemins, labels, transform=None, augment=True):
        self.chemins = chemins
        self.labels = labels
        self.augment = augment
        
        # Transformations de base
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentations pour les positifs (chèques)
        self.augmentations = transforms.Compose([
            transforms.RandomRotation(degrees=3),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),
            transforms.RandomPerspective(distortion_scale=0.05, p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ])
    
    def pretraiter_scan(self, chemin):
        """Prétraitement spécifique pour scans bruités"""
        # Lire l'image
        img = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # 1. Débruitage
        img = cv2.medianBlur(img, 3)
        
        # 2. Normalisation du contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        # 3. Binarisation adaptative (optionnel)
        # img = cv2.adaptiveThreshold(img, 255, 
        #                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                            cv2.THRESH_BINARY, 11, 2)
        
        return Image.fromarray(img)
    
    def __len__(self):
        return len(self.chemins)
    
    def __getitem__(self, idx):
        # Charger et prétraiter l'image
        img_pil = self.pretraiter_scan(self.chemins[idx])
        if img_pil is None:
            # En cas d'erreur, prendre une autre image
            return self.__getitem__((idx + 1) % len(self.chemins))
        
        # Appliquer les augmentations seulement sur les positifs
        if self.augment and self.labels[idx] == 1 and random.random() > 0.5:
            img_pil = self.augmentations(img_pil)
        
        # Transformer pour ResNet
        img_tensor = self.transform(img_pil)
        
        return img_tensor, torch.tensor(self.labels[idx], dtype=torch.float32)

class DetecteurChequeResNet:
    def __init__(self, chemin_modele=None):
        self.device = device
        
        # Charger ResNet50 pré-entraîné
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        
        # Modifier la dernière couche pour classification binaire
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.model = self.model.to(self.device)
        
        # Charger modèle existant
        if chemin_modele and os.path.exists(chemin_modele):
            self.model.load_state_dict(torch.load(chemin_modele, map_location=self.device))
            print(f"✅ Modèle chargé: {chemin_modele}")
        
        # Loss et métriques
        self.criterion = nn.BCELoss()
        self.optimizer = None
    
    def preparer_dataset(self, dossier_positifs, dossier_negatifs, val_split=0.2):
        """
        Prépare les datasets à partir des dossiers
        """
        print("\n📂 Préparation du dataset...")
        
        # Charger les positifs (chèques)
        chemins_pos = []
        for dossier in dossier_positifs:
            if os.path.exists(dossier):
                chemins_pos.extend(glob.glob(os.path.join(dossier, "*.png")))
                chemins_pos.extend(glob.glob(os.path.join(dossier, "*.jpg")))
        
        # Charger les négatifs (non-chèques)
        chemins_neg = []
        if os.path.exists(dossier_negatifs):
            for root, dirs, files in os.walk(dossier_negatifs):
                for file in files:
                    if file.endswith(('.png', '.jpg')):
                        chemins_neg.append(os.path.join(root, file))
        
        print(f"   ✅ {len(chemins_pos)} images de chèques (positifs)")
        print(f"   ✅ {len(chemins_neg)} images autres documents (négatifs)")
        
        if len(chemins_neg) < 50:
            print("⚠️ ATTENTION: Peu de négatifs, risque de mauvaises performances!")
        
        # Équilibrer les classes
        nb_min = min(len(chemins_pos), len(chemins_neg))
        chemins_pos = random.sample(chemins_pos, nb_min)
        chemins_neg = random.sample(chemins_neg, nb_min)
        
        # Créer les labels
        tous_chemins = chemins_pos + chemins_neg
        tous_labels = [1] * len(chemins_pos) + [0] * len(chemins_neg)
        
        # Split train/val
        train_chemins, val_chemins, train_labels, val_labels = train_test_split(
            tous_chemins, tous_labels, 
            test_size=val_split, 
            random_state=42,
            stratify=tous_labels
        )
        
        print(f"\n📊 Dataset final:")
        print(f"   - Train: {len(train_chemins)} images")
        print(f"   - Validation: {len(val_chemins)} images")
        
        # Datasets
        train_dataset = DatasetCheque(train_chemins, train_labels, augment=True)
        val_dataset = DatasetCheque(val_chemins, val_labels, augment=False)
        
        return train_dataset, val_dataset
    
    def entrainer(self, dossier_positifs, dossier_negatifs, epochs=20, batch_size=16, lr=1e-4):
        """
        Entraîne le modèle
        """
        print("\n" + "="*60)
        print("🏋️ ENTRAÎNEMENT RESNET50")
        print("="*60)
        
        # Préparer les données
        train_dataset, val_dataset = self.preparer_dataset(
            dossier_positifs, dossier_negatifs
        )
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimiseur
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Entraînement
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward
                self.optimizer.zero_grad()
                outputs = self.model(images).squeeze()
                loss = self.criterion(outputs, labels)
                
                # Backward
                loss.backward()
                self.optimizer.step()
                
                # Métriques
                train_loss += loss.item()
                preds = (outputs > 0.5).float()
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
                
                pbar.set_postfix({'loss': loss.item()})
            
            train_acc = train_correct / train_total
            
            # Validation
            self.model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images).squeeze()
                    loss = self.criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    preds = (outputs > 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = val_correct / val_total
            scheduler.step(val_loss)
            
            # Affichage
            print(f"\n   Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2%}")
            print(f"   Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2%}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), "meilleur_modele_resnet.pth")
                print(f"   💾 Meilleur modèle sauvegardé (val_acc: {val_acc:.2%})")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 7:
                    print("⏹️ Early stopping")
                    break
        
        print(f"\n✅ Entraînement terminé!")
        print(f"   Meilleure accuracy validation: {best_val_acc:.2%}")
        
        # Charger le meilleur modèle
        self.model.load_state_dict(torch.load("meilleur_modele_resnet.pth"))
    
    def predire(self, chemin_image, seuil=0.5):
        """
        Prédit si une image est un chèque
        """
        # Prétraiter l'image
        img_pil = self.pretraiter_image(chemin_image)
        if img_pil is None:
            return False, 0, "Erreur chargement"
        
        # Transformer
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img_pil).unsqueeze(0).to(self.device)
        
        # Prédire
        self.model.eval()
        with torch.no_grad():
            score = self.model(img_tensor).item()
        
        est_cheque = score > seuil
        
        # Résultat
        print("\n" + "="*50)
        print(f"📄 Fichier: {os.path.basename(chemin_image)}")
        print(f"   Score: {score:.2%}")
        print(f"   Seuil: {seuil:.0%}")
        
        if score > 0.8:
            confiance = "haute"
        elif score > 0.6:
            confiance = "bonne"
        elif score > 0.4:
            confiance = "moyenne"
        else:
            confiance = "faible"
        
        if est_cheque:
            print(f"   ✅ CHÈQUE (confiance {confiance})")
        else:
            print(f"   ❌ PAS CHÈQUE (confiance {confiance})")
        print("="*50)
        
        return est_cheque, score
    
    def pretraiter_image(self, chemin):
        """Prétraitement pour une seule image"""
        img = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        img = cv2.medianBlur(img, 3)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        return Image.fromarray(img)

# ============================================================
# MAIN - Exécution principale
# ============================================================

if __name__ == "__main__":
    # 1. Initialiser le détecteur
    detecteur = DetecteurChequeResNet()
    
    # 2. Définir les dossiers
    dossier_positifs = [
        "../output/images",           # Vos chèques originaux
        "../output/images_augmentees" # Vos chèques augmentés
    ]
    
    dossier_negatifs = "../output/negatifs"  # À créer avec vos négatifs
    
    # 3. Vérifier que les dossiers existent
    if not os.path.exists(dossier_negatifs):
        print(f"\n❌ Créez d'abord le dossier: {dossier_negatifs}")
        print("   Mettez-y des images de factures, contrats, etc.")
        exit()
    
    # 4. Entraîner le modèle
    detecteur.entrainer(
        dossier_positifs=dossier_positifs,
        dossier_negatifs=dossier_negatifs,
        epochs=25,
        batch_size=16,
        lr=1e-4
    )
    
    # 5. Tester sur des images
    print("\n" + "="*60)
    print("🧪 TEST SUR IMAGES")
    print("="*60)
    
    images_test = [
        "C:/Users/lenovo/Desktop/z.png",
        "C:/Users/lenovo/Desktop/8.png",
        "C:/Users/lenovo/Desktop/3.png",
        "C:/Users/lenovo/Desktop/9.jpg"
    ]
    
    for img in images_test:
        if os.path.exists(img):
            detecteur.predire(img)
    
    # 6. Mode interactif
    print("\n" + "="*60)
    print("⌨️ MODE INTERACTIF - Entrez 'q' pour quitter")
    print("="*60)
    
    while True:
        chemin = input("\n📝 Chemin de l'image à tester: ").strip()
        if chemin.lower() == 'q':
            break
        if not os.path.exists(chemin):
            print("❌ Fichier non trouvé")
            continue
        detecteur.predire(chemin)