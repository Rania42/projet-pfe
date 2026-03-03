# ============================================================
# DÉTECTION DE CHÈQUES AVEC LAYOUTLMv3 - VERSION ADAPTÉE À VOTRE STRUCTURE
# ============================================================

# 1. INSTALLATION DES DÉPENDANCES
!apt-get update
!apt-get install -y tesseract-ocr tesseract-ocr-fra unrar
!pip install pytesseract transformers torch torchvision pillow opencv-python scikit-learn tqdm patool

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from PIL import Image
import cv2
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import pytesseract
from pytesseract import Output
import warnings
import time
from google.colab import files
from google.colab.patches import cv2_imshow
import zipfile
import patoolib  # Pour extraire .rar

warnings.filterwarnings('ignore')

# Configuration GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Utilisation de: {device}")

# ============================================================
# TÉLÉCHARGEMENT DES DONNÉES - ADAPTÉ À VOTRE STRUCTURE
# ============================================================

def upload_dataset():
    """Upload votre fichier output.rar"""
    print("\n📤 Upload de votre fichier output.rar...")
    print("Le fichier doit contenir:")
    print("  📁 images/         (vos chèques)")
    print("  📁 images_augmentees/ (chèques augmentés)")
    print("  📁 negatifs/       (non-chèques)")
    
    uploaded = files.upload()
    
    for filename in uploaded.keys():
        print(f"Fichier uploadé: {filename}")
        
        # Extraire selon l'extension
        if filename.endswith('.zip'):
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall('./')
            print(f"✅ ZIP décompressé")
        elif filename.endswith('.rar'):
            # Extraire le .rar
            patoolib.extract_archive(filename, outdir='./')
            print(f"✅ RAR décompressé")
    
    # Vérifier la structure
    print("\n🔍 Vérification de la structure:")
    
    dossiers_trouves = []
    
    if os.path.exists('./output'):
        print("✅ Dossier 'output' trouvé")
        base_path = './output'
    else:
        # Chercher si les dossiers sont à la racine
        base_path = '.'
    
    # Vérifier chaque dossier
    chemins_positifs = []
    
    # Dossier images
    if os.path.exists(os.path.join(base_path, 'images')):
        print("✅ Dossier 'images' trouvé")
        chemins_positifs.append(os.path.join(base_path, 'images'))
    else:
        print("❌ Dossier 'images' non trouvé")
    
    # Dossier images_augmentees
    if os.path.exists(os.path.join(base_path, 'images_augmentees')):
        print("✅ Dossier 'images_augmentees' trouvé")
        chemins_positifs.append(os.path.join(base_path, 'images_augmentees'))
    else:
        print("⚠️ Dossier 'images_augmentees' non trouvé (optionnel)")
    
    # Dossier negatifs
    if os.path.exists(os.path.join(base_path, 'negatifs')):
        print("✅ Dossier 'negatifs' trouvé")
        dossier_negatifs = os.path.join(base_path, 'negatifs')
    else:
        print("❌ Dossier 'negatifs' non trouvé")
        dossier_negatifs = None
    
    # Afficher le contenu
    print("\n📂 Contenu trouvé:")
    for dossier in chemins_positifs:
        fichiers = glob.glob(os.path.join(dossier, "*.png")) + glob.glob(os.path.join(dossier, "*.jpg"))
        print(f"   {os.path.basename(dossier)}: {len(fichiers)} images")
    
    if dossier_negatifs:
        fichiers_neg = glob.glob(os.path.join(dossier_negatifs, "*.png")) + glob.glob(os.path.join(dossier_negatifs, "*.jpg"))
        print(f"   negatifs: {len(fichiers_neg)} images")
    
    return chemins_positifs, dossier_negatifs

def use_drive():
    """Utiliser Google Drive"""
    from google.colab import drive
    drive.mount('/content/drive')
    
    # À MODIFIER selon l'emplacement de votre dossier dans Drive
    chemin_drive = input("Chemin complet vers votre dossier output dans Drive (ex: /content/drive/MyDrive/output): ").strip()
    
    chemins_positifs = [
        os.path.join(chemin_drive, 'images'),
        os.path.join(chemin_drive, 'images_augmentees')
    ]
    dossier_negatifs = os.path.join(chemin_drive, 'negatifs')
    
    # Vérification
    for path in chemins_positifs:
        if os.path.exists(path):
            print(f"✅ {path} trouvé")
        else:
            print(f"❌ {path} non trouvé")
    
    if os.path.exists(dossier_negatifs):
        print(f"✅ {dossier_negatifs} trouvé")
    else:
        print(f"❌ {dossier_negatifs} non trouvé")
    
    return chemins_positifs, dossier_negatifs

# ============================================================
# DATASET (identique mais avec vos chemins)
# ============================================================

class DatasetChequeLayoutLM(Dataset):
    def __init__(self, chemins, labels, processor, augment=True):
        self.chemins = chemins
        self.labels = labels
        self.processor = processor
        self.augment = augment
        self.ocr_data = {}
        
        print(f"\n🔍 Pré-calcul OCR pour {len(chemins)} images...")
        self._precompute_all_ocr()
    
    def _precompute_all_ocr(self):
        """Calculer l'OCR une fois pour toutes les images"""
        for i, chemin in enumerate(tqdm(self.chemins, desc="OCR Progress")):
            img = cv2.imread(chemin)
            if img is None:
                self.ocr_data[chemin] = (["[PAD]"], [[0, 0, 0, 0]])
                continue
            
            # Redimensionner pour accélérer
            h, w = img.shape[:2]
            if max(h, w) > 1200:
                scale = 1200 / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h))
            
            # Convertir en RGB
            if len(img.shape) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # OCR
            pil_image = Image.fromarray(img_rgb)
            try:
                data = pytesseract.image_to_data(
                    pil_image, 
                    output_type=Output.DICT,
                    config='--psm 6 --oem 1 -l fra+eng'
                )
                
                words = []
                boxes = []
                h_img, w_img = img_rgb.shape[:2]
                
                for j in range(len(data['text'])):
                    conf = int(data['conf'][j])
                    text = data['text'][j].strip()
                    
                    if conf > 20 and len(text) >= 2:
                        x, y, w_box, h_box = data['left'][j], data['top'][j], data['width'][j], data['height'][j]
                        
                        words.append(text)
                        boxes.append([
                            int(x / w_img * 1000),
                            int(y / h_img * 1000),
                            int((x + w_box) / w_img * 1000),
                            int((y + h_box) / h_img * 1000)
                        ])
                
                if not words:
                    words = ["[PAD]"]
                    boxes = [[0, 0, 0, 0]]
                
                self.ocr_data[chemin] = (words, boxes)
                
            except Exception as e:
                self.ocr_data[chemin] = (["[PAD]"], [[0, 0, 0, 0]])
    
    def __len__(self):
        return len(self.chemins)
    
    def __getitem__(self, idx):
        chemin = self.chemins[idx]
        
        words, boxes = self.ocr_data.get(chemin, (["[PAD]"], [[0, 0, 0, 0]]))
        
        img = cv2.imread(chemin)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
        
        pil_image = Image.fromarray(img)
        
        encoding = self.processor(
            pil_image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )
        
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

# ============================================================
# MODÈLE (identique)
# ============================================================

class DetecteurChequeLayoutLM:
    def __init__(self):
        self.device = device
        
        print("\n📦 Chargement de LayoutLMv3-base...")
        
        self.processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base",
            apply_ocr=False
        )
        
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            "microsoft/layoutlmv3-base",
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        
        self.model = self.model.to(self.device)
        
        if torch.cuda.is_available():
            print("✅ GPU disponible - entraînement accéléré")
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            print("⚠️ CPU seulement - sera lent")
            self.scaler = None
        
        self.criterion = nn.CrossEntropyLoss()
    
    def preparer_dataset(self, dossier_positifs, dossier_negatifs, val_split=0.2):
        """Prépare les datasets"""
        print("\n📂 Préparation du dataset...")
        
        chemins_pos = []
        for dossier in dossier_positifs:
            if os.path.exists(dossier):
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    chemins_pos.extend(glob.glob(os.path.join(dossier, ext)))
                print(f"   {os.path.basename(dossier)}: {len(chemins_pos)} images")
        
        chemins_neg = []
        if os.path.exists(dossier_negatifs):
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                chemins_neg.extend(glob.glob(os.path.join(dossier_negatifs, ext)))
            print(f"   negatifs: {len(chemins_neg)} images")
        
        print(f"\n   ✅ TOTAL: {len(chemins_pos)} chèques, {len(chemins_neg)} non-chèques")
        
        # Équilibrer
        nb_min = min(len(chemins_pos), len(chemins_neg))
        chemins_pos = random.sample(chemins_pos, nb_min)
        chemins_neg = random.sample(chemins_neg, nb_min)
        
        tous_chemins = chemins_pos + chemins_neg
        tous_labels = [1] * len(chemins_pos) + [0] * len(chemins_neg)
        
        train_chemins, val_chemins, train_labels, val_labels = train_test_split(
            tous_chemins, tous_labels, 
            test_size=val_split, 
            random_state=42,
            stratify=tous_labels
        )
        
        print(f"\n📊 Dataset final:")
        print(f"   - Train: {len(train_chemins)} images")
        print(f"   - Validation: {len(val_chemins)} images")
        
        train_dataset = DatasetChequeLayoutLM(train_chemins, train_labels, self.processor, augment=True)
        val_dataset = DatasetChequeLayoutLM(val_chemins, val_labels, self.processor, augment=False)
        
        return train_dataset, val_dataset
    
    def entrainer(self, dossier_positifs, dossier_negatifs, epochs=5, batch_size=8, lr=2e-5):
        """Entraînement avec GPU"""
        print("\n" + "="*60)
        print("🏋️ ENTRAÎNEMENT LAYOUTLMv3 SUR GPU")
        print("="*60)
        
        t0 = time.time()
        train_dataset, val_dataset = self.preparer_dataset(
            dossier_positifs, dossier_negatifs
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            num_workers=2,
            pin_memory=True
        )
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Entraînement
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop('labels')
                
                optimizer.zero_grad()
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = outputs.loss if outputs.loss is not None else self.criterion(outputs.logits, labels)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss if outputs.loss is not None else self.criterion(outputs.logits, labels)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{train_correct/train_total:.2%}'
                })
            
            train_acc = train_correct / train_total
            
            # Validation
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    labels = batch.pop('labels')
                    
                    outputs = self.model(**batch)
                    preds = torch.argmax(outputs.logits, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = val_correct / val_total
            
            print(f"\n   Epoch {epoch+1}: Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), "meilleur_modele_layoutlm.pth")
                print(f"   💾 Meilleur modèle sauvegardé!")
        
        print(f"\n✅ Entraînement terminé en {time.time()-t0:.1f}s!")
        print(f"   Meilleure val_acc: {best_val_acc:.2%}")
        
        return self.model

# ============================================================
# MAIN - ADAPTÉ À VOTRE STRUCTURE
# ============================================================

print("="*60)
print("🖼️  DÉTECTION DE CHÈQUES AVEC LAYOUTLMv3")
print("="*60)
print("📁 Structure attendue: output.rar avec dossiers images/, images_augmentees/, negatifs/")

# Choix de la source
print("\n📁 SOURCE DES DONNÉES:")
print("1. Upload output.rar depuis mon PC (recommandé)")
print("2. Utiliser Google Drive (si déjà uploadé)")
print("3. Spécifier les chemins manuellement")

choix_source = input("\nVotre choix (1/2/3): ").strip()

if choix_source == "1":
    dossier_positifs, dossier_negatifs = upload_dataset()
elif choix_source == "2":
    dossier_positifs, dossier_negatifs = use_drive()
else:
    # Chemins manuels
    base_path = input("Chemin de base (ex: ./output ou /content/output): ").strip()
    dossier_positifs = [
        os.path.join(base_path, 'images'),
        os.path.join(base_path, 'images_augmentees')
    ]
    dossier_negatifs = os.path.join(base_path, 'negatifs')

# Vérification finale
if not dossier_negatifs or not os.path.exists(dossier_negatifs):
    print(f"\n❌ Dossier negatifs non trouvé: {dossier_negatifs}")
    print("Vérifiez que votre archive contient le dossier 'negatifs'")
    exit()

print("\n✅ Configuration OK - Prêt pour l'entraînement!")

# Menu principal
print("\n📋 MENU PRINCIPAL:")
print("1. Entraînement complet (recommandé)")
print("2. Télécharger le modèle entraîné")

choix = input("\nVotre choix (1/2): ").strip()

if choix == "1":
    detecteur = DetecteurChequeLayoutLM()
    
    epochs = input("Nombre d'epochs (défaut: 5): ").strip()
    epochs = int(epochs) if epochs else 5
    
    batch_size = input("Batch size (défaut: 8): ").strip()
    batch_size = int(batch_size) if batch_size else 8
    
    detecteur.entrainer(
        dossier_positifs=dossier_positifs,
        dossier_negatifs=dossier_negatifs,
        epochs=epochs,
        batch_size=batch_size,
        lr=2e-5
    )
    
    print("\n📥 Téléchargement du modèle...")
    files.download("meilleur_modele_layoutlm.pth")
    print("✅ Modèle téléchargé - à placer dans votre dossier local")

elif choix == "2":
    if os.path.exists("meilleur_modele_layoutlm.pth"):
        files.download("meilleur_modele_layoutlm.pth")
        print("✅ Modèle téléchargé")
    else:
        print("❌ Aucun modèle trouvé - faites d'abord un entraînement")