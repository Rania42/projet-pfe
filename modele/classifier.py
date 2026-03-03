import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
import albumentations as A
import cv2
import numpy as np
from PIL import Image
import pytesseract

class LayoutLMv3FewShot:
    """
    Véritable LayoutLMv3 avec fine-tuning sur 5 images
    Utilise la data augmentation pour simuler plus d'exemples
    """
    
    def __init__(self, nb_images=5):
        self.processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base",
            apply_ocr=False
        )
        
        # Modèle pré-entraîné
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            "microsoft/layoutlmv3-base",
            num_labels=2,  # chèque ou pas chèque
            ignore_mismatched_sizes=True
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Data augmentation extrême
        self.augmenter = A.Compose([
            A.RandomBrightnessContrast(p=0.9),
            A.Rotate(limit=10, p=0.8),
            A.GaussNoise(var_limit=(10, 30), p=0.7),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            A.CLAHE(p=0.5),
            A.CoarseDropout(max_holes=5, max_height=50, max_width=50, p=0.5),
            A.ElasticTransform(p=0.3),
            A.Perspective(p=0.3),
        ])
        
        print(f"🚀 LayoutLMv3 initialisé sur {self.device}")
        print(f"📊 Mode few-shot avec {nb_images} images de base")
    
    def augmenter_image(self, image_path, n_variations=20):
        """
        Crée 20 variations de chaque image
        Total: 5 images × 20 = 100 images d'entraînement
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        images_augmentees = []
        
        for _ in range(n_variations):
            augmente = self.augmenter(image=image)['image']
            images_augmentees.append(Image.fromarray(augmente))
        
        return images_augmentees
    
    def extraire_texte_boxes(self, image):
        """OCR pour LayoutLMv3"""
        # Image PIL -> texte + boxes
        largeur, hauteur = image.size
        
        # OCR avec Tesseract
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        mots = []
        boxes = []
        
        for i, texte in enumerate(data['text']):
            if texte.strip():
                mots.append(texte.strip())
                
                # Normalisation 0-1000
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                
                x1 = int(1000 * x / largeur)
                y1 = int(1000 * y / hauteur)
                x2 = int(1000 * (x + w) / largeur)
                y2 = int(1000 * (y + h) / hauteur)
                
                boxes.append([x1, y1, x2, y2])
        
        if not mots:
            mots = ["cheque", "document"]
            boxes = [[0, 0, 1000, 1000]]
        
        return " ".join(mots), boxes
    
    def preparer_dataset(self, chemins_images, labels):
        """
        Prépare le dataset avec augmentation
        chemins_images: liste des 5 chemins
        labels: [1, 1, 1, 1, 1] (tous des chèques)
        """
        donnees = []
        
        for chemin, label in zip(chemins_images, labels):
            # Image originale
            image_orig = Image.open(chemin).convert("RGB")
            texte, boxes = self.extraire_texte_boxes(image_orig)
            donnees.append((image_orig, texte, boxes, label))
            
            # 20 variations augmentées
            variations = self.augmenter_image(chemin, n_variations=20)
            for img_aug in variations:
                texte, boxes = self.extraire_texte_boxes(img_aug)
                donnees.append((img_aug, texte, boxes, label))
        
        print(f"📊 Dataset créé: {len(donnees)} images")
        print(f"   • Originales: {len(chemins_images)}")
        print(f"   • Augmentées: {len(donnees) - len(chemins_images)}")
        
        return donnees
    
    def fine_tune(self, chemins_cheques, epochs=10):
        """
        Fine-tuning du modèle LayoutLMv3
        """
        print("\n" + "="*60)
        print("FINE-TUNING LAYOUTLMv3 AVEC 5 IMAGES")
        print("="*60)
        
        # Préparer les données (5 images × 20 = 100)
        donnees = self.preparer_dataset(
            chemins_cheques, 
            [1] * len(chemins_cheques)  # tous label 1
        )
        
        # Split train/val
        split = int(0.8 * len(donnees))
        train_data = donnees[:split]
        val_data = donnees[split:]
        
        print(f"\n📁 Train: {len(train_data)} images")
        print(f"📁 Validation: {len(val_data)} images")
        
        # Optimiseur avec learning rate réduit (few-shot)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        # Early stopping
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            # Entraînement
            self.model.train()
            train_correct = 0
            
            for image, texte, boxes, label in train_data:
                # Préparer l'encoding pour LayoutLM
                encoding = self.processor(
                    image,
                    text=texte,
                    boxes=[boxes],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=512,
                    truncation=True
                )
                
                encoding = {k: v.to(self.device) for k, v in encoding.items()}
                labels = torch.tensor([label]).to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(**encoding, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                pred = outputs.logits.argmax(-1).item()
                if pred == label:
                    train_correct += 1
            
            train_acc = train_correct / len(train_data)
            
            # Validation
            self.model.eval()
            val_correct = 0
            
            with torch.no_grad():
                for image, texte, boxes, label in val_data:
                    encoding = self.processor(
                        image, texte, boxes=[boxes],
                        return_tensors="pt", padding="max_length", max_length=512
                    )
                    encoding = {k: v.to(self.device) for k, v in encoding.items()}
                    outputs = self.model(**encoding)
                    pred = outputs.logits.argmax(-1).item()
                    if pred == label:
                        val_correct += 1
            
            val_acc = val_correct / len(val_data)
            
            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Sauvegarder le meilleur modèle
                torch.save(self.model.state_dict(), "modele_layoutlm_fewshot.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"⏹️ Early stopping à l'epoch {epoch+1}")
                    break
        
        print(f"\n✅ Meilleure accuracy validation: {best_val_acc:.2%}")
        print("💾 Modèle sauvegardé: modele_layoutlm_fewshot.pth")
    
    def predire(self, chemin_image):
        """Prédiction sur une nouvelle image"""
        image = Image.open(chemin_image).convert("RGB")
        texte, boxes = self.extraire_texte_boxes(image)
        
        encoding = self.processor(
            image, texte, boxes=[boxes],
            return_tensors="pt", padding="max_length", max_length=512
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encoding)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred = outputs.logits.argmax(-1).item()
        
        prob_cheque = probs[0][1].item()
        
        print(f"\n📄 {os.path.basename(chemin_image)}")
        print(f"   Probabilité chèque: {prob_cheque:.2%}")
        print(f"   Prédiction: {'✅ CHÈQUE' if pred == 1 else '❌ AUTRE'}")
        
        return pred, prob_cheque

# Utilisation
if __name__ == "__main__":
    # 1. Initialiser LayoutLMv3
    detecteur = LayoutLMv3FewShot(nb_images=5)
    
    # 2. Liste de vos 5 chèques
    chemins = [
        "output/images/cheque_0000.png",
        "output/images/cheque_0001.png",
        "output/images/cheque_0002.png",
        "output/images/cheque_0003.png",
        "output/images/cheque_0004.png"
    ]
    
    # 3. Fine-tuning
    detecteur.fine_tune(chemins, epochs=10)
    
    # 4. Tester sur un vrai chèque manuscrit
    detecteur.predire("C:/Users/lenovo/Desktop/3.jpg")