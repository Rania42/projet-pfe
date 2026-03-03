import os
import sys
import random
import json
import string
from datetime import datetime, timedelta
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from faker import Faker
from pdf2image import convert_from_path
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

class GenerateurChequeBNP:
    def __init__(self):
        # Correction du chemin : remonter de 'generateurs' vers 'data_generation'
        self.base_path = Path(__file__).parent.parent  # data_generation/
        self.template_dir = self.base_path / 'templates'
        self.output_dir = self.base_path / 'output'
        
        print(f"📁 Base path: {self.base_path}")
        print(f"📁 Template path: {self.template_dir}")
        print(f"📁 Output path: {self.output_dir}")
        
        # Vérification que le template existe
        template_path = self.template_dir / 'cheque.html'
        if not template_path.exists():
            print(f"❌ ERREUR: Template non trouvé à {template_path}")
            print("Fichiers disponibles dans templates:", list(self.template_dir.glob("*.html")))
            sys.exit(1)
        else:
            print(f"✅ Template trouvé: {template_path}")
        
        # Créer les dossiers de sortie
        for dossier in ['images', 'images_augmentees', 'metadata', 'pdfs']:
            (self.output_dir / dossier).mkdir(parents=True, exist_ok=True)
            print(f"✅ Dossier créé/vérifié: {self.output_dir / dossier}")
        
        # Template Jinja2
        self.env = Environment(loader=FileSystemLoader(str(self.template_dir)))
        self.template = self.env.get_template('cheque.html')
        
        # Faker pour données aléatoires
        self.faker = Faker('fr_FR')
        
        # Données spécifiques BNP
        self.villes = ["PARIS", "LYON", "MARSEILLE", "BORDEAUX", "LILLE", "NANTES", 
                       "STRASBOURG", "TOULOUSE", "NICE", "RENNES"]
        
        self.rues = [
            "16, boulevard des Italiens",
            "125, avenue des Champs-Élysées",
            "45, rue de la République",
            "8, place de la Bastille",
            "22, rue du Commerce",
            "100, boulevard Haussmann"
        ]
        
        # Signatures types
        self.signatures = ["Marck", "J. Dupont", "M. Martin", "L. Bernard", 
                          "P. Petit", "A. Robert", "C. Richard", "S. Durand"]
        
        print("✅ Générateur initialisé avec succès!\n")
    
    def generer_iban(self):
        """Génère un IBAN français réaliste"""
        banque = random.randint(10000, 99999)
        guichet = random.randint(10000, 99999)
        compte = ''.join([str(random.randint(0, 9)) for _ in range(11)])
        cle = random.randint(10, 99)
        return f"FR{cle} {banque:05d} {guichet:05d} {compte} {random.randint(10,99)}"
    
    def generer_compte(self):
        """Génère un numéro de compte"""
        return f"000{random.randint(1000000, 9999999)}"
    
    def nombre_en_lettres(self, nombre):
        """Convertit un nombre en lettres"""
        nombre_entier = int(nombre)
        
        if nombre_entier == 0:
            return "ZERO"
        
        unites = ["", "UN", "DEUX", "TROIS", "QUATRE", "CINQ", "SIX", "SEPT", "HUIT", "NEUF",
                  "DIX", "ONZE", "DOUZE", "TREIZE", "QUATORZE", "QUINZE", "SEIZE",
                  "DIX-SEPT", "DIX-HUIT", "DIX-NEUF"]
        
        dizaines = ["", "DIX", "VINGT", "TRENTE", "QUARANTE", "CINQUANTE", 
                    "SOIXANTE", "SOIXANTE-DIX", "QUATRE-VINGT", "QUATRE-VINGT-DIX"]
        
        def convert_moins_1000(n):
            if n == 0:
                return ""
            
            parties = []
            
            # Centaines
            centaines = n // 100
            if centaines > 0:
                if centaines == 1:
                    parties.append("CENT")
                else:
                    parties.append(unites[centaines] + " CENT")
                n %= 100
            
            # Dizaines et unités
            if n > 0:
                if n < 20:
                    parties.append(unites[n])
                else:
                    d = n // 10
                    u = n % 10
                    
                    if d == 7:  # soixante-dix
                        if u == 1:
                            parties.append("SOIXANTE ET ONZE")
                        elif u == 0:
                            parties.append("SOIXANTE-DIX")
                        else:
                            parties.append("SOIXANTE-" + unites[10 + u])
                    elif d == 8:  # quatre-vingt
                        if u == 0:
                            parties.append("QUATRE-VINGTS")
                        else:
                            parties.append("QUATRE-VINGT-" + unites[u])
                    elif d == 9:  # quatre-vingt-dix
                        if u == 0:
                            parties.append("QUATRE-VINGT-DIX")
                        else:
                            parties.append("QUATRE-VINGT-" + unites[10 + u])
                    else:
                        if u == 1:
                            parties.append(dizaines[d] + " ET UN")
                        elif u == 0:
                            parties.append(dizaines[d])
                        else:
                            parties.append(dizaines[d] + "-" + unites[u])
            
            return " ".join(parties)
        
        resultat = []
        
        # Milliers
        milliers = nombre_entier // 1000
        if milliers > 0:
            if milliers == 1:
                resultat.append("MILLE")
            else:
                resultat.append(convert_moins_1000(milliers) + " MILLE")
            nombre_entier %= 1000
        
        if nombre_entier > 0:
            resultat.append(convert_moins_1000(nombre_entier))
        
        return " ".join(resultat) + " EUROS"
    
    def generer_donnees(self, index):
        """Génère toutes les données pour un chèque"""
        montant = round(random.uniform(10, 5000), 2)
        date = self.faker.date_between(start_date='-2y', end_date='today')
        ville = random.choice(self.villes)
        
        return {
            "ville_emission": ville,
            "date_emission": date.strftime("%d/%m/%Y"),
            "beneficiaire": self.faker.name().upper(),
            "montant_chiffres": f"{montant:,.2f}".replace(",", " ").replace(".", ","),
            "montant_lettres": self.nombre_en_lettres(montant),
            "signature": random.choice(self.signatures),
            "ville_agence": ville,
            "adresse_ligne1": "IMMEUBLE PAVILLON IN",
            "adresse_ligne2": random.choice(self.rues),
            "adresse_ligne3": f"{random.randint(1000, 99999)} {ville}",
            "compte": self.generer_compte(),
            "cle_rib": random.randint(10, 99),
            "iban": self.generer_iban(),
            "bic": "BNPAFRPPXXX",
            "numero_cheque": f"{random.randint(1000000, 9999999)}"
        }
    
    def generer_image(self, index):
        """Génère une image de chèque BNP"""
        donnees = self.generer_donnees(index)
        
        try:
            # Rendu HTML
            html = self.template.render(**donnees)
            
            # HTML -> PDF
            pdf_path = self.output_dir / 'pdfs' / f"cheque_{index:04d}.pdf"
            HTML(string=html).write_pdf(str(pdf_path))
            
            # PDF -> Image
            try:
                poppler_path = r"C:\Program Files\poppler\Library\bin"
                images = convert_from_path(str(pdf_path), dpi=200, poppler_path=poppler_path)
            except:
                print("⚠️ Poppler non trouvé, utilisation du chemin par défaut")
                images = convert_from_path(str(pdf_path), dpi=200)
            
            # Sauvegarder image
            img_path = self.output_dir / 'images' / f"cheque_{index:04d}.png"
            images[0].save(str(img_path), 'PNG')
            
            # Nettoyer le PDF (optionnel)
            # os.remove(pdf_path)
            
            return img_path, donnees
            
        except Exception as e:
            print(f"❌ Erreur génération chèque {index}: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def augmenter_image(self, img_path, niveau='moyen'):
        """Applique des augmentations réalistes"""
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"⚠️ Erreur lecture: {img_path}")
                return None
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Paramètres selon niveau
            params = {
                'faible': {'bruit': 10, 'flou': 1, 'rot': 1},
                'moyen': {'bruit': 20, 'flou': 2, 'rot': 2},
                'fort': {'bruit': 30, 'flou': 3, 'rot': 3},
                'noir_blanc': {'bruit': 15, 'flou': 1, 'rot': 1}
            }
            
            p = params[niveau]
            
            # Transformations
            transform = A.Compose([
                A.Rotate(limit=p['rot'], p=0.8),
                A.GaussNoise(var_limit=p['bruit'], p=0.7),
                A.GaussianBlur(blur_limit=p['flou'], p=0.5),
                A.RandomBrightnessContrast(p=0.8),
            ])
            
            if niveau == 'noir_blanc':
                transform = A.Compose([
                    A.ToGray(p=1.0),
                    A.Rotate(limit=p['rot'], p=0.8),
                    A.GaussNoise(var_limit=p['bruit'], p=0.7),
                ])
            
            aug = transform(image=image)['image']
            aug_bgr = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)
            
            # Sauvegarder
            name = img_path.stem
            aug_path = self.output_dir / 'images_augmentees' / f"{name}_{niveau}.png"
            cv2.imwrite(str(aug_path), aug_bgr)
            
            return aug_path
            
        except Exception as e:
            print(f"⚠️ Erreur augmentation {niveau}: {e}")
            return None
    
    def generer_dataset(self, nombre=30):
        """Génère un dataset complet"""
        print(f"\n{'='*60}")
        print(f"GÉNÉRATION DE {nombre} CHÈQUES BNP PARIBAS")
        print(f"{'='*60}\n")
        
        metadata = []
        
        for i in tqdm(range(nombre), desc="Progression"):
            # Image originale
            img_path, donnees = self.generer_image(i)
            
            if img_path is None:
                continue
            
            meta = {
                "id": i,
                "image_originale": str(img_path.relative_to(self.base_path)),
                "donnees": donnees,
                "variantes": []
            }
            
            # Générer 4 variantes par image
            for niveau in ['faible', 'moyen', 'fort', 'noir_blanc']:
                aug_path = self.augmenter_image(img_path, niveau)
                if aug_path:
                    meta["variantes"].append({
                        "type": niveau,
                        "chemin": str(aug_path.relative_to(self.base_path))
                    })
            
            metadata.append(meta)
        
        # Sauvegarder métadonnées
        meta_path = self.output_dir / 'metadata' / 'dataset_bnp.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Statistiques
        total_images = len(metadata) + sum(len(m['variantes']) for m in metadata)
        
        print(f"\n{'='*60}")
        print(f"✅ GÉNÉRATION TERMINÉE!")
        print(f"{'='*60}")
        print(f"📊 Statistiques:")
        print(f"   - Chèques originaux: {len(metadata)}")
        print(f"   - Images augmentées: {total_images - len(metadata)}")
        print(f"   - Total images: {total_images}")
        print(f"   - Métadonnées: {meta_path}")
        
        # Afficher les dossiers de sortie
        print(f"\n📁 Dossiers de sortie:")
        print(f"   - Images: {self.output_dir / 'images'}")
        print(f"   - Images augmentées: {self.output_dir / 'images_augmentees'}")
        print(f"   - PDFs: {self.output_dir / 'pdfs'}")
        print(f"{'='*60}\n")
        
        return metadata

def main():
    """Fonction principale"""
    print("=== GÉNÉRATEUR DE CHÈQUES BNP PARIBAS ===\n")
    
    try:
        nombre = int(input("📝 Nombre de chèques à générer (défaut: 30): ") or "30")
    except:
        nombre = 30
    
    generateur = GenerateurChequeBNP()
    generateur.generer_dataset(nombre)

if __name__ == "__main__":
    main()