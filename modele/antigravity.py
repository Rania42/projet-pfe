"""
╔══════════════════════════════════════════════════════════════════╗
║   Classification Chèque / RIB — Pipeline Hybride                ║
║   Modèle : EfficientNet-B0 (vision) + TF-IDF OCR (texte)        ║
║   Fusion  : Ensemble par vote pondéré                            ║
╚══════════════════════════════════════════════════════════════════╝

Structure attendue :
    dataset/
        cheque/   ← images de chèques
        rib/      ← images de RIBs

Usage :
    python train.py
"""

import os
import sys
import json
import random
import shutil
import warnings
import pickle
import logging
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance, ImageDraw

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
CFG = {
    # Chemins
    "dataset_dir"    : "dataset",
    "output_dir"     : "outputs",
    "model_dir"      : "outputs/models",

    # Classes
    "classes"        : ["cheque", "rib"],
    "other_label"    : "autre",

    # Augmentation
    "target_per_class": 80,       # images par classe après augmentation

    # Vision (EfficientNet)
    "img_size"       : 224,
    "batch_size"     : 8,
    "epochs"         : 30,
    "lr"             : 1e-4,
    "weight_decay"   : 1e-4,
    "dropout"        : 0.4,
    "patience"       : 8,         # early stopping

    # Texte (TF-IDF + SVM)
    "tfidf_max_features": 3000,
    "tfidf_ngram_range" : (1, 3),
    "svm_C"             : 10.0,
    "svm_kernel"        : "rbf",

    # Ensemble
    "vision_weight"  : 0.55,      # poids modèle vision dans l'ensemble
    "text_weight"    : 0.45,      # poids modèle texte dans l'ensemble
    "confidence_threshold": 0.70, # seuil pour répondre "autre"

    # Reproductibilité
    "seed"           : 42,
}

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])

os.makedirs(CFG["output_dir"], exist_ok=True)
os.makedirs(CFG["model_dir"],  exist_ok=True)

log.info(f"Device : {DEVICE}")
log.info(f"Config : {CFG}")


# ─────────────────────────────────────────────
#  1. PRÉ-TRAITEMENT IMAGE (OCR-optimisé)
# ─────────────────────────────────────────────
def preprocess_image(image: Image.Image, for_ocr: bool = False) -> Image.Image:
    """
    Pipeline de pré-traitement :
      - Redimensionnement minimum 1000px (pour OCR)
      - Débruitage fastNlMeans
      - CLAHE pour améliorer le contraste local
    """
    img = image.convert("RGB")
    w, h = img.size

    if for_ocr and min(w, h) < 1000:
        scale = 1000 / min(w, h)
        img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    arr      = np.array(img)
    gray     = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    rgb      = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


# ─────────────────────────────────────────────
#  2. OCR — EXTRACTION DU TEXTE
# ─────────────────────────────────────────────
def extract_text_from_image(image: Image.Image) -> str:
    """Extrait le texte via Tesseract avec config optimisée documents bancaires."""
    img_preprocessed = preprocess_image(image, for_ocr=True)
    try:
        # PSM 6 = bloc de texte uniforme, idéal pour docs bancaires
        custom_config = r"--oem 3 --psm 6 -l fra+eng"
        text = pytesseract.image_to_string(img_preprocessed, config=custom_config)
        return text.strip()
    except Exception as e:
        log.warning(f"OCR échoué : {e}")
        return ""


# ─────────────────────────────────────────────
#  3. AUGMENTATION DES DONNÉES
# ─────────────────────────────────────────────
def augment_image(img: Image.Image) -> list:
    """Génère des variantes augmentées d'une image."""
    augmented = []
    arr = np.array(img.convert("RGB"))

    # Rotations légères
    for angle in [-3, -1, 1, 3]:
        h, w = arr.shape[:2]
        M   = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rot = cv2.warpAffine(arr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmented.append(Image.fromarray(rot))

    # Luminosité
    for factor in [0.75, 0.9, 1.1, 1.25]:
        augmented.append(ImageEnhance.Brightness(img).enhance(factor))

    # Contraste
    for factor in [0.8, 1.2]:
        augmented.append(ImageEnhance.Contrast(img).enhance(factor))

    # Bruit gaussien
    for sigma in [3, 7]:
        noisy = arr.copy().astype(np.float32)
        noisy += np.random.normal(0, sigma, arr.shape)
        augmented.append(Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8)))

    # Flou léger
    augmented.append(Image.fromarray(cv2.GaussianBlur(arr, (3, 3), 0)))

    return augmented


def build_augmented_dataset(dataset_dir: str, target_per_class: int) -> dict:
    """
    Charge le dataset, équilibre et augmente les deux classes.
    Retourne : {label: [chemins images]}
    """
    aug_dir = os.path.join(CFG["output_dir"], "augmented")

    all_paths = {}
    for cls in CFG["classes"]:
        cls_dir = os.path.join(dataset_dir, cls)
        assert os.path.isdir(cls_dir), f"❌ Dossier introuvable : {cls_dir}"

        src = sorted([
            os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
            if f.lower().endswith(EXTENSIONS)
        ])
        assert len(src) > 0, f"❌ Aucune image dans {cls_dir}"

        out_dir = os.path.join(aug_dir, cls)
        os.makedirs(out_dir, exist_ok=True)

        paths = []
        # Copier originaux
        for p in src:
            dst = os.path.join(out_dir, os.path.basename(p))
            shutil.copy(p, dst)
            paths.append(dst)

        # Augmenter jusqu'à target_per_class
        i = 0
        while len(paths) < target_per_class:
            img = Image.open(src[i % len(src)]).convert("RGB")
            for j, aug in enumerate(augment_image(img)):
                if len(paths) >= target_per_class:
                    break
                name = f"aug_{i:03d}_{j}_{os.path.basename(src[i % len(src)])}"
                dst  = os.path.join(out_dir, name)
                aug.save(dst)
                paths.append(dst)
            i += 1

        all_paths[cls] = paths
        log.info(f"  [{cls}] {len(src)} originaux → {len(paths)} images après augmentation")

    return all_paths


# ─────────────────────────────────────────────
#  4. DATASET PyTorch (pour EfficientNet)
# ─────────────────────────────────────────────
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])


class DocumentImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            img = preprocess_image(img, for_ocr=False)
        except Exception:
            img = Image.new("RGB", (CFG["img_size"], CFG["img_size"]), "white")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(self.labels[idx], dtype=torch.long)


# ─────────────────────────────────────────────
#  5. MODÈLE VISION — EfficientNet-B0
# ─────────────────────────────────────────────
def build_efficientnet(num_classes: int = 2) -> nn.Module:
    """EfficientNet-B0 pré-entraîné sur ImageNet, tête remplacée."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Geler les premières couches (feature extraction)
    for name, param in model.features[:5].named_parameters():
        param.requires_grad = False

    # Remplacer la tête de classification
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=CFG["dropout"]),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=CFG["dropout"] / 2),
        nn.Linear(256, num_classes)
    )
    return model


def train_vision_model(train_paths, train_labels, val_paths, val_labels):
    """Entraîne EfficientNet-B0 avec early stopping."""
    log.info("\n" + "═" * 55)
    log.info("  MODULE 1 : ENTRAÎNEMENT VISION (EfficientNet-B0)")
    log.info("═" * 55)

    train_ds = DocumentImageDataset(train_paths, train_labels, TRAIN_TRANSFORMS)
    val_ds   = DocumentImageDataset(val_paths,   val_labels,   EVAL_TRANSFORMS)
    train_dl = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False, num_workers=0)

    model     = build_efficientnet(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG["epochs"])

    best_val_acc  = 0.0
    patience_ctr  = 0
    history       = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_path     = os.path.join(CFG["model_dir"], "efficientnet_best.pth")

    log.info(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>9} | {'Train Acc':>10} | {'Val Acc':>8}")
    log.info("─" * 55)

    for epoch in range(1, CFG["epochs"] + 1):
        # ── Train ──
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss    += loss.item()
            tr_correct += (logits.argmax(1) == lbls).sum().item()
            tr_total   += lbls.size(0)
        scheduler.step()

        # ── Eval ──
        model.eval()
        vl_loss, vl_correct, vl_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_dl:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                logits      = model(imgs)
                vl_loss    += criterion(logits, lbls).item()
                vl_correct += (logits.argmax(1) == lbls).sum().item()
                vl_total   += lbls.size(0)

        tr_acc = tr_correct / tr_total
        vl_acc = vl_correct / vl_total
        history["train_loss"].append(tr_loss / len(train_dl))
        history["val_loss"]  .append(vl_loss / len(val_dl))
        history["train_acc"] .append(tr_acc)
        history["val_acc"]   .append(vl_acc)

        flag = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_ctr = 0
            torch.save(model.state_dict(), best_path)
            flag = "  🏆"
        else:
            patience_ctr += 1

        log.info(
            f"{epoch:>6} | {tr_loss/len(train_dl):>10.4f} | {vl_loss/len(val_dl):>9.4f} | "
            f"{tr_acc:>10.4f} | {vl_acc:>8.4f}{flag}"
        )

        if patience_ctr >= CFG["patience"]:
            log.info(f"  ⏹  Early stopping à l'epoch {epoch}")
            break

    log.info(f"\n✅ Vision — Meilleure val accuracy : {best_val_acc:.4f}")

    # Recharger le meilleur modèle
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    return model, history


@torch.no_grad()
def get_vision_probs(model, paths: list) -> np.ndarray:
    """Retourne les probabilités softmax pour une liste d'images."""
    model.eval()
    all_probs = []
    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
            img = preprocess_image(img, for_ocr=False)
        except Exception:
            img = Image.new("RGB", (CFG["img_size"], CFG["img_size"]), "white")
        tensor = EVAL_TRANSFORMS(img).unsqueeze(0).to(DEVICE)
        probs  = torch.softmax(model(tensor), dim=-1)[0].cpu().numpy()
        all_probs.append(probs)
    return np.array(all_probs)


# ─────────────────────────────────────────────
#  6. MODÈLE TEXTE — TF-IDF + SVM
# ─────────────────────────────────────────────
def train_text_model(train_paths, train_labels, val_paths, val_labels):
    """Entraîne un pipeline TF-IDF + SVM sur le texte OCR."""
    log.info("\n" + "═" * 55)
    log.info("  MODULE 2 : ENTRAÎNEMENT TEXTE (TF-IDF + SVM)")
    log.info("═" * 55)

    log.info("  ⏳ Extraction OCR du train set ...")
    train_texts = [extract_text_from_image(Image.open(p)) for p in train_paths]
    log.info("  ⏳ Extraction OCR du val set ...")
    val_texts   = [extract_text_from_image(Image.open(p)) for p in val_paths]

    # Pipeline : TF-IDF → SVM
    text_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features  = CFG["tfidf_max_features"],
            ngram_range   = CFG["tfidf_ngram_range"],
            sublinear_tf  = True,
            strip_accents = "unicode",
            analyzer      = "word",
            min_df        = 1,
        )),
        ("svm", SVC(
            C          = CFG["svm_C"],
            kernel     = CFG["svm_kernel"],
            probability= True,
            random_state= CFG["seed"],
        )),
    ])

    text_pipeline.fit(train_texts, train_labels)

    val_preds = text_pipeline.predict(val_texts)
    val_acc   = accuracy_score(val_labels, val_preds)
    log.info(f"✅ Texte — Val accuracy : {val_acc:.4f}")

    model_path = os.path.join(CFG["model_dir"], "text_pipeline.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(text_pipeline, f)
    log.info(f"   Modèle texte sauvegardé : {model_path}")

    return text_pipeline, train_texts, val_texts


def get_text_probs(text_model, paths: list) -> np.ndarray:
    """Retourne les probabilités SVM pour une liste d'images (via OCR)."""
    texts = [extract_text_from_image(Image.open(p)) for p in paths]
    return text_model.predict_proba(texts)


# ─────────────────────────────────────────────
#  7. ÉVALUATION ENSEMBLE
# ─────────────────────────────────────────────
def ensemble_predict(vision_probs: np.ndarray, text_probs: np.ndarray) -> np.ndarray:
    """Fusion pondérée des deux modèles."""
    combined = (
        CFG["vision_weight"] * vision_probs +
        CFG["text_weight"]   * text_probs
    )
    return combined


def evaluate_ensemble(vision_model, text_model, paths, labels, split_name="Test"):
    """Évalue le modèle ensemble sur un jeu de données."""
    log.info(f"\n  ⏳ Évaluation ensemble sur {split_name} ...")
    v_probs  = get_vision_probs(vision_model, paths)
    t_probs  = get_text_probs(text_model, paths)
    combined = ensemble_predict(v_probs, t_probs)
    preds    = combined.argmax(axis=1)

    log.info(f"\n{'═'*60}")
    log.info(f"  RAPPORT DE CLASSIFICATION — {split_name}")
    log.info(f"{'═'*60}")
    log.info("\n" + classification_report(
        labels, preds,
        target_names=CFG["classes"],
        digits=4
    ))
    return preds, combined


# ─────────────────────────────────────────────
#  8. GRAPHIQUES
# ─────────────────────────────────────────────
def plot_learning_curves(history: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(history["train_loss"]) + 1)

    ax1.plot(ep, history["train_loss"], label="Train", color="#2196F3", lw=2)
    ax1.plot(ep, history["val_loss"],   label="Val",   color="#F44336", lw=2)
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(ep, history["train_acc"], label="Train", color="#2196F3", lw=2)
    ax2.plot(ep, history["val_acc"],   label="Val",   color="#F44336", lw=2)
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch")
    ax2.set_ylim(0, 1.05); ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Courbes d'apprentissage — EfficientNet-B0", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CFG["output_dir"], "learning_curves.png"), dpi=150)
    plt.close()
    log.info("  📊 Courbes sauvegardées")


def plot_confusion_matrix(true_labels, pred_labels, title="Matrice de Confusion"):
    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CFG["classes"],
        yticklabels=CFG["classes"],
        ax=ax, linewidths=1, annot_kws={"size": 14}
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Réel", fontsize=12)
    ax.set_xlabel("Prédit", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG["output_dir"], "confusion_matrix.png"), dpi=150)
    plt.close()
    log.info("  📊 Matrice de confusion sauvegardée")


def plot_distribution(train_labels, test_labels):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, lbls, title in zip(axes, [train_labels, test_labels], ["Train", "Test"]):
        counts = [lbls.count(0), lbls.count(1)]
        bars   = ax.bar(CFG["classes"], counts, color=["#2196F3", "#4CAF50"], edgecolor="white", width=0.5)
        ax.set_title(f"Distribution {title}", fontsize=12)
        ax.set_ylabel("Nombre d'images")
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, c + 0.2, str(c),
                    ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CFG["output_dir"], "distribution.png"), dpi=100)
    plt.close()
    log.info("  📊 Distribution sauvegardée")


# ─────────────────────────────────────────────
#  9. SAUVEGARDE
# ─────────────────────────────────────────────
def save_results(metrics: dict):
    path = os.path.join(CFG["output_dir"], "results_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log.info(f"  💾 Résultats sauvegardés : {path}")

    path_cfg = os.path.join(CFG["output_dir"], "inference_config.json")
    inference_cfg = {
        "classes"              : CFG["classes"],
        "other_label"          : CFG["other_label"],
        "confidence_threshold" : CFG["confidence_threshold"],
        "vision_weight"        : CFG["vision_weight"],
        "text_weight"          : CFG["text_weight"],
        "img_size"             : CFG["img_size"],
    }
    with open(path_cfg, "w") as f:
        json.dump(inference_cfg, f, indent=2)
    log.info(f"  💾 Config inférence sauvegardée : {path_cfg}")


# ─────────────────────────────────────────────
#  10. MAIN
# ─────────────────────────────────────────────
def main():
    log.info("\n" + "╔" + "═" * 53 + "╗")
    log.info("║  Classification Chèque / RIB — Pipeline Hybride   ║")
    log.info("╚" + "═" * 53 + "╝\n")

    # ── Chargement et augmentation ──
    log.info("📁 Étape 1 : Chargement et augmentation du dataset")
    all_paths = build_augmented_dataset(CFG["dataset_dir"], CFG["target_per_class"])

    paths  = []
    labels = []
    for cls_idx, cls in enumerate(CFG["classes"]):
        paths  += all_paths[cls]
        labels += [cls_idx] * len(all_paths[cls])

    log.info(f"   Total : {len(paths)} images")

    # ── Split ──
    log.info("\n📊 Étape 2 : Split Train / Test (80/20 stratifié)")
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels,
        test_size=0.2,
        stratify=labels,
        random_state=CFG["seed"]
    )
    log.info(f"   Train : {len(train_paths)}  |  Test : {len(test_paths)}")
    plot_distribution(train_labels, test_labels)

    # ── Module Vision ──
    log.info("\n🖼️  Étape 3 : Entraînement du modèle Vision")
    vision_model, history = train_vision_model(
        train_paths, train_labels,
        test_paths,  test_labels
    )
    plot_learning_curves(history)

    # ── Module Texte ──
    log.info("\n📝 Étape 4 : Entraînement du modèle Texte")
    text_model, _, _ = train_text_model(
        train_paths, train_labels,
        test_paths,  test_labels
    )

    # ── Ensemble ──
    log.info("\n🔀 Étape 5 : Évaluation de l'ensemble")
    preds, combined_probs = evaluate_ensemble(
        vision_model, text_model, test_paths, test_labels
    )
    plot_confusion_matrix(test_labels, preds)

    # ── Métriques finales ──
    acc  = accuracy_score(test_labels, preds)
    f1   = f1_score(test_labels, preds, average="weighted")
    prec = precision_score(test_labels, preds, average="weighted")
    rec  = recall_score(test_labels, preds, average="weighted")

    metrics = {
        "accuracy" : acc,  "precision": prec,
        "recall"   : rec,  "f1"       : f1,
        "train_size": len(train_paths),
        "test_size" : len(test_paths),
        "epochs_trained": len(history["train_acc"]),
        "best_vision_val_acc": max(history["val_acc"]),
    }

    log.info("\n" + "═" * 60)
    log.info("            📊 RÉCAPITULATIF FINAL DES PERFORMANCES")
    log.info("═" * 60)
    log.info(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    log.info(f"  Precision : {prec:.4f}  ({prec*100:.1f}%)")
    log.info(f"  Recall    : {rec:.4f}  ({rec*100:.1f}%)")
    log.info(f"  F1-Score  : {f1:.4f}  ({f1*100:.1f}%)")
    log.info("═" * 60)

    if f1 >= 0.90:
        log.info("  🏆 Excellentes performances !")
    elif f1 >= 0.75:
        log.info("  ✅ Bonnes performances.")
    else:
        log.info("  ⚠️  Ajoutez plus de données pour améliorer.")

    # ── Sauvegarde ──
    save_results(metrics)
    log.info(f"\n✅ Tout est sauvegardé dans : {CFG['output_dir']}/")
    log.info("   → Lancez predict.py pour classer de nouvelles images\n")


if __name__ == "__main__":
    main()