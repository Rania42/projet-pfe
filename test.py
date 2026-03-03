import os
from pathlib import Path

print("=== VÉRIFICATION DU PROJET ===\n")

# Vérifier les dossiers
dossiers = ['templates', 'static/css', 'static/logos', 'generateurs', 'output']
for d in dossiers:
    if Path(d).exists():
        print(f"✅ {d}")
    else:
        print(f"❌ {d} (manquant)")

# Vérifier les fichiers
fichiers = [
    'templates/cheque.html',
    'static/css/cheque.css',
    'generateurs/generateur_cheque.py',
    'modele/classifier.py'
]
for f in fichiers:
    if Path(f).exists():
        print(f"✅ {f}")
    else:
        print(f"❌ {f} (manquant)")

print("\n=== COMMANDES ===\n")
print("1. Générer des images:")
print("   python generateurs/generateur_cheque.py")
print("\n2. Entraîner le modèle:")
print("   python modele/classifier.py train")
print("\n3. Tester avec vos images:")
print("   python modele/classifier.py")