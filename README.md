📄 OCR Document Classifier (V1)
Ce projet propose une solution automatisée pour classifier des documents (factures, RIB, chèques, etc.) à partir d'images. Il combine l'extraction de texte et une analyse par scoring pour identifier le type de document avec précision.

✨ Points Forts
Traitement Hybride : Allie l'OCR bilingue (Français/Arabe) à une logique de pondération par mots-clés.

Robustesse : Utilise le Fuzzy Matching pour rester efficace malgré d'éventuelles erreurs de lecture OCR.

Confidentialité : Conçu pour garantir la sécurité et la protection des données traitées.

⚙️ Fonctionnement du Pipeline
Le système suit un flux de travail en quatre étapes clés :

Prétraitement : Amélioration de l'image (contraste, binarisation) pour une lecture optimale.

Extraction OCR : Conversion de l'image en texte brut via Pytesseract.

Normalisation : Nettoyage du texte (minuscules, suppression des accents).

Classification : Calcul d'un score final basé sur un dictionnaire de mots-clés et la similarité textuelle.

📂 Organisation du Projet
Le projet est structuré dans le dossier suivant :

cd mon_projet_ocr

Il s'articule autour des fichiers principaux :

app.py : Interface web Flask.

classifier.py : Cœur algorithmique de traitement et de classification.

dictionary.py : Base de connaissances des mots-clés.

🔮 Évolutions Futures
Intelligence Sémantique : Intégration de LLM locaux (Ollama).

Analyse de mise en page : Fusion multimodale (texte + layout) pour les documents complexes.
