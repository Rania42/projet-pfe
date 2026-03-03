** OCR Document Classifier
Ce projet permet de classifier automatiquement des documents (facture, RIB, chèque, etc.) à partir d’une image.
Le système repose sur une combinaison de :
-OCR (extraction de texte depuis l’image)
-Règles intelligentes (mots-clés pondérés + fuzzy matching)
👉 Cette version est une première implémentation (V1) visant à valider l’approche et tester les performances.

** Comment lancer le projet ?
Pour tester l'application, ouvrez votre terminal et suivez ces étapes :

Accédez au dossier du projet :
cd mon_projet_ocr

Installez les dépendances :
pip install -r requirement.txt

Lancez l'application :
python app.py

Ouvrez votre navigateur à l'adresse : http://127.0.0.1:5000

** Comment ça marche ? (Le Workflow)
Le système repose sur un pipeline de traitement en 4 étapes clés :

1. Prétraitement de l'image
Avant l'analyse, l'image est nettoyée pour maximiser la précision de la lecture :

Contraste (CLAHE) : Pour faire ressortir les caractères.

Réduction du bruit : Utilisation d'un filtre bilatéral pour lisser l'image sans flouter les bords des lettres.

Binarisation : Transformation en noir et blanc pur (Adaptive Threshold).

2. Extraction OCR
Utilisation de Pytesseract configuré pour une reconnaissance bilingue (Français + Arabe). L'image devient alors un bloc de texte brut exploitable.

3. Nettoyage & Normalisation
Pour éviter les erreurs de frappe ou de lecture, le texte est "normalisé" :

Passage en minuscules.

Suppression des accents.

Nettoyage des caractères spéciaux tout en préservant l'arabe et le latin.

4. Classification par Scoring (Le Cœur du Système)
L'application compare le texte extrait avec un dictionnaire de mots-clés pondérés (chaque mot a un "poids" ou une importance) :

Matching Exact : Recherche de mots précis.

Fuzzy Matching (RapidFuzz) : Reconnaissance des mots même s'il manque une lettre ou s'il y a une petite erreur de lecture OCR.

Score Final : Le type de document qui obtient le score le plus élevé est retenu.

** Structure du Projet
app.py : Le serveur Flask (interface web).

classifier.py : Logique de traitement d'image et de calcul de score.

dictionary.py : Base de connaissances contenant les mots-clés par type de document.

uploads/ : Stockage temporaire des fichiers analysés.

** Perspectives d'évolution
Pour franchir un palier supplémentaire en termes de performance et de polyvalence, les prochaines étapes incluent :

->Intelligence Sémantique (LLM Local) : Intégration d'Ollama pour combiner le système de scoring actuel avec une compréhension contextuelle profonde. Cela permettra de lever les ambiguïtés sur des documents très similaires.

->Optimisation pour Documents Volumineux : Mise en place d'une stratégie d'extraction rapide pour les fichiers complexes (PDF multi-pages, Word, Excel). L'idée est de cibler prioritairement les en-têtes, les titres principaux ou les métadonnées clés pour classifier le document instantanément sans avoir à analyser l'intégralité des pages.

->Fusion Multimodale : Combiner l'analyse de la mise en page (layout) et l'analyse textuelle pour atteindre une précision maximale, même sur des documents administratifs denses.
Développé avec passion pour simplifier la gestion documentaire. 💡
