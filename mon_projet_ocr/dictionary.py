"""
DICTIONNAIRE OPTIMISÉ - Stratégie "Super-Poids" et phrases composées
Version autonome compatible avec classifier.py existant
"""

# =========================
# FAMILLES DE DOCUMENTS
# =========================
DOC_FAMILIES = {
    "Identité": ["PIECE_IDENTITE", "ACTE_NAISSANCE", "ACTE_DECES", "ACTE_HEREDITE", "CERTIFICAT_VIE"],
    "Médical": ["CERTIFICAT_MEDICAL", "CERTIFICAT_INVALIDITE", "RAPPORT_MEDICAL", "CERTIFICAT_HOSPITALISATION"],
    "Financier": ["RIB", "RELEVE_COMPTE", "ATTESTATION_SOLDE"],
    "Paiement": ["CHEQUE", "LETTRE_CHANGE", "BAO", "QUITTANCE"],
    "Facture/Devis": ["FACTURE", "DEVIS", "FACTURE_PROFORMA"],
    "Contrat": ["CONTRAT_ASSURANCE", "CONTRAT_GARANTIE", "AVENANT", "TABLEAU_AMORTISSEMENT"],
    "Juridique": ["PROCURATION", "HYPOTHEQUE", "SURETE_MOBILIERE", "CAUTION", "DOCUMENT_JURIDIQUE"],
    "Crédit": ["DOCUMENT_PDV", "REALISATION_GARANTIE"]
}

# =========================
# DICTIONNAIRE PRINCIPAL - SUPER-POIDS
# =========================
KEYWORDS = {

    # =========================
    # IDENTITÉ - Documents officiels
    # =========================
    "PIECE_IDENTITE": {
        # Phrases composées (SUPER-POIDS)
        "carte nationale d'identite": 18,
        "carte nationale d identite": 18,
        "carte d'identite nationale": 17,
        "royaume du maroc carte nationale": 16,
        "cin n°": 15,
        "carte d'identite": 12,
        
        # Mots-clés simples (poids réduits)
        "cin": 8,
        "cni": 8,
        "identite": 4,
        "المملكة": 6,
        "البطاقة الوطنية": 15,
        "البطاقة": 5,
        "الهوية": 5
    },

    "ACTE_NAISSANCE": {
        # Phrases composées (SUPER-POIDS)
        "acte de naissance": 20,
        "extrait de naissance": 18,
        "copie integrale de naissance": 17,
        "acte d'etat civil naissance": 16,
        "registre des actes de naissance": 18,
        "officier d'etat civil naissance": 16,
        "ne le": 10,
        
        # Mots-clés simples
        "naissance": 6,
        "ازدياد": 12,
        "رسم الولادة": 18,
        "الولادة": 8,
        "نسخة موجزة": 12
    },

    "ACTE_DECES": {
        # Phrases composées (SUPER-POIDS)
        "acte de deces": 20,
        "certificat de deces": 20,
        "extrait de deces": 18,
        "acte d'etat civil deces": 18,
        "registre des actes de deces": 20,
        "registre des actes deces": 19,
        "acte deces": 15,
        "certificat deces": 15,
        "cause du deces": 12,
        
        # Mots-clés simples
        "deces": 8,
        "defunt": 8,
        "decede": 8,
        "وفاة": 15,
        "رسم الوفاة": 18,
        "شهادة الوفاة": 18
    },

    "ACTE_HEREDITE": {
        # Phrases composées
        "acte d'heredite": 20,
        "attestation d'heredite": 20,
        "acte de succession": 18,
        "certificat d'heredite": 18,
        "ordre de succession": 15,
        "part successorale": 14,
        
        # Mots-clés simples
        "heredite": 10,
        "heritiers": 10,
        "succession": 8,
        "ارث": 15,
        "الورثة": 12
    },

    "CERTIFICAT_VIE": {
        # Phrases composées
        "certificat de vie": 20,
        "certificat de survie": 18,
        "attestation d'existence": 18,
        "attestation de vie": 17,
        "certificat de vie collective": 16,
        
        # Mots-clés simples
        "شهادة الحياة": 18,
        "على قيد الحياة": 15,
        "vie": 3  # Poids très faible car trop générique
    },

    # =========================
    # MÉDICAL
    # =========================
    "CERTIFICAT_MEDICAL": {
        # Phrases composées
        "certificat medical": 18,
        "certificat d'aptitude": 16,
        "certificat de visite medicale": 17,
        "certificat medical d'aptitude": 17,
        "certificat de non contre indication": 16,
        "certificat de contre indication": 15,
        
        # Mots-clés simples
        "docteur": 5,
        "medecin": 6,
        "patient": 4,
        "diagnostic": 6,
        "examen clinique": 8
    },

    "CERTIFICAT_INVALIDITE": {
        # Phrases composées
        "certificat d'invalidite": 20,
        "certificat d'incapacite": 18,
        "taux d'invalidite": 18,
        "incapacite permanente": 16,
        "certificat medical d'invalidite": 19,
        
        # Mots-clés simples
        "invalidite": 10,
        "incapacite": 8,
        "handicap": 20
    },

    "RAPPORT_MEDICAL": {
        # Phrases composées
        "rapport medical": 18,
        "compte rendu medical": 18,
        "rapport d'hospitalisation": 16,
        "compte rendu d'hospitalisation": 17,
        "conclusion medicale": 14,
        
        # Mots-clés simples
        "operation": 5,
        "antecedents": 6,
        "traitement": 5
    },

    "CERTIFICAT_HOSPITALISATION": {
        # Phrases composées
        "certificat d'hospitalisation": 20,
        "certificat de sejour": 16,
        "certificat d'admission": 15,
        "certificat de sortie": 14,
        "frais d'hospitalisation": 15,
        
        # Mots-clés simples
        "hospitalisation": 8,
        "admission": 5,
        "hopital": 4
    },

    # =========================
    # FINANCIER
    # =========================
    "RIB": {
        # Phrases composées (SUPER-POIDS pour éviter confusion avec factures)
        "releve d'identite bancaire": 20,
        "releve d identite bancaire": 20,
        "attestation de rib": 18,
        "rib n°": 15,
        
        # Structure IBAN (très spécifique)
        "iban": 15,  # Un IBAN est exclusif au RIB
        "iban:": 16,
        "iban :": 16,
        
        # Mots-clés simples
        "rib": 10,
        "bic": 12,
        "swift": 12,
        "code banque": 8,
        "code guichet": 8,
        "numero de compte": 6,  # Faible car peut apparaître ailleurs
        "domiciliation": 7
    },

    "RELEVE_COMPTE": {
        # Phrases composées
        "releve de compte": 20,
        "extrait de compte": 20,
        "releve des operations": 16,
        "releve bancaire": 15,
        "mouvements du compte": 16,
        "detail des operations": 15,
        "situation du compte": 15,
        
        # Mots-clés simples
        "solde": 6,
        "debit": 4,
        "credit": 4,
        "operations": 4
    },

    "ATTESTATION_SOLDE": {
        # Phrases composées
        "attestation de solde": 20,
        "certificat de solde": 18,
        "attestation de solde bancaire": 20,
        "certificat de solde bancaire": 19,
        "solde credit": 12,
        "solde debiteur": 12,
        
        # Mots-clés simples
        "attestation": 4  # Faible car trop générique
    },

    # =========================
    # PAIEMENT
    # =========================
    "CHEQUE": {
        # Phrases composées (très spécifiques aux chèques)
        "payez contre ce cheque": 25,
        "payez contre ce chèque": 25,
        "a l'ordre de": 20,
        "à l'ordre de": 20,
        "montant en lettres": 18,
        "montant en chiffres": 15,
        " cheque n°": 15,
        "chèque n°": 15,
        
        # Mots-clés simples
        "cheque": 8,
        "chèque": 8,
        "payez": 12,  # Très spécifique au chèque
        "ordre": 6,
        "signature": 4
    },

    "LETTRE_CHANGE": {
        # Phrases composées
        "lettre de change": 22,
        "lettre de change n°": 20,
        "effet de commerce": 18,
        "traite financiere": 16,
        "acceptation lettre de change": 17,
        
        # Mots-clés simples
        "traite": 10,
        "echeance": 8,
        "acceptation": 8,
        "tire": 6
    },

    "BAO": {
        # Phrases composées
        "bon d'achat": 20,
        "bon de commande": 18,
        "bon d'achat n°": 18,
        "bon de commande n°": 17,
        
        # Mots-clés simples
        "bao": 15,
        "achat": 4,
        "commande": 5
    },

    "QUITTANCE": {
        # Phrases composées
        "quittance de loyer": 22,
        "quittance de paiement": 20,
        "recu de paiement": 18,
        "quittance n°": 15,
        "dont quittance": 15,
        
        # Mots-clés simples
        "quittance": 10,
        "recu": 6,
        "paiement": 3  # Faible car trop générique
    },

    # =========================
    # FACTURE / DEVIS
    # =========================
    "FACTURE": {
        # Phrases composées (SUPER-POIDS pour dominer RIB et ICE)
        "facture n°": 25,
        "facture proforma": 20,
        "facture definitive": 20,
        "total ttc": 20,  # Très spécifique facture
        "total ht": 18,
        "tva 20%": 18,
        "tva 14%": 18,
        "tva 10%": 18,
        "tva 7%": 18,
        "numero de facture": 20,
        "date de facture": 18,
        "condition de paiement": 15,
        "mode de reglement": 15,
        
        # Mots-clés fiscaux marocains (très spécifiques)
        "identifiant commun de l'entreprise": 25,
        "identifiant commun des entreprises": 25,
        "ice:": 25,  # L'ICE est l'arme anti-confusion
        "ice :": 25,
        "rc:": 15,
        "rc :": 15,
        "patente:": 15,
        "cnss:": 14,
        "if:": 14,
        
        # Mots-clés simples
        "facture": 8,  # Poids réduit car trop court
        "tva": 8,
        "ttc": 10,
        "ht": 8
    },

    "FACTURE_PROFORMA": {
        # Phrases composées
        "facture proforma": 25,
        "proforma n°": 20,
        "devis facture": 18,
        "facture proformat": 18,
        "projet de facture": 18,
        
        # Mots-clés simples
        "proforma": 15
    },

    "DEVIS": {
        # Phrases composées
        "devis n°": 22,
        "devis descriptif": 18,
        "devis quantitatif": 18,
        "proposition commerciale": 18,
        "valable jusqu'au": 16,
        "sous reserve de": 15,
        
        # Mots-clés simples
        "devis": 10,
        "estimation": 8,
        "prix": 3  # Faible car générique
    },

    # =========================
    # CONTRAT
    # =========================
    "CONTRAT_ASSURANCE": {
        # Phrases composées
        "contrat d'assurance": 25,
        "police d'assurance": 23,
        "conditions generales d'assurance": 22,
        "conditions particulieres d'assurance": 22,
        "avenant au contrat d'assurance": 20,
        "certificat d'assurance": 18,
        
        # Mots-clés simples
        "assureur": 8,
        "assure": 8,
        "primes": 8,
        "franchise": 7
    },

    "CONTRAT_GARANTIE": {
        # Phrases composées
        "contrat de garantie": 25,
        "certificat de garantie": 22,
        "conditions de garantie": 20,
        "duree de garantie": 18,
        "objet de la garantie": 18,
        
        # Mots-clés simples
        "garantie": 5  # Poids réduit car trop générique
    },

    "AVENANT": {
        # Phrases composées
        "avenant n°": 25,
        "avenant au contrat": 25,
        "modification de contrat": 20,
        "rectificatif au contrat": 18,
        "additif au contrat": 18,
        
        # Mots-clés simples
        "avenant": 15
    },

    "TABLEAU_AMORTISSEMENT": {
        # Phrases composées
        "tableau d'amortissement": 28,  # SUPER-POIDS maximum
        "echeancier d'amortissement": 25,
        "plan d'amortissement": 22,
        "tableau de remboursement": 20,
        "echeancier de credits": 18,
        "mensualites": 15,
        
        # Mots-clés simples
        "amortissement": 12,
        "echeancier": 10,
        "capital restant": 12,
        "interets": 8
    },

    # =========================
    # JURIDIQUE
    # =========================
    "PROCURATION": {
        # Phrases composées
        "procuration n°": 22,
        "acte de procuration": 22,
        "mandat de representation": 20,
        "pouvoir special": 18,
        "procuration generale": 18,
        "mandant et mandataire": 20,
        
        # Mots-clés simples
        "procuration": 12,
        "mandant": 10,
        "mandataire": 10,
        "وكالة": 15
    },

    "HYPOTHEQUE": {
        # Phrases composées
        "contrat d'hypotheque": 25,
        "inscription hypothecaire": 23,
        "mainlevee d'hypotheque": 23,
        "certificat hypothecaire": 20,
        "rang hypothecaire": 18,
        "bien hypotheque": 16,
        
        # Mots-clés simples
        "hypotheque": 12,
        "hypothecaire": 10,
        "credit immobilier": 8
    },

    "SURETE_MOBILIERE": {
        # Phrases composées
        "surete mobiliere": 25,
        "contrat de nantissement": 23,
        "acte de gage": 22,
        "nantissement de fonds de commerce": 25,
        "gage sur vehicule": 20,
        
        # Mots-clés simples
        "surete": 12,
        "mobilier": 8,
        "nantissement": 15,
        "gage": 12
    },

    "CAUTION": {
        # Phrases composées
        "acte de caution": 25,
        "acte de cautionnement": 25,
        "caution solidaire": 22,
        "caution simple": 20,
        "engagement de caution": 22,
        
        # Mots-clés simples
        "caution": 12,
        "cautionnement": 15,
        "garant": 10
    },

    "DOCUMENT_JURIDIQUE": {
        # Phrases composées
        "tribunal de premiere instance": 20,
        "cour d'appel": 18,
        "jugement n°": 22,
        "arret de la cour": 20,
        "decision de justice": 18,
        "signification d'huissier": 20,
        
        # Mots-clés simples
        "tribunal": 8,
        "jugement": 10,
        "justice": 5,
        "droit": 4
    },

    # =========================
    # CRÉDIT
    # =========================
    "DOCUMENT_PDV": {
        # Phrases composées
        "point de vente": 12,
        "convention point de vente": 20,
        "contrat point de vente": 20,
        "depart volontaire": 22,
        "demande de credit point de vente": 20,
        
        # Mots-clés simples
        "pdv": 15
    },

    "REALISATION_GARANTIE": {
        # Phrases composées
        "realisation de garantie": 28,
        "mise en jeu de la garantie": 26,
        "appel en garantie": 24,
        "execution de garantie": 22,
        
        # Mots-clés simples
        "realisation": 8,
        "garantie": 5  # Poids faible car générique
    }
}

# =========================
# MOTIFS RÉGEX POUR DÉTECTION SUPPLÉMENTAIRE
# (Optionnel - peut être utilisé dans classifier.py)
# =========================
DOCUMENT_PATTERNS = {
    "RIB": [
        r'[a-zA-Z]{2}\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}',  # IBAN
        r'rib\s*[:\s]*\d{20,30}',
        r'iban\s*[:\s]*[a-zA-Z]{2}\d{2}'
    ],
    
    "ACTE_DECES": [
        r'acte?\s+de\s+décès',
        r'certificat\s+de\s+décès',
        r'registre\s+des\s+actes?\s+de\s+décès'
    ],
    
    "CHEQUE": [
        r'payez\s+contre\s+ce\s+chèque',
        r'à\s+l\'?\s*ordre\s+de'
    ],
    
    "FACTURE": [
        r'facture\s+n°\s*\d+',
        r'total\s+t\.?t\.?c\.?',
        r'identifiant\s+commun\s+de\s+l\'?\s*entreprise'
    ]
}