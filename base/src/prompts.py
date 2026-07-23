"""
prompts.py — Napoleon LLM Prompts
===================================
Two calls per consultation:
  1. build_dpi_prompt  → returns enriched DPI JSON (DPI.json schema)
  2. build_cr_prompt   → returns CR JSON (cr_modele.json schema)
"""

import json


def build_dpi_prompt(transcript: str, existing_dpi: dict | None = None) -> str:
    """
    Call 1: extract consultation data from the transcript into the agreed schema.
    If existing_dpi is provided, enrich it; otherwise build from scratch.
    """

    dpi_schema = {
        "motif_de_consultation": "string ou null",
        "antecedents": {
            "medicaux": ["liste de strings ou tableau vide"],
            "chirurgicaux": ["liste de strings ou tableau vide"],
            "gynecologiques": ["liste de strings ou tableau vide"],
            "familiaux": ["liste de strings ou tableau vide"]
        },
        "mode_de_vie": {
            "tabac": "string ou null",
            "alcool": "string ou null",
            "drogues": "string ou null",
            "activite_physique": "string ou null",
            "voyages_recents": "string ou null",
            "autre": "string ou null"
        },
        "vaccins": [],
        "traitements_habituels": [
            {
                "nom_commercial": "string",
                "molecule": "string ou null",
                "posologie": "string ou null"
            }
        ],
        "allergies": ["liste de strings ou tableau vide"],
        "interrogatoire": {
            "symptomes_generaux": {
                "description": "Uniquement : asthénie, perte de poids, fièvre, anorexie. Null si aucun de ces symptômes n'est mentionné.",
                "asthenie": "string ou null",
                "perte_de_poids": "string ou null",
                "fievre": "string ou null",
                "anorexie": "string ou null"
            },
            "symptomes_par_organe": [
                {
                    "organe": "string — nom de l'organe ou système concerné",
                    "symptomes": "string — description des symptômes",
                    "date_debut": "string ou null",
                    "intensite": "string ou null",
                    "evolution": "string ou null",
                    "traitements_testes": "string ou null",
                    "examens_complementaires_realises": "string ou null"
                }
            ],
            "autres": ["liste de strings — éléments évoqués brièvement non rattachés à un organe, ou tableau vide"]
        },
        "examen_clinique": {
            "constantes": {
                "poids_kg": "number ou null",
                "taille_cm": "number ou null",
                "imc": "number ou null",
                "pression_arterielle": "string ou null",
                "frequence_cardiaque": "number ou null",
                "temperature": "number ou null",
                "spo2": "number ou null",
                "frequence_respiratoire": "number ou null"
            },
            "examen_specifique": "string ou null"
        },
        "conclusion": {
            "points_cles": ["liste de strings — éléments importants issus de l'interrogatoire et de l'examen clinique"],
            "diagnostic": ["liste de strings — diagnostic(s) probable(s) nouveau(x) issus de cette consultation"],
            "proposition_therapeutique": {
                "medicaments": ["liste de strings ou tableau vide"],
                "paramedical": ["liste de strings ou tableau vide"],
                "examens_complementaires": ["liste de strings ou tableau vide"],
                "orientation": "string ou null"
            },
            "prochaine_consultation": "string ou null"
        }
    }

    if existing_dpi:
        dpi_context = f"""
Le DPI existant du patient est fourni ci-dessous. Enrichis-le avec les informations
extraites de la transcription. Conserve toutes les données existantes et ajoute ou
mets à jour uniquement ce qui est mentionné dans la transcription.

DPI EXISTANT :
{json.dumps(existing_dpi, ensure_ascii=False, indent=2)}
"""
    else:
        dpi_context = """
Aucun DPI existant. Construis le DPI complet à partir des informations extraites
de la transcription. Les champs non mentionnés doivent être null ou tableau vide.
"""

    return f"""Tu es un assistant médical expert en extraction d'informations cliniques.
{dpi_context}

TRANSCRIPTION DE LA CONSULTATION :
{transcript}

INSTRUCTIONS :
- Extrais toutes les informations médicales présentes dans la transcription.
- symptomes_generaux : uniquement asthénie, perte de poids, fièvre, anorexie.
  Ne pas y mettre d'autres symptômes même s'ils sont mentionnés.
- symptomes_par_organe : un item par organe/système. Les examens complémentaires
  réalisés sont rattachés à l'organe concerné dans examens_complementaires_realises.
- autres : éléments évoqués brièvement non rattachables à un organe spécifique.
- conclusion.points_cles : synthèse des éléments importants de l'interrogatoire et
  de l'examen clinique.
- conclusion.diagnostic : uniquement les nouveaux diagnostics émergents de cette consultation.
- conclusion.proposition_therapeutique : réponse directe à chaque point clé, séparée
  en médicaments, paramédical, examens complémentaires, orientation.
- Les champs non mentionnés dans la transcription doivent être null ou [], jamais inventés.
- Réponds UNIQUEMENT avec le JSON valide, sans texte avant ni après, sans markdown.

SCHÉMA À RESPECTER :
{json.dumps(dpi_schema, ensure_ascii=False, indent=2)}
"""


def build_error_correction_prompt(kyutai_transcript: str) -> str:
    """
    Error correction call: fixes Kyutai streaming STT errors.
    Runs immediately after Kyutai transcription to catch medical term misses,
    medication names, and dosage errors before medical review.
    Returns a corrected transcript and confidence metrics.
    """

    correction_schema = {
        "transcript_corrigee": "string — la transcription complète avec les corrections appliquées",
        "corrections_appliquees": [
            {
                "original": "string — le mot ou groupe de mots tel que transcrit par Kyutai",
                "corrige": "string — la correction proposée",
                "type": "string — medicament | dosage | anatomie | numero | autre",
                "confiance": "string — haute | moyenne | faible",
                "raison": "string — pourquoi cette correction a été appliquée"
            }
        ],
        "qualite_transcription": "string — excellente | bonne | acceptable | mauvaise"
    }

    return f"""Tu es un assistant médical expert en correction de transcriptions STT (speech-to-text).

TRANSCRIPTION BRUTE DE KYUTAI :
{kyutai_transcript}

INSTRUCTIONS :
- Corrige les erreurs courantes de STT streaming, notamment :
  • Noms de médicaments mal reconnus (ex: "Doliprane" → "Doliprane", "Advil" → "Ibuprofène")
  • Dosages mal transcrits (ex: "250 milligrammes" → "250 mg")
  • Termes anatomiques déformés (ex: "cardia" → "cardiaque")
  • Chiffres mal reconnus (ex: "mille deux cents" → "1200")
  • Hésitations et faux démarrages (ex: "euh le patient a euh... le patient a mal au genou" → "le patient a mal au genou")
- Conserve l'intention et la structure originale.
- N'invente PAS d'informations médicales qui n'étaient pas présentes.
- Si la transcription semble déjà correcte, retourne corrections_appliquees=[].
- Évalue la qualité globale de la transcription (excellente/bonne/acceptable/mauvaise).
- Réponds UNIQUEMENT avec le JSON valide, sans texte avant ni après, sans markdown.

SCHÉMA À RETOURNER :
{json.dumps(correction_schema, ensure_ascii=False, indent=2)}
"""


def build_review_prompt(transcript: str) -> str:
    """
    Medical review call: checks the transcript for transcription errors,
    mis-heard medication names, wrong dosages, and anatomical term mistakes.
    Returns a structured JSON with corrections and alerts.
    """

    review_schema = {
        "resume": "string — une phrase résumant l'état général de la transcription",
        "corrections": [
            {
                "original": "string — le mot ou groupe de mots tel que transcrit",
                "corrige": "string — la correction proposée",
                "type": "string — medicament | dosage | anatomie | autre",
                "confiance": "string — haute | moyenne | faible",
                "explication": "string — pourquoi cette correction est proposée"
            }
        ],
        "alertes": [
            {
                "texte": "string — le passage concerné",
                "raison": "string — pourquoi ce passage mérite attention du médecin"
            }
        ],
        "transcription_corrigee": "string — la transcription complète avec les corrections appliquées"
    }

    return f"""Tu es un assistant médical expert en relecture de transcriptions de consultations médicales.

TRANSCRIPTION À VÉRIFIER :
{transcript}

INSTRUCTIONS :
- Vérifie les noms de médicaments : sont-ils orthographiés correctement ?
  Un nom mal transcrit peut indiquer un médicament différent (ex: "Doliprane" vs "Dolipranne").
- Vérifie les dosages : sont-ils cohérents et plausibles pour les médicaments mentionnés ?
- Vérifie les termes anatomiques et médicaux : sont-ils correctement transcrits ?
- Signale dans "alertes" tout passage ambigu que le médecin devrait relire, même si tu n'es
  pas certain qu'il y ait une erreur.
- Si la transcription semble correcte, retourne corrections=[] et alertes=[].
- Ne corrige PAS le style ou la grammaire — uniquement les erreurs médicales potentielles.
- Réponds UNIQUEMENT avec le JSON valide, sans texte avant ni après, sans markdown.

SCHÉMA À RETOURNER :
{json.dumps(review_schema, ensure_ascii=False, indent=2)}
"""


def build_cr_prompt(transcript: str, enriched_dpi: dict) -> str:
    """
    Call 2: generate CR textuels and prescription lines.
    Receives the transcript + the already-enriched DPI for context.
    Returns cr_modele.json format.
    """

    cr_schema = {
        "dpi_textuel": (
            "string — aperçu synthétique du dossier patient en format structuré, "
            "sans phrases, uniquement les éléments importants. "
            "Format attendu (n'inclure que les sections non vides) :\n"
            "Antécédents\n- [antécédent 1]\n- [antécédent 2]\n\n"
            "Traitements habituels\n- [médicament posologie]\n\n"
            "Allergies : [liste ou 0 si aucune]\n\n"
            "Tabac : [quantité ou non-fumeur si mentionné]\n"
            "Alcool : [quantité si mentionné]\n"
            "Activité physique : [description si mentionnée]\n\n"
            "Null si le DPI ne contient aucune information pertinente."
        ),
        "cr_textuel": (
            "string — compte-rendu textuel complet de cette consultation : "
            "motif, interrogatoire, examen clinique, diagnostic et proposition "
            "thérapeutique. Rédigé en français médical comme un vrai CR."
        ),
        "prescription_lines": [
            "string — une ligne par médicament prescrit lors de cette consultation. "
            "Format : 'NomCommercial (DCI) — posologie — fréquence — durée'. "
            "Liste vide [] si aucune prescription."
        ]
    }

    return f"""Tu es un assistant médical expert en rédaction de comptes-rendus cliniques.

TRANSCRIPTION DE LA CONSULTATION :
{transcript}

DPI ENRICHI DU PATIENT (pour contexte) :
{json.dumps(enriched_dpi, ensure_ascii=False, indent=2)}

INSTRUCTIONS :
- dpi_textuel : aperçu rapide du dossier, sans phrases. Sections à inclure si non vides :
    "Antécédents" avec une puce par antécédent médical/chirurgical/familial,
    "Traitements habituels" avec une puce par médicament et sa posologie,
    "Allergies : X" (ou "Allergies : 0" si aucune allergie connue),
    "Tabac / Alcool / Activité physique" si mentionnés.
  Ne pas faire de phrases. Ne pas inclure le motif de consultation ni l'interrogatoire.
- cr_textuel : compte-rendu complet de cette consultation en français médical.
  Inclure motif, interrogatoire, examen clinique, diagnostic, proposition thérapeutique.
  Rédige comme un médecin dans un dossier patient.
- prescription_lines : une entrée par médicament prescrit lors de cette consultation.
  Format : "NomCommercial (DCI) — posologie — fréquence — durée".
  Liste vide [] si aucune prescription.
- Réponds UNIQUEMENT avec le JSON valide, sans texte avant ni après, sans markdown.

SCHÉMA À RETOURNER :
{json.dumps(cr_schema, ensure_ascii=False, indent=2)}
"""


def build_diarization_prompt(transcript: str) -> str:
    """
    LLM-based speaker diarization for a 2-speaker medical consultation.
    Only returns segments list — labeled_transcript is built in Python.
    """

    schema = {
        "segments": [
            {
                "speaker": "Medecin | Patient",
                "text": "the exact sentence or phrase as it appeared in the transcript"
            }
        ]
    }

    return f"""Tu es un assistant medical expert en analyse de consultations medicales.

TRANSCRIPTION DE LA CONSULTATION :
{transcript}

INSTRUCTIONS :
- Il y a exactement 2 interlocuteurs : le Medecin et le Patient.
- Decoupe en repliques et attribue chaque replique au bon interlocuteur.
- Le Medecin : pose des questions, prescrit, diagnostique, vocabulaire medical.
- Le Patient : decrit symptomes, repond, dit oui/non/d accord.
- Ne modifie PAS le texte des repliques, copie-les exactement.
- Reponds UNIQUEMENT avec du JSON valide, sans texte supplementaire.

SCHEMA :
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""
