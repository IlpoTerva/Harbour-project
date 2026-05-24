"""
i18n.py — driver-facing spoken prompt translations.

Add a new language by adding its ISO 639-1 code to SUPPORTED_LANGS and
providing translations for every key in _T.  No other file needs to change.
"""

from typing import Dict

SUPPORTED_LANGS: frozenset = frozenset({"en", "es", "fr", "pt"})
_FALLBACK = "en"

_T: Dict[str, Dict[str, str]] = {
    "plate_low_conf": {
        "en": "I detected the plate {plate} but I am not fully confident. Could you please confirm your licence plate out loud?",
        "es": "He detectado la matrícula {plate} pero no tengo total confianza. ¿Podría confirmar su matrícula en voz alta?",
        "fr": "J'ai détecté la plaque {plate} mais je ne suis pas totalement certain. Pourriez-vous confirmer votre plaque d'immatriculation à voix haute?",
        "pt": "Detetei a matrícula {plate} mas não tenho total certeza. Pode confirmar a sua matrícula em voz alta?",
    },
    "plate_not_read": {
        "en": "I could not read your licence plate. Please say it out loud now.",
        "es": "No he podido leer su matrícula. Por favor, dígala en voz alta ahora.",
        "fr": "Je n'ai pas pu lire votre plaque d'immatriculation. Veuillez la dire à voix haute maintenant.",
        "pt": "Não consegui ler a sua matrícula. Por favor, diga-a em voz alta agora.",
    },
    "confirm_plate": {
        "en": "I read your licence plate as {plate}. Is that correct? Please say yes or no.",
        "es": "He leído su matrícula como {plate}. ¿Es correcto? Por favor, diga sí o no.",
        "fr": "J'ai lu votre plaque comme {plate}. Est-ce correct? Veuillez dire oui ou non.",
        "pt": "Li a sua matrícula como {plate}. Está correto? Por favor, diga sim ou não.",
    },
    "request_name": {
        "en": "Please say your full name for verification.",
        "es": "Por favor, diga su nombre completo para la verificación.",
        "fr": "Veuillez dire votre nom complet pour la vérification.",
        "pt": "Por favor, diga o seu nome completo para verificação.",
    },
    "alert_not_in_db": {
        "en": "I'm sorry, but the plate {plate} is not registered for today. A gate worker has been alerted. Please wait.",
        "es": "Lo siento, pero la matrícula {plate} no está registrada para hoy. Se ha alertado a un agente. Por favor, espere.",
        "fr": "Je suis désolé, mais la plaque {plate} n'est pas enregistrée pour aujourd'hui. Un agent a été alerté. Veuillez patienter.",
        "pt": "Lamento, mas a matrícula {plate} não está registada para hoje. Um agente foi alertado. Por favor, aguarde.",
    },
    "alert_name_mismatch": {
        "en": "I'm sorry, but the name you provided does not match our records. A gate worker has been alerted. Please wait.",
        "es": "Lo siento, pero el nombre que ha proporcionado no coincide con nuestros registros. Se ha alertado a un agente. Por favor, espere.",
        "fr": "Je suis désolé, mais le nom que vous avez fourni ne correspond pas à nos dossiers. Un agent a été alerté. Veuillez patienter.",
        "pt": "Lamento, mas o nome que forneceu não corresponde aos nossos registos. Um agente foi alertado. Por favor, aguarde.",
    },
    "alert_generic": {
        "en": "Access denied. A gate worker has been alerted. Please wait.",
        "es": "Acceso denegado. Se ha alertado a un agente. Por favor, espere.",
        "fr": "Accès refusé. Un agent a été alerté. Veuillez patienter.",
        "pt": "Acesso negado. Um agente foi alertado. Por favor, aguarde.",
    },
    "access_granted": {
        "en": "Access granted. Welcome, {name}. Please proceed to {dock}. Your cargo is {cargo}. Your arrival window is {window}. Have a safe unloading.",
        "es": "Acceso concedido. Bienvenido, {name}. Por favor, diríjase a {dock}. Su carga es {cargo}. Su ventana de llegada es {window}. Que tenga una descarga segura.",
        "fr": "Accès accordé. Bienvenue, {name}. Veuillez vous diriger vers {dock}. Votre cargaison est {cargo}. Votre créneau d'arrivée est {window}. Bonne déchargement.",
        "pt": "Acesso concedido. Bem-vindo, {name}. Por favor, dirija-se a {dock}. A sua carga é {cargo}. A sua janela de chegada é {window}. Boa descarga.",
    },
}


def get(key: str, lang: str, **kwargs) -> str:
    """Return the localised prompt for `key` in `lang`, with placeholders filled.

    Falls back to English if `lang` is not in SUPPORTED_LANGS or the key is missing.
    """
    resolved = lang if lang in SUPPORTED_LANGS else _FALLBACK
    bucket = _T.get(key, {})
    text = bucket.get(resolved) or bucket.get(_FALLBACK, "")
    return text.format(**kwargs) if kwargs else text
