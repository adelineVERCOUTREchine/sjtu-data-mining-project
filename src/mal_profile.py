"""Import des listes MAL, par OAuth, pseudo public ou export XML."""

import gzip
import io
import secrets
import time
import zlib
from threading import Lock
from urllib.parse import quote, urlencode
from xml.etree.ElementTree import ParseError

import requests
from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

import config


class ProfileError(Exception):
    pass


class OAuthCallbacks:
    """Le nouvel onglet dépose le code ; la session d'origine conserve le vérificateur PKCE."""

    def __init__(self):
        self.pending = {}
        self.lock = Lock()

    def start(self):
        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        with self.lock:
            now = time.monotonic()
            self.pending = {k: v for k, v in self.pending.items() if v[0] > now}
            self.pending[state] = (now + 300, None)
        params = dict(response_type="code", client_id=config.MAL_CLIENT_ID,
                      state=state, redirect_uri=config.MAL_REDIRECT_URI,
                      code_challenge=verifier, code_challenge_method="plain")
        return state, verifier, "https://myanimelist.net/v1/oauth2/authorize?" + urlencode(params)

    def receive(self, state, code, error):
        with self.lock:
            entry = self.pending.get(state)
            if not entry or entry[0] <= time.monotonic() or entry[1] is not None:
                return False
            self.pending[state] = (entry[0], (code if not error else "", bool(error)))
            return True

    def take(self, state):
        with self.lock:
            expiry, reply = self.pending.get(state, (0, None))
            if expiry <= time.monotonic():
                self.pending.pop(state, None)
                raise ProfileError("Sign-in expired. Start a new connection.")
            if reply is not None:
                del self.pending[state]
            return reply

    def cancel(self, state):
        with self.lock:
            self.pending.pop(state, None)


def api_json(method, url, **kwargs):
    try:
        response = requests.request(method, url, timeout=(5, 20), allow_redirects=False, **kwargs)
        if response.status_code in (401, 403):
            raise ProfileError("MAL denied access. Check the app credentials or import your XML export.")
        if response.status_code == 404:
            raise ProfileError("MAL could not find this list. Check the username or import an XML export.")
        if response.status_code == 429:
            raise ProfileError("MAL is receiving too many requests. Try again later.")
        if response.status_code != 200:
            raise ProfileError(f"MAL returned HTTP {response.status_code}. Please retry or import an export.")
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise ProfileError("Could not read a response from MyAnimeList. Try again or import an export.") from None


def download_list(username=None, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {"X-MAL-CLIENT-ID": config.MAL_CLIENT_ID}
    user = "@me" if token else quote(username.strip(), safe="")
    url = f"{config.MAL_API_URL}/users/{user}/animelist"
    entries = {}
    offset = 0
    while True:
        page = api_json("GET", url, headers=headers, params={
            "fields": "list_status", "limit": 1000, "offset": offset, "nsfw": "true"})
        for row in page["data"]:
            status = row["list_status"]
            entries[int(row["node"]["id"])] = (status["status"], int(status["score"]))
        if not page.get("paging", {}).get("next"):
            return entries
        if not page["data"]:
            raise ProfileError("MAL returned an incomplete list. Please try again.")
        offset += len(page["data"])


def import_account(code, verifier):
    data = dict(client_id=config.MAL_CLIENT_ID, code=code, code_verifier=verifier,
                grant_type="authorization_code", redirect_uri=config.MAL_REDIRECT_URI)
    if config.MAL_CLIENT_SECRET:
        data["client_secret"] = config.MAL_CLIENT_SECRET
    token = api_json("POST", "https://myanimelist.net/v1/oauth2/token", data=data).get("access_token")
    if not token:
        raise ProfileError("MAL did not issue an access token. Reconnect your account.")
    return download_list(token=token)


def read_export(raw):
    limit = 32 * 1024 * 1024
    try:
        if len(raw) > limit:
            raise ProfileError("Choose an export smaller than 32 MiB.")
        if raw.startswith(b"\x1f\x8b"):
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as archive:
                raw = archive.read(limit + 1)
        if len(raw) > limit:
            raise ProfileError("The uncompressed export exceeds 32 MiB.")
        root = ElementTree.fromstring(raw, forbid_dtd=True)
        if root.tag != "myanimelist" or root.find("manga") is not None:
            raise ProfileError("Choose a MyAnimeList anime list export.")
        entries = {}
        for item in root.findall("anime"):
            status = item.findtext("my_status", "").strip().lower().replace(" ", "_").replace("-", "_")
            entries[int(item.findtext("series_animedb_id", ""))] = (status, int(item.findtext("my_score") or 0))
        return entries
    except ProfileError:
        raise
    except (ValueError, ParseError, DefusedXmlException, OSError, EOFError, zlib.error):
        raise ProfileError("Could not read the anime list. Choose the original MAL XML or XML.GZ export.") from None


WATCHED = {"watching", "completed", "on_hold", "dropped"}


def positif(status, score):
    return status in WATCHED and status != "dropped" and not 1 <= score <= 4


def model_profile(entries, ids):
    positives, seen = [], []
    for i, mal_id in enumerate(ids):
        status, score = entries.get(int(mal_id), ("", 0))
        if status in WATCHED:
            seen.append(i)
            if positif(status, score):
                positives.append(i)
    return positives, seen


def rated_series_profile(entries, catalogue):
    """Une note moyenne par série, avec un repli sur les vues pour les listes sans notes."""
    from collections import defaultdict
    from statistics import mean

    indices = {ident: i for i, ident in enumerate(catalogue.ids)}
    saisons = defaultdict(list)
    notes = defaultdict(list)
    vues = set()
    for ident, (statut, note) in entries.items():
        groupe = catalogue.groupe(ident)
        if groupe not in catalogue.entrees or statut not in WATCHED:
            continue
        vues.add(groupe)
        if statut == "dropped":
            continue
        indices_saisons = saisons[groupe]
        if ident in indices and positif(statut, note):
            indices_saisons.append(indices[ident])
        if 1 <= note <= 10:
            notes[groupe].append(note)
    moyennes = {g: mean(valeurs) for g, valeurs in notes.items()}
    seuil = max(6, mean(moyennes.values())) if moyennes else None
    preferences = ({g: (note - 5) / 5 for g, note in moyennes.items() if note >= seuil}
                   if moyennes else dict.fromkeys(saisons, 1.0))
    selection = [i for g in preferences for i in (saisons[g] or [catalogue.entrees[g]])]
    return selection, vues, preferences, seuil
