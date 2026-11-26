from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request, session

from data import SEEN_STATUSES, Catalog, UserLists, parse_mal_export

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("ANIME_DATA_DIR", BASE_DIR / "data"))
DATASET_CSV = Path(os.environ.get("ANIME_DATASET", DATA_DIR / "anime_dataset.csv"))
RESULTS_CSV = Path(os.environ.get("ANIME_RESULTS", DATA_DIR / "results.csv"))
# Max similar titles kept per anime in memory (0 = keep everything).
MAX_NEIGHBORS = int(os.environ.get("ANIME_MAX_NEIGHBORS", "200"))
# The FastAPI service that merges the score files and writes results.csv.
WEIGHTS_API_URL = os.environ.get("ANIME_WEIGHTS_API", "http://127.0.0.1:8000").rstrip("/")
API_TIMEOUT = int(os.environ.get("ANIME_WEIGHTS_API_TIMEOUT", "900"))

# One entry per field of the API's SettingsRequest (keep keys identical).
SCORE_WEIGHTS = [
    {"key": "users", "label": "Users",
     "help": "Taste overlap: how similarly people rated both anime."},
    {"key": "genres_tags", "label": "Genres and tags",
     "help": "Content overlap: how many genres and tags the two anime share."},
]
DEFAULT_WEIGHT = 0.5

app = Flask(__name__, instance_path=str(BASE_DIR / "instance"))
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
Path(app.instance_path).mkdir(parents=True, exist_ok=True)

# Persist the session secret so uploaded lists survive server restarts.
_key_file = Path(app.instance_path) / "secret_key"
if not _key_file.exists():
    _key_file.write_text(secrets.token_hex(32))
app.secret_key = _key_file.read_text().strip()
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.permanent_session_lifetime = 60 * 60 * 24 * 365

if not DATASET_CSV.exists():
    sys.exit(f"Missing data file: {DATASET_CSV}\nPut anime_dataset.csv in {DATA_DIR} (or set ANIME_DATASET).")
if not RESULTS_CSV.exists():
    print(f"Note: {RESULTS_CSV} doesn't exist yet. Open Settings and apply weights to generate it.", flush=True)

print("Loading data...", flush=True)
catalog = Catalog(DATASET_CSV, RESULTS_CSV, MAX_NEIGHBORS)
lists = UserLists(Path(app.instance_path) / "lists")
settings_file = Path(app.instance_path) / "settings.json"
_recalc_lock = threading.Lock()
print(f"Ready: {len(catalog.records):,} anime, {catalog.edge_count:,} similarity links.", flush=True)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def current_list() -> dict | None:
    return lists.load(session.get("list_id"))


def mine_lookup(user_list: dict | None):
    by_id = {e["id"]: e for e in (user_list or {}).get("entries", [])}

    def lookup(mal_id: int):
        e = by_id.get(mal_id)
        if not e:
            return None
        return {"status": e["status"], "score": e["score"], "seen": e["status"] in SEEN_STATUSES}

    return lookup, by_id


# --------------------------------------------------------------------------- #
# pages
# --------------------------------------------------------------------------- #
@app.get("/")
def popular():
    return render_template("popular.html")


@app.get("/my-list")
def my_list():
    return render_template("mylist.html")


@app.get("/settings")
def settings_page():
    return render_template("settings.html")


# --------------------------------------------------------------------------- #
# API: popularity page + similar anime
# --------------------------------------------------------------------------- #
@app.get("/api/popular")
def api_popular():
    page = max(1, request.args.get("page", 1, type=int))
    total, items = catalog.popular(page, 60, request.args.get("q", ""))
    return jsonify(total=total, page=page, has_more=page * 60 < total, items=items)


@app.get("/api/anime/<int:mal_id>")
def api_anime(mal_id: int):
    user_list = current_list()
    lookup, by_id = mine_lookup(user_list)

    if mal_id not in catalog.records and mal_id not in by_id:
        return jsonify(error="That anime isn't in the dataset."), 404

    anime = catalog.detail(mal_id, by_id.get(mal_id))
    anime["mine"] = lookup(mal_id)

    limit = min(max(request.args.get("limit", 60, type=int), 1), 200)
    similar = catalog.similar(mal_id, limit)
    for s in similar:
        s["mine"] = lookup(s["id"])
    return jsonify(anime=anime, similar=similar, has_list=user_list is not None)


# --------------------------------------------------------------------------- #
# API: my list
# --------------------------------------------------------------------------- #
@app.get("/api/my-list")
def api_my_list_get():
    user_list = current_list()
    if not user_list:
        return jsonify(loaded=False)

    items = []
    for e in user_list["entries"]:
        c = catalog.card(e["id"], e)
        c.update(my_score=e["score"], status=e["status"], watched=e["watched"],
                 in_dataset=e["id"] in catalog.records)
        items.append(c)
    items.sort(key=lambda c: (-c["my_score"],
                              c["popularity"] is None, c["popularity"] or 0, c["id"]))
    return jsonify(loaded=True, user=user_list["user"], entries=items)


@app.post("/api/my-list")
def api_my_list_upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="Choose your MyAnimeList export (.xml) first."), 400
    try:
        payload = parse_mal_export(f.read())
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    session.permanent = True
    session["list_id"] = lists.save(payload, session.get("list_id"))
    return jsonify(ok=True, count=len(payload["entries"]))


@app.delete("/api/my-list")
def api_my_list_delete():
    lists.delete(session.pop("list_id", None))
    return jsonify(ok=True)


@app.get("/api/my-list/recommendations")
def api_recommendations():
    user_list = current_list()
    if not user_list:
        return jsonify(error="Upload your list first."), 400
    min_score = min(max(request.args.get("min_score", 8, type=int), 5), 10)
    include_plan = request.args.get("include_plan", "1") != "0"
    limit = min(max(request.args.get("limit", 100, type=int), 1), 300)
    items = catalog.recommend(user_list["entries"], min_score, include_plan, limit)
    return jsonify(items=items, min_score=min_score)


# --------------------------------------------------------------------------- #
# API: similarity settings (weights -> FastAPI -> results.csv -> reload)
# --------------------------------------------------------------------------- #
def read_saved_settings() -> dict | None:
    try:
        return json.loads(settings_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def call_weights_api(weights: dict) -> dict:
    # include_data=false is ignored by the API as written; if it supports it, we skip a huge payload.
    req = urllib.request.Request(
        f"{WEIGHTS_API_URL}/settings?include_data=false",
        data=json.dumps(weights).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
        return json.loads(resp.read())


@app.get("/api/settings")
def api_settings_get():
    saved = read_saved_settings()
    weights = {w["key"]: DEFAULT_WEIGHT for w in SCORE_WEIGHTS}
    if saved:
        weights.update(saved.get("weights", {}))
    return jsonify(scores=SCORE_WEIGHTS, weights=weights, last=saved,
                   links=catalog.edge_count, api_url=WEIGHTS_API_URL)


@app.post("/api/settings")
def api_settings_post():
    body = request.get_json(silent=True) or {}
    weights = {}
    for w in SCORE_WEIGHTS:
        try:
            v = float(body.get(w["key"], DEFAULT_WEIGHT))
        except (TypeError, ValueError):
            return jsonify(error=f"Weight for {w['label']} must be a number."), 400
        if not 0 <= v <= 1000:
            return jsonify(error=f"Weight for {w['label']} must be 0 or more."), 400
        weights[w["key"]] = v
    if sum(weights.values()) <= 0:
        return jsonify(error="Give at least one score a weight above zero."), 400

    if not _recalc_lock.acquire(blocking=False):
        return jsonify(error="A recalculation is already running. Wait for it to finish."), 409
    try:
        before = RESULTS_CSV.stat().st_mtime if RESULTS_CSV.exists() else None
        try:
            result = call_weights_api(weights)
        except urllib.error.HTTPError as exc:
            return jsonify(error=f"The weights API answered {exc.code}: {exc.reason}."), 502
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as exc:
            return jsonify(error=f"Couldn't get a result from the weights API at {WEIGHTS_API_URL} "
                                 f"({getattr(exc, 'reason', exc)}). Is it running?"), 502

        after = RESULTS_CSV.stat().st_mtime if RESULTS_CSV.exists() else None
        if after is None or after == before:
            return jsonify(error=f"The API finished (it reported saving to "
                                 f"{result.get('saved_to', 'data/results.csv')}), but {RESULTS_CSV} "
                                 f"did not change. Make sure the API writes to the same file this app reads."), 502

        links = catalog.reload_graph(RESULTS_CSV)
        saved = {"weights": weights, "pairs": result.get("count"), "links": links,
                 "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        settings_file.write_text(json.dumps(saved), encoding="utf-8")
        return jsonify(saved)
    finally:
        _recalc_lock.release()


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
