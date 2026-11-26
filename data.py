"""Data layer: anime catalog, similarity graph, MAL export parser, recommender."""
from __future__ import annotations

import gzip
import html
import io
import json
import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

try:  # defusedxml protects against malicious uploads (entity bombs etc.)
    from defusedxml import ElementTree as ET
except ImportError:  # pragma: no cover
    import xml.etree.ElementTree as ET

# Statuses that mean "I have (at least partly) seen this".
SEEN_STATUSES = {"Completed", "Watching", "On-Hold", "Dropped"}

_STATUS_ALIASES = {
    "1": "Watching", "watching": "Watching",
    "2": "Completed", "completed": "Completed",
    "3": "On-Hold", "on-hold": "On-Hold", "onhold": "On-Hold", "on hold": "On-Hold",
    "4": "Dropped", "dropped": "Dropped",
    "6": "Plan to Watch", "plan to watch": "Plan to Watch", "plantowatch": "Plan to Watch",
}

MAX_UNCOMPRESSED = 100 * 1024 * 1024


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _none(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _int(v):
    v = _none(v)
    if v is None:
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _float(v):
    v = _none(v)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _str(v):
    v = _none(v)
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def _split_list(v):
    v = _str(v)
    if not v:
        return []
    return [p for p in re.split(r"\s*[,|;]\s*", v) if p]


def _plain_text(v):
    """Descriptions often contain HTML (<br>, <i>...). Turn into plain text."""
    v = _str(v)
    if not v:
        return None
    v = re.sub(r"<br\s*/?>", "\n", v, flags=re.I)
    v = re.sub(r"<[^>]+>", "", v)
    return html.unescape(v).strip() or None


def normalize_status(raw: str | None) -> str:
    raw = (raw or "").strip()
    return _STATUS_ALIASES.get(raw.lower(), raw or "Unknown")


# --------------------------------------------------------------------------- #
# Catalog + similarity graph
# --------------------------------------------------------------------------- #
class Catalog:
    CARD_FIELDS = ("id", "title", "romaji", "format", "year", "episodes",
                   "score", "popularity", "cover", "genres")

    def __init__(self, dataset_path: Path, results_path: Path, max_neighbors: int = 200):
        self.max_neighbors = max_neighbors
        self._load_dataset(dataset_path)
        # (dst, sim, span) is swapped as one tuple so readers never see a half-loaded graph.
        self._graph = self._build_graph(results_path, max_neighbors)

    # ---- dataset -------------------------------------------------------- #
    def _load_dataset(self, path: Path) -> None:
        df = pd.read_csv(path, low_memory=False)
        df["mal_id"] = pd.to_numeric(df["mal_id"], errors="coerce")
        df = df.dropna(subset=["mal_id"]).drop_duplicates("mal_id")

        self.records: dict[int, dict] = {}
        for r in df.to_dict("records"):
            mid = int(r["mal_id"])
            title = _str(r.get("title_english")) or _str(r.get("title_romaji")) \
                or _str(r.get("title_native")) or f"MAL #{mid}"
            romaji = _str(r.get("title_romaji"))
            season = " ".join(str(x) for x in (_str(r.get("season")), _int(r.get("season_year"))) if x)
            rec = {
                "id": mid,
                "title": title,
                "romaji": romaji if romaji and romaji != title else None,
                "native": _str(r.get("title_native")),
                "format": _str(r.get("format")),
                "status": _str(r.get("status")),
                "episodes": _int(r.get("episodes")),
                "year": _int(r.get("start_year")),
                "season": season or None,
                "source": _str(r.get("source")),
                "score": _float(r.get("score")),
                "rank": _int(r.get("rank")),
                "popularity": _int(r.get("popularity_rank")),
                "members": _int(r.get("members")),
                "genres": _split_list(r.get("genres")),
                "tags": _split_list(r.get("tags")),
                "studios": _split_list(r.get("studios")),
                "description": _plain_text(r.get("description")),
                "cover": _str(r.get("cover_image")),
            }
            rec["_search"] = " ".join(
                s.lower() for s in (
                    title, romaji, _str(r.get("title_native")), _str(r.get("synonyms"))
                ) if s
            )
            self.records[mid] = rec

        # Popularity order: rank 1 first, unranked entries last.
        self.popular_ids = sorted(
            self.records,
            key=lambda i: (self.records[i]["popularity"] is None,
                           self.records[i]["popularity"] or 0, i),
        )

    def card(self, mal_id: int, fallback: dict | None = None) -> dict:
        rec = self.records.get(mal_id)
        if rec:
            return {k: rec[k] for k in self.CARD_FIELDS}
        fb = fallback or {}
        return {"id": mal_id, "title": fb.get("title") or f"MAL #{mal_id}", "romaji": None,
                "format": fb.get("type"), "year": None, "episodes": fb.get("episodes"),
                "score": None, "popularity": None, "cover": None, "genres": []}

    def detail(self, mal_id: int, fallback: dict | None = None) -> dict:
        rec = self.records.get(mal_id)
        if rec:
            return {k: v for k, v in rec.items() if not k.startswith("_")}
        d = self.card(mal_id, fallback)
        d.update(native=None, status=None, season=None, source=None, rank=None, members=None,
                 tags=[], studios=[], description=None)
        return d

    def popular(self, page: int, per_page: int, query: str = "") -> tuple[int, list[dict]]:
        ids = self.popular_ids
        q = query.strip().lower()
        if q:
            ids = [i for i in ids if q in self.records[i]["_search"]]
        start = (page - 1) * per_page
        return len(ids), [self.card(i) for i in ids[start:start + per_page]]

    # ---- similarity graph (CSR layout in numpy) ------------------------- #
    @staticmethod
    def _build_graph(path: Path, max_neighbors: int):
        if not Path(path).exists():  # not generated yet: start empty, Settings can create it
            return np.empty(0, "int32"), np.empty(0, "float32"), {}
        df = pd.read_csv(
            path,
            usecols=["article1", "article2", "combined_score"],
            dtype={"article1": "int32", "article2": "int32", "combined_score": "float32"},
        ).dropna()
        df = df[df.article1 != df.article2]

        # Treat every pair as undirected: index it in both directions.
        fwd = df.rename(columns={"article1": "src", "article2": "dst"})
        rev = df.rename(columns={"article2": "src", "article1": "dst"})
        edges = pd.concat([fwd, rev], ignore_index=True)
        del df, fwd, rev

        edges = edges.sort_values(["src", "combined_score"], ascending=[True, False], kind="stable")
        edges = edges.drop_duplicates(["src", "dst"], keep="first")  # keep best score per pair
        if max_neighbors and max_neighbors > 0:
            edges = edges.groupby("src", sort=False).head(max_neighbors)

        src = edges["src"].to_numpy()
        dst = edges["dst"].to_numpy()
        sim = edges["combined_score"].to_numpy()
        ids, starts = np.unique(src, return_index=True)
        ends = np.append(starts[1:], len(src))
        span = {int(i): (int(s), int(e)) for i, s, e in zip(ids, starts, ends)}
        return dst, sim, span

    def reload_graph(self, path: Path) -> int:
        """Re-read results.csv (e.g. after the weights changed). Returns the new link count."""
        self._graph = self._build_graph(path, self.max_neighbors)
        return self.edge_count

    @property
    def edge_count(self) -> int:
        return len(self._graph[0])

    def has_similarity(self, mal_id: int) -> bool:
        return mal_id in self._graph[2]

    def neighbors(self, mal_id: int, limit: int | None = None) -> list[tuple[int, float]]:
        """[(neighbor_id, combined_score)] best first, only titles present in the dataset."""
        g_dst, g_sim, g_span = self._graph
        span = g_span.get(mal_id)
        if not span:
            return []
        s, e = span
        out = []
        for d, v in zip(g_dst[s:e].tolist(), g_sim[s:e].tolist()):
            if d in self.records:
                out.append((d, v))
                if limit and len(out) >= limit:
                    break
        return out

    def similar(self, mal_id: int, limit: int = 60) -> list[dict]:
        out = []
        for d, v in self.neighbors(mal_id, limit):
            c = self.card(d)
            c["similarity"] = round(v, 4)
            out.append(c)
        return out

    # ---- recommendations ------------------------------------------------ #
    def recommend(self, entries: list[dict], min_score: int = 8,
                  include_plan: bool = True, limit: int = 100) -> list[dict]:
        """
        Priority = sum over every anime you rated >= min_score of
                   weight(your score) * combined_score(liked, candidate)

        weight = (score - 4) / 6   ->  6: 0.33, 7: 0.5, 8: 0.67, 9: 0.83, 10: 1.0

        Anime you already saw (Completed / Watching / On-Hold / Dropped) are excluded.
        Plan-to-watch titles are not seen yet, so they stay (flagged) unless include_plan=False.
        """
        by_id = {e["id"]: e for e in entries}
        seen = {i for i, e in by_id.items() if e["status"] in SEEN_STATUSES}
        planned = {i for i, e in by_id.items() if e["status"] == "Plan to Watch"}
        excluded = seen | (set() if include_plan else planned)

        g_dst, g_sim, g_span = self._graph
        srcs, dsts, contrib = [], [], []
        for e in entries:
            sc = e["score"]
            if sc < min_score or e["status"] not in SEEN_STATUSES:
                continue
            span = g_span.get(e["id"])
            if not span:
                continue
            s, t = span
            w = (sc - 4) / 6.0
            srcs.append(np.full(t - s, e["id"], dtype=np.int64))
            dsts.append(g_dst[s:t].astype(np.int64))
            contrib.append(g_sim[s:t].astype(np.float64) * w)
        if not srcs:
            return []

        df = pd.DataFrame({"src": np.concatenate(srcs), "dst": np.concatenate(dsts),
                           "c": np.concatenate(contrib)})
        df = df[~df["dst"].isin(excluded) & df["dst"].isin(self.records.keys())]
        if df.empty:
            return []

        totals = df.groupby("dst")["c"].agg(total="sum", n="count").sort_values("total", ascending=False)
        totals = totals.head(limit)
        top_ids = set(totals.index)
        why = (df[df["dst"].isin(top_ids)].sort_values("c", ascending=False)
               .groupby("dst").head(3).groupby("dst")["src"].apply(list))

        best = float(totals["total"].iloc[0]) or 1.0
        out = []
        for mal_id, row in totals.iterrows():
            card = self.card(int(mal_id))
            card.update(
                priority=round(float(row["total"]), 4),
                relative=round(float(row["total"]) / best, 4),
                n_sources=int(row["n"]),
                planned=int(mal_id) in planned,
                reasons=[
                    {"id": int(s), "title": self.card(int(s), by_id.get(int(s)))["title"],
                     "score": by_id[int(s)]["score"]}
                    for s in why.get(mal_id, [])
                ],
            )
            out.append(card)
        return out


# --------------------------------------------------------------------------- #
# MyAnimeList export
# --------------------------------------------------------------------------- #
def parse_mal_export(raw: bytes) -> dict:
    """Parse a MAL XML export (plain or .gz). Returns {"user": str, "entries": [...]}."""
    if raw[:2] == b"\x1f\x8b":
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
            raw = gz.read(MAX_UNCOMPRESSED + 1)
        if len(raw) > MAX_UNCOMPRESSED:
            raise ValueError("The file is too large once decompressed.")
    try:
        root = ET.fromstring(raw)
    except Exception as exc:
        raise ValueError("That file isn't valid XML.") from exc
    if root.tag != "myanimelist":
        raise ValueError("That doesn't look like a MyAnimeList export (no <myanimelist> root).")

    user = (root.findtext("myinfo/user_name") or "").strip()
    entries = []
    for a in root.findall("anime"):
        try:
            mal_id = int(a.findtext("series_animedb_id"))
        except (TypeError, ValueError):
            continue
        try:
            score = max(0, min(10, int(float(a.findtext("my_score") or 0))))
        except ValueError:
            score = 0
        entries.append({
            "id": mal_id,
            "title": (a.findtext("series_title") or "").strip(),
            "type": (a.findtext("series_type") or "").strip() or None,
            "episodes": _int(a.findtext("series_episodes")),
            "watched": _int(a.findtext("my_watched_episodes")) or 0,
            "score": score,
            "status": normalize_status(a.findtext("my_status")),
        })
    if not entries:
        raise ValueError("No anime entries found in that file.")
    return {"user": user, "entries": entries}


class UserLists:
    """Stores parsed lists on disk; the browser session only holds an opaque id."""
    _TOKEN = re.compile(r"[0-9a-f]{32}")

    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, payload: dict, token: str | None = None) -> str:
        if not token or not self._TOKEN.fullmatch(token):
            token = uuid.uuid4().hex
        (self.dir / f"{token}.json").write_text(json.dumps(payload), encoding="utf-8")
        return token

    def load(self, token: str | None) -> dict | None:
        if not token or not self._TOKEN.fullmatch(token):
            return None
        p = self.dir / f"{token}.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def delete(self, token: str | None) -> None:
        if token and self._TOKEN.fullmatch(token):
            (self.dir / f"{token}.json").unlink(missing_ok=True)
