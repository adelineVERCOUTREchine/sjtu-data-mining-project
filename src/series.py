"""Classement EASE de séries distinctes à partir des scores appris par titre MAL."""

import csv
from collections import defaultdict

import numpy as np


class CatalogueSeries:
    def __init__(self, ids, fichier):
        with fichier.open(encoding="utf-8", newline="") as source:
            liens = {int(r["mal_id"]): (int(r["series_id"]), int(r["entry_order"]))
                     for r in csv.DictReader(source)}
        self.ids = ids
        self.liens = liens
        self.series = [self.groupe(i) for i in ids]
        self.entrees = {}
        for index, ident in enumerate(ids):
            groupe = self.series[index]
            ancien = self.entrees.get(groupe)
            if ancien is None or liens[ident][1] < liens[ids[ancien]][1]:
                self.entrees[groupe] = index

    def groupe(self, ident):
        return self.liens.get(ident, (ident, 0))[0]

    def audiences(self, fichier):
        """Audience de la fiche MAL la plus suivie de chaque série du modèle."""
        membres = {}
        with fichier.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                groupe = self.groupe(int(row["mal_id"]))
                if groupe in self.entrees and row["members"]:
                    audience = int(row["members"])
                    if audience > 0:
                        membres[groupe] = max(membres.get(groupe, 0), audience)
        return membres

    def recommander(self, poids, selection, exclus_mal, limite=10, preferences=None):
        sources = defaultdict(list)
        for index in selection:
            sources[self.series[index]].append(index)
        scores = np.zeros(len(self.ids), dtype=np.float64)
        contribution_max = np.full(len(self.ids), -np.inf)
        raisons = np.zeros(len(self.ids), dtype=int)
        for groupe, indices in sources.items():
            contribution = poids[indices].mean(axis=0)
            if preferences is not None:
                contribution *= preferences[groupe]
            scores += contribution
            masque = contribution > contribution_max
            raisons[masque] = self.entrees[groupe]
            contribution_max[masque] = contribution[masque]

        deja = {self.groupe(i) for i in exclus_mal} | set(sources)
        resultats = []
        for i in np.argsort(-scores, kind="stable"):
            if scores[i] <= 0:
                break
            groupe = self.series[i]
            if groupe not in deja:
                resultats.append((self.entrees[groupe], raisons[i]))
                deja.add(groupe)
                if len(resultats) == limite:
                    break
        return resultats
