from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

RESULTS_CSV_PATH = Path("data/results.csv")
JOIN_KEYS = ["article1", "article2"]

app = FastAPI()

# Allow the HTML page to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass(frozen=True)
class ScoreSpec:
    name: str  # Logical key used in weights, output, and normalized column names
    csv_path: str
    source_col: str  # Score column name in this CSV
    output_col: str  # Column name kept in the merged DataFrame


# To add a new score later: add a ScoreSpec here and a weight field to SettingsRequest.
SCORE_SPECS: list[ScoreSpec] = [
    ScoreSpec(
        name="users",
        csv_path="data/users.csv",
        source_col="similarity_score",
        output_col="users_similarity_score",
    ),
    ScoreSpec(
        name="genres_tags",
        csv_path="data/prediction_genres_tags.csv",
        source_col="Est. J",
        output_col="Est. J",
    ),
]


class SettingsRequest(BaseModel):
    """Weights for each registered score source."""

    users: float = Field(default=0.5, ge=0.0)
    genres_tags: float = Field(default=0.5, ge=0.0)


def normalize_0_to_1(series: pd.Series) -> pd.Series:
    """Min-max normalize a Series to [0, 1] while preserving NA values."""
    numeric = pd.to_numeric(series, errors="coerce")
    min_val = numeric.min()
    max_val = numeric.max()
    span = max_val - min_val

    if pd.isna(span) or span == 0:
        return pd.Series(
            np.where(numeric.notna(), 0.0, np.nan),
            index=series.index,
            dtype="float64",
        )

    return ((numeric - min_val) / span).astype("float64")


def build_merged_dataframe(specs: list[ScoreSpec]) -> pd.DataFrame:
    """Outer-merge every CSV on (article1, article2), keeping NAs for missing pairs."""
    if not specs:
        return pd.DataFrame(columns=JOIN_KEYS)

    merged: pd.DataFrame | None = None

    for spec in specs:
        raw_df = pd.read_csv(spec.csv_path)
        part = (
            raw_df[JOIN_KEYS + [spec.source_col]]
            .rename(columns={spec.source_col: spec.output_col})
            .drop_duplicates(subset=JOIN_KEYS)
        )
        part[spec.output_col] = pd.to_numeric(part[spec.output_col], errors="coerce")

        if merged is None:
            merged = part
        else:
            merged = pd.merge(merged, part, on=JOIN_KEYS, how="outer")

    return merged


def add_weighted_score(
    df: pd.DataFrame,
    specs: list[ScoreSpec],
    weights: dict[str, float],
    output_col: str = "combined_score",
) -> pd.DataFrame:
    """Normalize each configured score to [0, 1] and compute each row's weighted mean over non-NA scores."""
    df = df.copy()
    weighted_sum = pd.Series(0.0, index=df.index, dtype="float64")
    active_weight_sum = pd.Series(0.0, index=df.index, dtype="float64")

    for spec in specs:
        w = float(weights.get(spec.name, 0.0))
        norm_col = f"{spec.name}_norm"
        norm_series = normalize_0_to_1(df[spec.output_col])
        df[norm_col] = norm_series

        available = norm_series.notna()
        weighted_sum = weighted_sum + np.where(available, w * norm_series.fillna(0.0), 0.0)
        active_weight_sum = active_weight_sum + np.where(available, w, 0.0)

    df[output_col] = np.where(
        active_weight_sum > 0,
        weighted_sum / active_weight_sum,
        np.nan,
    )

    return df


@app.post("/settings")
def settings(payload: SettingsRequest):
    weights = payload.model_dump()
    merged_df = build_merged_dataframe(SCORE_SPECS)
    scored_df = add_weighted_score(merged_df, SCORE_SPECS, weights)

    RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(RESULTS_CSV_PATH, index=False)

    # Convert NaN to None so FastAPI serializes missing values cleanly as JSON null
    records = scored_df.astype(object).where(scored_df.notna(), None).to_dict(orient="records")

    return {
        "weights": weights,
        "count": len(scored_df),
        "saved_to": str(RESULTS_CSV_PATH),
        "data": records,
    }


# uvicorn app:app --reload