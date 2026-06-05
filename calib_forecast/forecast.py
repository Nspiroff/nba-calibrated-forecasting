"""Calibrated NBA box-score forecasting — predict the DISTRIBUTION, score it properly.

Most box-score models predict a number and report MAE. A number can't tell you the
*chance* a player clears a line, and MAE hides whether the model knows what it
doesn't know. This predicts the full distribution of a player's next game and scores
it with proper rules:

    CRPS      (lower = better) — sharpness + calibration of the whole distribution
    log-score (lower = better) — penalizes low probability on what actually happened
    PIT / cov80               — is the distribution calibrated? (80% interval should
                                contain the outcome 80% of the time)

Three models, increasing sophistication:
    Poisson(trailing mean)         — the naive baseline (and badly mis-calibrated)
    NegBinomial(GBM mean, r)       — learned mean + the right over-dispersion
    Quantile Regression (GBM)      — learns conditional quantiles directly

All features are leakage-safe (trailing rollups, shifted one game). Train on past
seasons, evaluate on the most recent — never on games the model has seen.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C

TAUS = np.array([0.02, 0.05, 0.10, 0.16, 0.25, 0.40, 0.50, 0.60, 0.75, 0.84, 0.90, 0.95, 0.98])
I10, I90 = int(np.where(TAUS == 0.10)[0][0]), int(np.where(TAUS == 0.90)[0][0])
KMAX = {"pts": 80, "reb": 40, "ast": 30, "fg3m": 15}


# ---- proper scoring rules ----
def crps_count(cdf, y, kgrid):
    ind = (kgrid[None, :] >= y[:, None]).astype(float)
    return ((cdf - ind) ** 2).sum(axis=1)


def pit_mid(cdf_y, cdf_ym1):
    return 0.5 * (cdf_y + cdf_ym1)


def calib(pit):
    pit = pit[np.isfinite(pit)]
    return {"cov80": round(float(((pit > .1) & (pit < .9)).mean()), 3),
            "cov50": round(float(((pit > .25) & (pit < .75)).mean()), 3),
            "pit_mean": round(float(pit.mean()), 3)}


# ---- features ----
def load_features() -> pd.DataFrame:
    df = pd.read_parquet(C.DATA).sort_values(["player_id", "date"]).reset_index(drop=True)
    g = df.groupby("player_id", sort=False)
    for w in (5, 10):
        roll = (g[C.ROLL_COLS].shift(1).groupby(df["player_id"], sort=False)
                .rolling(w, min_periods=3).mean().reset_index(level=0, drop=True))
        df = df.join(roll.add_prefix(f"t{w}_"))
    return df.dropna(subset=[f"t10_{c}" for c in C.ROLL_COLS])


FEATS = [f"t{w}_{c}" for w in (5, 10) for c in ["min", "pts", "reb", "ast", "fg3m", "fga", "fta", "fg3a"]]


def fit_score(df: pd.DataFrame, stat: str) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor as GBR

    kmax = KMAX[stat]
    kgrid = np.arange(0, kmax + 1)
    tr, te = df[df["Season"] < C.TEST_SEASON], df[df["Season"] == C.TEST_SEASON]
    ytr, yte = tr[stat].to_numpy(float), te[stat].to_numpy(float)
    lam = te[f"t10_{stat}"].to_numpy(float)                      # trailing-mean baseline
    Xtr, Xte = tr[FEATS].to_numpy(float), te[FEATS].to_numpy(float)
    out = {"stat": C.STATS[stat], "n_test": int(len(te)), "models": {}}

    def record(name, cdf, cdf_y, cdf_ym1):
        out["models"][name] = {"crps": round(float(crps_count(cdf, yte, kgrid).mean()), 4),
                               **calib(pit_mid(cdf_y, cdf_ym1))}
        out.setdefault("_pit", {})[name] = pit_mid(cdf_y, cdf_ym1)

    # 1. Poisson(trailing mean)
    record("poisson", stats.poisson.cdf(kgrid[None, :], lam[:, None]),
           stats.poisson.cdf(yte, lam), stats.poisson.cdf(yte - 1, lam))

    # 2. NegBinomial(GBM mean, dispersion from train)
    r = max(float(np.mean(tr[f"t10_{stat}"])) ** 2 /
            max(float(np.var(ytr - tr[f"t10_{stat}"].to_numpy())) - float(np.mean(tr[f"t10_{stat}"])), 1e-6), 0.3)
    mean_gbm = np.clip(GBR(max_iter=300, learning_rate=0.05, max_depth=6, random_state=0)
                       .fit(Xtr, ytr).predict(Xte), 0.01, None)
    p = r / (r + mean_gbm)
    record(f"negbin(gbm,r={r:.1f})", stats.nbinom.cdf(kgrid[None, :], r, p[:, None]),
           stats.nbinom.cdf(yte, r, p), stats.nbinom.cdf(yte - 1, r, p))

    # 3. Quantile regression -> CDF
    Q = np.zeros((len(te), len(TAUS)))
    for j, t in enumerate(TAUS):
        Q[:, j] = GBR(loss="quantile", quantile=float(t), max_iter=150,
                      learning_rate=0.07, max_depth=6, random_state=0).fit(Xtr, ytr).predict(Xte)
    Q = np.clip(np.maximum.accumulate(np.sort(Q, axis=1), axis=1), 0, None)
    qcdf = np.vstack([np.interp(kgrid, Q[i], TAUS, left=0, right=1) for i in range(len(te))])
    cov80 = float(((yte >= Q[:, I10]) & (yte <= Q[:, I90])).mean())
    out["models"]["quantile_reg"] = {"crps": round(float(crps_count(qcdf, yte, kgrid).mean()), 4),
                                     "cov80": round(cov80, 3), "cov50": None,
                                     "pit_mean": round(float(np.array([np.interp(yte[i], Q[i], TAUS)
                                                                       for i in range(len(te))]).mean()), 3)}
    out["_pit"]["quantile_reg"] = np.array([np.interp(yte[i], Q[i], TAUS) for i in range(len(te))])
    return out


def run_all(save=True) -> dict:
    df = load_features()
    results = {C.STATS[s]: fit_score(df, s) for s in C.STATS}
    table = {stat: {m: v for m, v in r["models"].items()} for stat, r in results.items()}
    if save:
        clean = {s: {"n_test": r["n_test"], "models": r["models"]} for s, r in results.items()}
        (C.ASSETS.parent / "scoreboard.json").write_text(json.dumps(clean, indent=2))
    _print(results)
    return results


def _print(results):
    print(f"\n=== Calibrated box-score forecasting | test {C.TEST_SEASON} ===")
    print("(CRPS lower=better · cov80 target 0.80 · PITmean target 0.50)\n")
    for stat, r in results.items():
        print(f"{stat.upper()}  (n={r['n_test']})")
        print(f"  {'model':<22}{'CRPS':>8}{'cov80':>8}{'PITmean':>9}")
        for m, v in r["models"].items():
            print(f"  {m:<22}{v['crps']:>8.3f}{v['cov80']:>8.2f}{v['pit_mean']:>9.2f}")
        print()


if __name__ == "__main__":
    run_all()
