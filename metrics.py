"""Deterministic and ensemble metrics, evaluated over a sea mask."""

import numpy as np


def _pair(pred, ref, mask=None):
    pred = np.asarray(pred, float).ravel()
    ref = np.asarray(ref, float).ravel()
    ok = np.isfinite(pred) & np.isfinite(ref)
    if mask is not None:
        ok &= np.asarray(mask, bool).ravel()
    return pred[ok], ref[ok]


def bias(pred, ref, mask=None):
    p, r = _pair(pred, ref, mask)
    return float(np.mean(p - r))


def rmse(pred, ref, mask=None):
    p, r = _pair(pred, ref, mask)
    return float(np.sqrt(np.mean((p - r) ** 2)))


def mae(pred, ref, mask=None):
    p, r = _pair(pred, ref, mask)
    return float(np.mean(np.abs(p - r)))


def correlation(pred, ref, mask=None):
    p, r = _pair(pred, ref, mask)
    if p.size < 2:
        return np.nan
    return float(np.corrcoef(p, r)[0, 1])


def deterministic_scores(pred, ref, mask=None):
    return {
        "bias": bias(pred, ref, mask),
        "rmse": rmse(pred, ref, mask),
        "mae": mae(pred, ref, mask),
        "corr": correlation(pred, ref, mask),
    }


def _ens_valid(ensemble, ref, mask=None):
    ens = np.asarray(ensemble, float)
    ref = np.asarray(ref, float)
    ok = np.isfinite(ref) & np.all(np.isfinite(ens), axis=0)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    return ens[:, ok], ref[ok]


def spread(ensemble, ref=None, mask=None):
    if ref is None:
        ens = np.asarray(ensemble, float)
        return float(np.nanmean(ens.std(axis=0)))
    ens, _ = _ens_valid(ensemble, ref, mask)
    return float(np.mean(ens.std(axis=0)))


def picp(ensemble, ref, level=0.9, mask=None):
    ens, r = _ens_valid(ensemble, ref, mask)
    lo = np.quantile(ens, (1 - level) / 2, axis=0)
    hi = np.quantile(ens, (1 + level) / 2, axis=0)
    return float(np.mean((r >= lo) & (r <= hi)))


def zscore(ensemble, ref, mask=None):
    ens, r = _ens_valid(ensemble, ref, mask)
    m = ens.mean(axis=0)
    s = ens.std(axis=0)
    s[s == 0] = np.nan
    z = (m - r) / s
    z = z[np.isfinite(z)]
    return float(np.mean(z)), float(np.std(z))


def crps(ensemble, ref, mask=None):
    ens, r = _ens_valid(ensemble, ref, mask)
    n = ens.shape[0]
    term1 = np.mean(np.abs(ens - r[None, :]), axis=0)
    xs = np.sort(ens, axis=0)
    w = np.arange(1, n + 1)
    term2 = (2.0 / n ** 2) * np.sum((2 * w - n - 1)[:, None] * xs, axis=0)
    return float(np.mean(term1 - 0.5 * term2))


def nll(ensemble, ref, mask=None, eps=1e-6):
    ens, r = _ens_valid(ensemble, ref, mask)
    m = ens.mean(axis=0)
    s = np.maximum(ens.std(axis=0), eps)
    val = 0.5 * np.log(2 * np.pi * s ** 2) + (m - r) ** 2 / (2 * s ** 2)
    return float(np.mean(val))


def coverage(obs_mask, sea_mask):
    obs_mask = np.asarray(obs_mask, bool)
    sea_mask = np.asarray(sea_mask, bool)
    n = sea_mask.sum()
    return float((obs_mask & sea_mask).sum() / n) if n else np.nan
