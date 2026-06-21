# cygnss-wind-reconstruction

Code for the paper

**Score-Based Data Assimilation of 10-m Sea Surface Winds in the Western North
Pacific: From Routine Conditions to Tropical Cyclones**

We reconstruct gridded 10-m sea surface winds over the western North Pacific by
assimilating sparse CYGNSS observations into a score-based diffusion prior, and
evaluate the fields from routine conditions through tropical cyclones. This repo
holds the gridding operator, the evaluation metrics, and the plotting used in
the paper. The prior and the sampler build on
[EDM](https://github.com/NVlabs/edm) and
[SDA](https://github.com/francois-rozet/sda).

## Scripts

- `mvc_grid.py` — grid sparse CYGNSS L2 winds onto a 0.25° grid, per-cell maximum
  (`mvc_grid`) or mean (`mean_grid`).
- `metrics.py` — deterministic (bias, RMSE, MAE, correlation) and probabilistic
  (spread, PICP, z-score, CRPS, NLL) scores over a sea mask.
- `plot_reconstruction.py` — plot a reconstruction with an optional reference and
  their difference.

## Usage

```python
import numpy as np
from metrics import deterministic_scores, picp, crps, zscore

recon = np.load("recon.npy")      # (H, W)
ref   = np.load("era5.npy")       # (H, W)
sea   = np.load("sea_mask.npy")   # (H, W) bool

print(deterministic_scores(recon, ref, mask=sea))

ens = np.load("ensemble.npy")     # (N, H, W)
print(picp(ens, ref, level=0.9, mask=sea))
print(crps(ens, ref, mask=sea))
print(zscore(ens, ref, mask=sea))
```

```bash
python plot_reconstruction.py recon.npy --ref era5.npy --out case.png
```

## Data

CCMP for training, CYGNSS as the constraint, ERA5 / IBTrACS / SAR for evaluation.

- CCMP v3.1: https://data.remss.com/ccmp/v03.1/
- CYGNSS L2 SWSP v1.2: https://podaac.jpl.nasa.gov/dataset/CYGNSS_NOAA_L2_SWSP_25KM_V1.2
- ERA5: https://cds.climate.copernicus.eu/
- IBTrACS: https://www.ncei.noaa.gov/products/international-best-track-archive
- SAR winds (NOAA STAR): https://www.star.nesdis.noaa.gov/socd/mecb/sar/index.php

## Requirements

`numpy`, `matplotlib`, and `xarray` (only for NetCDF input).

## Citation

Han et al. (2026). Score-Based Data Assimilation of 10-m Sea Surface Winds in the
Western North Pacific: From Routine Conditions to Tropical Cyclones.

## License

MIT
