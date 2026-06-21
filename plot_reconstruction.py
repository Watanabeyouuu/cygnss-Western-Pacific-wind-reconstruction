"""Plot a reconstruction, optionally with a reference field and their difference."""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_fields(recon, ref=None, out="reconstruction.png", vmax=None, title=None):
    recon = np.asarray(recon, float)
    if vmax is None:
        vmax = float(np.nanpercentile(recon, 99))

    ncol = 1 if ref is None else 3
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.0), squeeze=False)
    axes = axes[0]

    im = axes[0].imshow(recon, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
    axes[0].set_title("Reconstruction")
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="Wind speed (m/s)")

    if ref is not None:
        ref = np.asarray(ref, float)
        im = axes[1].imshow(ref, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        axes[1].set_title("Reference")
        fig.colorbar(im, ax=axes[1], shrink=0.8, label="Wind speed (m/s)")

        diff = recon - ref
        d = float(np.nanpercentile(np.abs(diff), 99))
        im = axes[2].imshow(diff, origin="lower", cmap="RdBu_r", vmin=-d, vmax=d)
        axes[2].set_title("Difference")
        fig.colorbar(im, ax=axes[2], shrink=0.8, label="Difference (m/s)")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def _load(path):
    if path.endswith(".npy"):
        return np.load(path)
    import xarray as xr
    ds = xr.open_dataset(path)
    var = list(ds.data_vars)[0]
    return ds[var].values


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("recon", help=".npy or .nc reconstruction field")
    ap.add_argument("--ref", default=None, help="reference field")
    ap.add_argument("--out", default="reconstruction.png")
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    ref = _load(a.ref) if a.ref else None
    plot_fields(_load(a.recon), ref, out=a.out, vmax=a.vmax, title=a.title)
