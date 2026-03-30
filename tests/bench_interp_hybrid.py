"""
Benchmark: interp_hybrid_to_pressure — the #1 geocat-comp workflow.

This is what CESM/CAM researchers actually do: convert hybrid-sigma
vertical coordinates to standard pressure levels. geocat-comp uses
metpy.interpolate.interpolate_1d; we use metrust's Rust version.

Run: python tests/bench_interp_hybrid.py
"""

import time
import numpy as np
import xarray as xr


def make_cesm_like_data(nlev=32, nlat=192, nlon=288, ntime=1):
    """Create synthetic CESM-like data on hybrid-sigma levels."""
    rng = np.random.default_rng(42)

    # Hybrid coefficients (approximate CESM values)
    hyam = xr.DataArray(
        np.linspace(0.003, 0.0, nlev),
        dims='lev', coords={'lev': np.arange(nlev)}
    )
    hybm = xr.DataArray(
        np.linspace(0.0, 1.0, nlev),
        dims='lev', coords={'lev': np.arange(nlev)}
    )

    # Surface pressure (Pa) — realistic range
    ps = xr.DataArray(
        98000 + rng.normal(0, 3000, (ntime, nlat, nlon)),
        dims=['time', 'lat', 'lon'],
        coords={
            'time': np.arange(ntime),
            'lat': np.linspace(-90, 90, nlat),
            'lon': np.linspace(0, 360, nlon),
        }
    )

    # Temperature field (K) — realistic lapse rate
    lev_frac = np.linspace(0, 1, nlev)
    t_base = 220 + 80 * lev_frac  # ~220K at top, ~300K at surface
    temperature = xr.DataArray(
        t_base[np.newaxis, :, np.newaxis, np.newaxis]
        + rng.normal(0, 3, (ntime, nlev, nlat, nlon)),
        dims=['time', 'lev', 'lat', 'lon'],
        coords={
            'time': np.arange(ntime),
            'lev': np.arange(nlev),
            'lat': np.linspace(-90, 90, nlat),
            'lon': np.linspace(0, 360, nlon),
        }
    )

    return temperature, ps, hyam, hybm


def main():
    print("=" * 72)
    print("  interp_hybrid_to_pressure Benchmark")
    print("  The #1 geocat-comp workflow (replacing NCL's vinth2p)")
    print("=" * 72)

    # Standard CESM grid: 32 levels x 192 lat x 288 lon
    temperature, ps, hyam, hybm = make_cesm_like_data()
    n_elements = temperature.size
    print(f"\n  Grid: 32 lev x 192 lat x 288 lon = {n_elements:,} elements")
    print(f"  Interpolating to 21 standard pressure levels\n")

    # --- geocat.comp (uses metpy) ---
    from geocat.comp import interp_hybrid_to_pressure as gc_interp

    # Warmup
    _ = gc_interp(temperature, ps, hyam, hybm, lev_dim='lev')

    t0 = time.perf_counter()
    gc_result = gc_interp(temperature, ps, hyam, hybm, lev_dim='lev')
    t_gc = time.perf_counter() - t0

    print(f"  geocat.comp (metpy):  {t_gc*1000:8.1f} ms")

    # --- geocat-rs (uses metrust) ---
    from geocat_rs.interp_hybrid import interp_hybrid_to_pressure as rs_interp

    # Warmup
    _ = rs_interp(temperature, ps, hyam, hybm, lev_dim='lev')

    t0 = time.perf_counter()
    rs_result = rs_interp(temperature, ps, hyam, hybm, lev_dim='lev')
    t_rs = time.perf_counter() - t0

    print(f"  geocat-rs (metrust):  {t_rs*1000:8.1f} ms")
    print(f"  Speedup:              {t_gc/t_rs:8.1f}x")

    # Verify results match
    gc_vals = gc_result.values
    rs_vals = rs_result.values

    # Both will have NaNs where extrapolation wasn't done
    mask = np.isfinite(gc_vals) & np.isfinite(rs_vals)
    if mask.sum() > 0:
        max_diff = np.max(np.abs(gc_vals[mask] - rs_vals[mask]))
        mean_diff = np.mean(np.abs(gc_vals[mask] - rs_vals[mask]))
        pct_valid = mask.sum() / gc_vals.size * 100
        print(f"\n  Valid points:  {mask.sum():,} / {gc_vals.size:,} ({pct_valid:.1f}%)")
        print(f"  Max diff:      {max_diff:.6e}")
        print(f"  Mean diff:     {mean_diff:.6e}")
        print(f"  Match:         {max_diff < 0.01}")
    else:
        print("\n  WARNING: No valid overlapping points to compare")

    # --- Larger grid ---
    print(f"\n{'='*72}")
    print("  Larger grid: 32 lev x 500 lat x 500 lon")
    print("=" * 72)

    temperature2, ps2, hyam2, hybm2 = make_cesm_like_data(nlat=500, nlon=500)
    n2 = temperature2.size
    print(f"  {n2:,} elements\n")

    t0 = time.perf_counter()
    gc_result2 = gc_interp(temperature2, ps2, hyam2, hybm2, lev_dim='lev')
    t_gc2 = time.perf_counter() - t0

    t0 = time.perf_counter()
    rs_result2 = rs_interp(temperature2, ps2, hyam2, hybm2, lev_dim='lev')
    t_rs2 = time.perf_counter() - t0

    print(f"  geocat.comp (metpy):  {t_gc2*1000:8.1f} ms")
    print(f"  geocat-rs (metrust):  {t_rs2*1000:8.1f} ms")
    print(f"  Speedup:              {t_gc2/t_rs2:8.1f}x")

    gc_v2 = gc_result2.values
    rs_v2 = rs_result2.values
    mask2 = np.isfinite(gc_v2) & np.isfinite(rs_v2)
    if mask2.sum() > 0:
        max_diff2 = np.max(np.abs(gc_v2[mask2] - rs_v2[mask2]))
        print(f"  Max diff:     {max_diff2:.6e}")
        print(f"  Match:        {max_diff2 < 0.01}")

    print(f"\n{'='*72}")
    print("  This is the function CESM researchers use most.")
    print("  geocat-rs uses metrust's Rust interpolation engine.")
    print("=" * 72)


if __name__ == "__main__":
    main()
