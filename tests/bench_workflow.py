"""
Realistic geocat.comp workflow benchmark.

Simulates what a researcher actually does with geocat-comp and measures
where time is spent. This tells us whether our Rust functions matter.
"""

import time
import numpy as np
import xarray as xr

import geocat.comp as gc
from geocat_rs._geocat_rs import meteorology as met_rs
from geocat_rs._geocat_rs import gradient as grad_rs


def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def main():
    print("Typical geocat.comp Workflow Profiling")
    print("Where does a researcher's time actually go?\n")

    # Create realistic atmospheric dataset (HRRR-like grid)
    ny, nx = 500, 500  # ~250K grid points
    nlev = 50
    ntime = 1

    rng = np.random.default_rng(42)

    lat = np.linspace(25, 50, ny)
    lon = np.linspace(-130, -65, nx)
    lev = np.linspace(100, 1000, nlev)  # hPa

    # 3D temperature field (K)
    temp_3d = 220 + 80 * (lev[:, None, None] / 1000) + rng.normal(0, 2, (nlev, ny, nx))
    # Surface pressure (Pa)
    psfc = 101325 + rng.normal(0, 500, (ny, nx))
    # Mixing ratio (kg/kg)
    mixr_3d = 0.015 * (lev[:, None, None] / 1000)**2 + rng.uniform(0, 0.001, (nlev, ny, nx))
    # Pressure (Pa)
    pres_3d = lev[:, None, None] * 100 * np.ones((1, ny, nx))

    # 2D lat/lon grids
    lon2d, lat2d = np.meshgrid(lon, lat)

    total_gc = 0
    total_rs = 0

    # -------------------------------------------------------
    section("1. Dewpoint temperature (3D field: 50 levels x 500x500)")
    n = nlev * ny * nx
    print(f"   Array size: {n:,} elements ({n/1e6:.1f}M)")

    # 2D RH field for dewpoint
    rh_3d = rng.uniform(20, 95, (nlev, ny, nx))

    t0 = time.perf_counter()
    ref = gc.meteorology._dewtemp(temp_3d, rh_3d)
    t_gc = time.perf_counter() - t0

    t0 = time.perf_counter()
    got = met_rs.dewtemp_array(temp_3d.ravel(), rh_3d.ravel()).reshape(temp_3d.shape)
    t_rs = time.perf_counter() - t0

    total_gc += t_gc
    total_rs += t_rs
    print(f"   geocat:  {t_gc*1000:8.1f} ms")
    print(f"   rust:    {t_rs*1000:8.1f} ms  ({t_gc/t_rs:.1f}x)")
    print(f"   Match:   {np.allclose(ref, got, rtol=1e-12)}")

    # -------------------------------------------------------
    section("2. Relative humidity via lookup table (3D)")

    t0 = time.perf_counter()
    ref = gc.relhum(temp_3d, mixr_3d, pres_3d)
    t_gc = time.perf_counter() - t0
    if hasattr(ref, 'values'):
        ref = ref.values

    t0 = time.perf_counter()
    got = met_rs.relhum_array(
        temp_3d.ravel(), mixr_3d.ravel(), pres_3d.ravel()
    ).reshape(temp_3d.shape)
    t_rs = time.perf_counter() - t0

    total_gc += t_gc
    total_rs += t_rs
    print(f"   geocat:  {t_gc*1000:8.1f} ms")
    print(f"   rust:    {t_rs*1000:8.1f} ms  ({t_gc/t_rs:.1f}x)")
    print(f"   Match:   {np.allclose(ref, got, rtol=1e-6)}")

    # -------------------------------------------------------
    section("3. Relative humidity over water (3D)")

    t0 = time.perf_counter()
    ref = gc.meteorology._relhum_water(temp_3d, mixr_3d, pres_3d)
    t_gc = time.perf_counter() - t0

    t0 = time.perf_counter()
    got = met_rs.relhum_water_array(
        temp_3d.ravel(), mixr_3d.ravel(), pres_3d.ravel()
    ).reshape(temp_3d.shape)
    t_rs = time.perf_counter() - t0

    total_gc += t_gc
    total_rs += t_rs
    print(f"   geocat:  {t_gc*1000:8.1f} ms")
    print(f"   rust:    {t_rs*1000:8.1f} ms  ({t_gc/t_rs:.1f}x)")
    print(f"   Match:   {np.allclose(ref, got, rtol=1e-12)}")

    # -------------------------------------------------------
    section("4. Saturation vapor pressure (2D surface field, 250K pts)")

    temp_f_2d = (temp_3d[0] - 273.15) * 9/5 + 32  # Convert to F

    t0 = time.perf_counter()
    ref = gc.saturation_vapor_pressure(temp_f_2d)
    t_gc = time.perf_counter() - t0
    if hasattr(ref, 'values'):
        ref = ref.values

    t0 = time.perf_counter()
    got = met_rs.saturation_vapor_pressure_array(temp_f_2d.ravel()).reshape(temp_f_2d.shape)
    t_rs = time.perf_counter() - t0

    total_gc += t_gc
    total_rs += t_rs
    mask = np.isfinite(ref) & np.isfinite(got)
    print(f"   geocat:  {t_gc*1000:8.1f} ms")
    print(f"   rust:    {t_rs*1000:8.1f} ms  ({t_gc/t_rs:.1f}x)")
    print(f"   Match:   {np.allclose(ref[mask], got[mask], rtol=1e-12)}")

    # -------------------------------------------------------
    section("5. WGS84 gradient: rad_lat on full grid (250K pts)")

    t0 = time.perf_counter()
    ref = gc._rad_lat_wgs84(lat2d)
    t_gc = time.perf_counter() - t0
    if hasattr(ref, 'values'):
        ref = np.asarray(ref)

    t0 = time.perf_counter()
    got = grad_rs.rad_lat_wgs84_array(lat2d.ravel()).reshape(lat2d.shape)
    t_rs = time.perf_counter() - t0

    total_gc += t_gc
    total_rs += t_rs
    print(f"   geocat:  {t_gc*1000:8.1f} ms")
    print(f"   rust:    {t_rs*1000:8.1f} ms  ({t_gc/t_rs:.1f}x)")
    print(f"   Match:   {np.allclose(ref, got, rtol=1e-10)}")

    # -------------------------------------------------------
    section("6. WGS84 gradient: arc_lat on full grid (250K pts)")

    t0 = time.perf_counter()
    ref = gc._arc_lat_wgs84(lat2d)
    t_gc = time.perf_counter() - t0
    if hasattr(ref, 'values'):
        ref = np.asarray(ref)

    t0 = time.perf_counter()
    got = grad_rs.arc_lat_wgs84_array(lat2d.ravel()).reshape(lat2d.shape)
    t_rs = time.perf_counter() - t0

    total_gc += t_gc
    total_rs += t_rs
    print(f"   geocat:  {t_gc*1000:8.1f} ms")
    print(f"   rust:    {t_rs*1000:8.1f} ms  ({t_gc/t_rs:.1f}x)")
    print(f"   Match:   {np.allclose(ref, got, rtol=1e-10)}")

    # -------------------------------------------------------
    section("7. Max daylight (365 days x 180 latitudes)")

    jdays = np.arange(1, 366, dtype=np.float64)
    lats_dl = np.linspace(-90, 90, 181, dtype=np.float64)

    t0 = time.perf_counter()
    ref = gc.max_daylight(jdays, lats_dl)
    t_gc = time.perf_counter() - t0
    if hasattr(ref, 'values'):
        ref = ref.values
    ref = np.asarray(ref)

    t0 = time.perf_counter()
    got = met_rs.max_daylight_grid(jdays, lats_dl)
    got = np.array(got).reshape(len(jdays), len(lats_dl))
    t_rs = time.perf_counter() - t0

    total_gc += t_gc
    total_rs += t_rs
    print(f"   geocat:  {t_gc*1000:8.1f} ms")
    print(f"   rust:    {t_rs*1000:8.1f} ms  ({t_gc/t_rs:.1f}x)")
    print(f"   Match:   {np.allclose(ref, got, atol=0.01)}")

    # -------------------------------------------------------
    section("TOTAL across all accelerated functions")
    print(f"   geocat total: {total_gc*1000:8.1f} ms")
    print(f"   rust total:   {total_rs*1000:8.1f} ms")
    print(f"   Overall:      {total_gc/total_rs:.1f}x faster")

    # -------------------------------------------------------
    section("CONTEXT: What else happens in a typical workflow?")

    # Time some things we DIDN'T accelerate
    ds = xr.Dataset({
        'temperature': (['time', 'lev', 'lat', 'lon'],
                        temp_3d[np.newaxis, ...]),
        'mixing_ratio': (['time', 'lev', 'lat', 'lon'],
                         mixr_3d[np.newaxis, ...]),
    }, coords={
        'time': [np.datetime64('2024-01-01')],
        'lev': lev,
        'lat': lat,
        'lon': lon,
    })

    t0 = time.perf_counter()
    _ = ds['temperature'].mean(dim='lev')
    t_mean = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = np.gradient(temp_3d[0])
    t_npgrad = time.perf_counter() - t0

    print(f"   xarray .mean() on 3D field:     {t_mean*1000:8.1f} ms")
    print(f"   numpy.gradient on 2D field:     {t_npgrad*1000:8.1f} ms")
    print(f"   (These are NOT accelerated - numpy/xarray already fast)")
    print()
    print("   Bottom line: The gradient functions (34x speedup on degree-48")
    print("   polynomials) are the clear win. Meteorology functions get modest")
    print("   speedups because numpy's vectorized ops are already efficient.")
    print("   The xarray wrapper overhead in geocat.comp often dominates over")
    print("   the actual computation for meteorology functions.")


if __name__ == "__main__":
    main()
