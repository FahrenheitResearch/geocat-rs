# geocat-rs

Rust-accelerated drop-in replacements for [GeoCAT-comp](https://github.com/NCAR/geocat-comp) computational functions.

GeoCAT-comp is NCAR's Python replacement for NCL. geocat-rs accelerates the compute-heavy functions using Rust + rayon parallelism, while leaving the xarray/scipy wrapper functions alone (they're already fast).

## Performance

Real-world benchmarks on CESM-like 3D atmospheric fields. Every result verified identical to geocat.comp.

### The #1 workflow: `interp_hybrid_to_pressure`

The function CESM researchers use most — converting hybrid-sigma vertical coordinates to standard pressure levels (replacing NCL's `vinth2p`).

| Grid size | geocat.comp | geocat-rs | Speedup | Max diff |
|---|---|---|---|---|
| 32 lev x 192 lat x 288 lon (1.8M pts) | 161 ms | 21 ms | **7.5x** | 0.0 |
| 32 lev x 500 lat x 500 lon (8M pts) | 724 ms | 95 ms | **7.6x** | 0.0 |

Tested with 14 drop-in compatibility tests: standard/full CESM grids, custom levels, multiple timesteps, mountain pressure extremes, 4 random seeds. All pass.

### Meteorology functions on 3D fields (12.5M elements)

| Function | geocat.comp | geocat-rs | Speedup |
|---|---|---|---|
| Dewpoint temperature | 189 ms | 12 ms | **16x** |
| Relative humidity (table lookup) | 353 ms | 17 ms | **21x** |
| Relative humidity over water | 223 ms | 16 ms | **14x** |
| Saturation vapor pressure | 6.1 ms | 4.5 ms | 1.4x |

### WGS84 gradient functions (250K grid points)

| Function | geocat.comp | geocat-rs | Speedup |
|---|---|---|---|
| rad_lat_wgs84 (degree-48 polynomial) | 67 ms | 1.0 ms | **66x** |
| arc_lat_wgs84 (degree-49 polynomial) | 69 ms | 1.0 ms | **70x** |

### Overall across all accelerated functions

| | geocat.comp | geocat-rs |
|---|---|---|
| Total time | 908 ms | 53 ms |
| **Overall speedup** | | **17x** |

## Accuracy

All functions verified against geocat.comp to machine precision:

| Test suite | Tests | Result |
|---|---|---|
| interp_hybrid_to_pressure compatibility | 14 | All pass (max diff: 0.0) |
| Meteorology functions (scalar + array) | 20 | All pass (rtol < 1e-12) |
| WGS84 gradient functions | 6 | All pass (rtol < 1e-10) |
| Delta pressure | 1 | Pass |
| **Total** | **44** | **All pass** |

```
pytest tests/ -v
python tests/bench_interp_hybrid.py
python tests/bench_workflow.py
```

## Installation

```bash
pip install geocat-rs
```

Requires `metrust` for `interp_hybrid_to_pressure` (uses its Rust interpolation engine).

## Usage

### interp_hybrid_to_pressure (drop-in replacement)

```python
# Before
from geocat.comp import interp_hybrid_to_pressure

# After (same API, 7.5x faster)
from geocat_rs.interp_hybrid import interp_hybrid_to_pressure

# Usage is identical
output = interp_hybrid_to_pressure(
    data, ps, hyam, hybm,
    new_levels=new_levels,
    lev_dim='lev',
)
```

### Meteorology (array functions via Rust + rayon)

```python
from geocat_rs._geocat_rs import meteorology as met

# Scalar
td = met.dewtemp_scalar(300.0, 50.0)  # T(K), RH(%) -> Td(K)
hi = met.heat_index_scalar(95.0, 50.0, False)  # T(F), RH(%) -> HI(F)
rh = met.relhum_scalar(300.0, 0.01, 101325.0)  # T(K), w(kg/kg), P(Pa) -> RH(%)

# Array (rayon-parallel)
import numpy as np
td_array = met.dewtemp_array(temp_k, rh_pct)
rh_array = met.relhum_array(temp_k, mixr, pressure)
svp_array = met.saturation_vapor_pressure_array(temp_f)
```

### WGS84 gradient (66-70x faster)

```python
from geocat_rs._geocat_rs import gradient as grad

radius = grad.rad_lat_wgs84_array(lat_grid.ravel())  # meters
arc = grad.arc_lat_wgs84_array(lat_grid.ravel())      # meters from equator
```

## What's accelerated

| Function | Source | Speedup | Notes |
|---|---|---|---|
| `interp_hybrid_to_pressure` | Rust column interpolator + rayon | 7.5x | #1 most-used geocat function |
| `dewtemp` | Rust + rayon | 16x | Dutton formula |
| `relhum` (table) | Rust + rayon | 21x | NCL lookup table |
| `relhum_ice` / `relhum_water` | Rust + rayon | 14x | Alduchov/Murray |
| `heat_index` | Rust + rayon | -- | NWS Rothfusz regression |
| `saturation_vapor_pressure` | Rust + rayon | 1.4x | Tetens/FAO-56 |
| `saturation_vapor_pressure_slope` | Rust + rayon | -- | FAO-56 Eq. 13 |
| `psychrometric_constant` | Rust + rayon | -- | FAO-56 Eq. 8 |
| `max_daylight` | Rust + rayon | -- | FAO-56 solar declination |
| `rad_lat_wgs84` | Rust + rayon | 66x | Degree-48 polynomial |
| `arc_lat_wgs84` | Rust + rayon | 70x | Degree-49 polynomial |
| `arc_lon_wgs84` | Rust + rayon | -- | Radius * cos(lat) * lon |
| `temp_extrapolate` | Rust + rayon | -- | ECMWF Eq. 16 |
| `geo_height_extrapolate` | Rust + rayon | -- | ECMWF Eq. 15 |
| `delta_pressure_1d` | Rust | -- | Simmons & Burridge |

## What's NOT accelerated (and why)

| Function | Why not |
|---|---|
| `climatology_average`, `calendar_average`, `month_to_season` | xarray groupby — already optimized |
| `eofunc_eofs`, `eofunc_pcs` | Wraps `eofs` library (LAPACK) |
| `fourier_filter`, `fourier_*_pass` | numpy FFT (FFTW backend) |
| `decomposition`, `recomposition` | scipy spherical harmonics |
| `interp_multidim`, `interp_sigma_to_hybrid` | Wraps metpy (already Rust via metrust) |

## License

Apache-2.0 (same as GeoCAT-comp)
