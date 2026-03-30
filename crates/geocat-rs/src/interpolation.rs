//! Vertical interpolation extrapolation from GeoCAT-comp.
//!
//! ECMWF formulations from Trenberth, Berry, & Buja (NCAR/TN-396, 1993).

/// Dry air gas constant [J/(kg*K)]
const R_D: f64 = 287.04;
/// Inverse of gravity [s^2/m]
const G_INV: f64 = 1.0 / 9.80616;
/// Temperature lapse rate * R_d * g_inv
const ALPHA: f64 = 0.0065 * R_D * G_INV;

/// Extrapolate temperature below ground (ECMWF Equation 16).
///
/// All pressures in same units, phi_sfc in m^2/s^2 (geopotential).
pub fn temp_extrapolate(t_bot: f64, lev: f64, p_sfc: f64, ps: f64, phi_sfc: f64) -> f64 {
    let tstar = t_bot * (1.0 + ALPHA * (ps / p_sfc - 1.0));
    let hgt = phi_sfc * G_INV;
    let t0 = tstar + 0.0065 * hgt;
    let tplat = 298.0_f64.min(t0);

    let tprime0 = if hgt >= 2000.0 && hgt <= 2500.0 {
        0.002 * ((2500.0 - hgt) * t0 + (hgt - 2000.0) * tplat)
    } else if hgt > 2500.0 {
        tplat
    } else {
        f64::NAN // not used for hgt < 2000
    };

    let alnp = if hgt < 2000.0 {
        ALPHA * (lev / ps).ln()
    } else if tprime0 >= tstar {
        R_D * (tprime0 - tstar) / phi_sfc * (lev / ps).ln()
    } else {
        0.0
    };

    tstar * (1.0 + alnp + 0.5 * alnp * alnp + (1.0 / 6.0) * alnp * alnp * alnp)
}

/// Extrapolate geopotential height below ground (ECMWF Equation 15).
///
/// Returns geopotential height in geopotential meters.
pub fn geo_height_extrapolate(t_bot: f64, lev: f64, p_sfc: f64, ps: f64, phi_sfc: f64) -> f64 {
    let mut tstar = t_bot * (1.0 + ALPHA * (ps / p_sfc - 1.0));
    let hgt = phi_sfc * G_INV;
    let t0 = tstar + 0.0065 * hgt;

    let mut alph = if tstar <= 290.5 && t0 > 290.5 {
        R_D / phi_sfc * (290.5 - tstar)
    } else {
        ALPHA
    };

    if tstar > 290.5 && t0 > 290.5 {
        alph = 0.0;
        tstar = 0.5 * (290.5 + tstar);
    }

    if tstar < 255.0 {
        tstar = 0.5 * (tstar + 255.0);
    }

    let alnp = alph * (lev / ps).ln();
    hgt - R_D * tstar * G_INV * (lev / ps).ln() * (1.0 + 0.5 * alnp + (1.0 / 6.0) * alnp * alnp)
}

/// Pressure at hybrid levels: p(k) = hya(k) * p0 + hyb(k) * psfc.
#[inline]
pub fn pressure_at_hybrid_level(hya: f64, hyb: f64, psfc: f64, p0: f64) -> f64 {
    hya * p0 + hyb * psfc
}

/// Batch vertical interpolation: interpolate many columns in parallel.
///
/// xp_flat: flattened pressure array (nlev_in * ncols), column-major
/// data_flat: flattened data array (nlev_in * ncols), column-major
/// new_levels: target pressure levels (nlev_out)
/// nlev_in: number of input levels per column
/// ncols: number of columns
///
/// Returns flattened output (nlev_out * ncols), column-major.
pub fn interpolate_columns(
    xp_flat: &[f64],
    data_flat: &[f64],
    new_levels: &[f64],
    nlev_in: usize,
    ncols: usize,
) -> Vec<f64> {
    use rayon::prelude::*;

    let nlev_out = new_levels.len();
    let mut result = vec![f64::NAN; nlev_out * ncols];

    // Process columns in parallel
    result
        .par_chunks_mut(nlev_out)
        .enumerate()
        .for_each(|(col, out_col)| {
            let xp_start = col * nlev_in;
            let xp = &xp_flat[xp_start..xp_start + nlev_in];
            let data = &data_flat[xp_start..xp_start + nlev_in];

            // Determine if xp is increasing or decreasing
            let increasing = xp[0] <= xp[nlev_in - 1];

            for (out_idx, &xi) in new_levels.iter().enumerate() {
                // Out-of-bounds: return NaN (matching metpy behavior)
                if increasing {
                    if xi < xp[0] || xi > xp[nlev_in - 1] {
                        out_col[out_idx] = f64::NAN;
                        continue;
                    }
                } else {
                    if xi > xp[0] || xi < xp[nlev_in - 1] {
                        out_col[out_idx] = f64::NAN;
                        continue;
                    }
                }

                // Binary search for bracketing interval
                let (lo, hi) = if increasing {
                    let mut hi = nlev_in - 1;
                    for i in 0..nlev_in {
                        if xp[i] >= xi {
                            hi = i;
                            break;
                        }
                    }
                    let lo = if hi > 0 { hi - 1 } else { 0 };
                    (lo, hi)
                } else {
                    let mut hi = 0;
                    for i in 0..nlev_in {
                        if xp[i] <= xi {
                            hi = i;
                            break;
                        }
                    }
                    let lo = if hi > 0 { hi - 1 } else { 0 };
                    (lo, hi)
                };

                let denom = xp[hi] - xp[lo];
                if denom.abs() < 1e-30 {
                    out_col[out_idx] = data[lo];
                } else {
                    let t = (xi - xp[lo]) / denom;
                    out_col[out_idx] = data[lo] + t * (data[hi] - data[lo]);
                }
            }
        });

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temp_extrapolate_basic() {
        // Below ground extrapolation
        let t = temp_extrapolate(280.0, 105000.0, 100000.0, 101325.0, 500.0 * 9.80616);
        assert!(t > 270.0 && t < 300.0, "temp_extrap = {}", t);
    }

    #[test]
    fn test_geo_height_extrapolate_basic() {
        let h = geo_height_extrapolate(280.0, 105000.0, 100000.0, 101325.0, 500.0 * 9.80616);
        assert!(h < 500.0, "geo_height = {}", h); // below surface should be negative or < surface
    }

    #[test]
    fn test_pressure_hybrid() {
        let p = pressure_at_hybrid_level(0.5, 0.5, 101325.0, 100000.0);
        assert!((p - 100662.5).abs() < 1.0, "p_hybrid = {}", p);
    }
}
