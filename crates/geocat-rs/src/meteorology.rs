//! Meteorological calculations from GeoCAT-comp.
//!
//! All functions operate element-wise on f64 values. Array/batch operations
//! are handled by the PyO3 bindings layer using rayon.

// ============================================================
// Constants
// ============================================================

/// Gas constant for water vapor [J/(kg*K)]
const GC: f64 = 461.5;
/// GC / (1000 * 4.186) [cal/(g*K)]
const GCX: f64 = GC / (1000.0 * 4.186);

// Relative humidity constants
const T0: f64 = 273.15; // 0C in Kelvin
const EP: f64 = 0.622; // ratio of molecular masses (water/dry air)
const ONEMEP: f64 = 0.378; // 1 - EP

// Relhum lookup table (176 entries, 173.16K to 375.16K)
// Saturation vapor pressure in Pa, from NCL/geocat
const RELHUM_TABLE: [f64; 204] = [
    0.01403, 0.01719, 0.02101, 0.02561, 0.03117, 0.03784, 0.04584, 0.05542,
    0.06685, 0.08049, 0.09672, 0.1160, 0.1388, 0.1658, 0.1977, 0.2353,
    0.2796, 0.3316, 0.3925, 0.4638, 0.5472, 0.6444, 0.7577, 0.8894, 1.042,
    1.220, 1.425, 1.662, 1.936, 2.252, 2.615, 3.032, 3.511, 4.060, 4.688,
    5.406, 6.225, 7.159, 8.223, 9.432, 10.80, 12.36, 14.13, 16.12, 18.38,
    20.92, 23.80, 27.03, 30.67, 34.76, 39.35, 44.49, 50.26, 56.71, 63.93,
    71.98, 80.97, 90.98, 102.1, 114.5, 128.3, 143.6, 160.6, 179.4, 200.2,
    223.3, 248.8, 276.9, 307.9, 342.1, 379.8, 421.3, 466.9, 517.0, 572.0,
    632.3, 698.5, 770.9, 850.2, 937.0, 1032.0, 1146.6, 1272.0, 1408.1,
    1556.7, 1716.9, 1890.3, 2077.6, 2279.6, 2496.7, 2729.8, 2980.0, 3247.8,
    3534.1, 3839.8, 4164.8, 4510.5, 4876.9, 5265.1, 5675.2, 6107.8, 6566.2,
    7054.7, 7575.3, 8129.4, 8719.2, 9346.50, 10013.0, 10722.0, 11474.0,
    12272.0, 13119.0, 14017.0, 14969.0, 15977.0, 17044.0, 18173.0, 19367.0,
    20630.0, 21964.0, 23373.0, 24861.0, 26430.0, 28086.0, 29831.0, 31671.0,
    33608.0, 35649.0, 37796.0, 40055.0, 42430.0, 44927.0, 47551.0, 50307.0,
    53200.0, 56236.0, 59422.0, 62762.0, 66264.0, 69934.0, 73777.0, 77802.0,
    82015.0, 86423.0, 91034.0, 95855.0, 100890.0, 106160.0, 111660.0,
    117400.0, 123400.0, 129650.0, 136170.0, 142980.0, 150070.0, 157460.0,
    165160.0, 173180.0, 181530.0, 190220.0, 199260.0, 208670.0, 218450.0,
    228610.0, 239180.0, 250160.0, 261560.0, 273400.0, 285700.0, 298450.0,
    311690.0, 325420.0, 339650.0, 354410.0, 369710.0, 385560.0, 401980.0,
    418980.0, 436590.0, 454810.0, 473670.0, 493170.0, 513350.0, 534220.0,
    555800.0, 578090.0, 601130.0, 624940.0, 649530.0, 674920.0, 701130.0,
    728190.0, 756110.0, 784920.0, 814630.0, 845280.0, 876880.0, 909450.0,
    943020.0, 977610.0, 1013250.0, 1049940.0, 1087740.0, 1087740.0,
];

const RELHUM_MINTEMP: f64 = 173.16;
const RELHUM_MAXTEMP: f64 = 375.16;

// ============================================================
// Dew point temperature
// ============================================================

/// Dew point temperature from temperature (K) and relative humidity (0-100%).
/// Formula from Dutton's "Ceaseless Wind".
#[inline]
pub fn dewtemp(tk: f64, rh: f64) -> f64 {
    let lhv = (597.3 - 0.57 * (tk - 273.0)) / GCX;
    tk * lhv / (lhv - tk * (rh * 0.01).ln())
}

// ============================================================
// Heat index (NWS)
// ============================================================

/// NWS heat index coefficients (default, for T >= 80F).
const HI_C_DEFAULT: [f64; 9] = [
    -42.379, 2.04901523, 10.14333127, -0.22475541, -0.00683783,
    -0.05481717, 0.00122874, 0.00085282, -0.00000199,
];

/// Alternate heat index coefficients (for 70F < T < 115F, 0% < RH < 80%).
const HI_C_ALT: [f64; 9] = [
    0.363445176, 0.988622465, 4.777114035, -0.114037667, -0.000850208,
    -0.020716198, 0.000687678, 0.000274954, 0.0,
];

/// NWS polynomial: c0 + c1*T + c2*RH + c3*T*RH + c4*T^2 + c5*RH^2 + c6*T^2*RH + c7*T*RH^2 + c8*T^2*RH^2
#[inline]
fn nws_eqn(t: f64, rh: f64, c: &[f64; 9]) -> f64 {
    c[0] + c[1] * t + c[2] * rh + c[3] * t * rh
        + c[4] * t * t + c[5] * rh * rh
        + c[6] * t * t * rh + c[7] * t * rh * rh
        + c[8] * t * t * rh * rh
}

/// Heat index (Fahrenheit in, Fahrenheit out).
/// Uses NWS formulation with Steadman + Rothfusz regression + adjustments.
pub fn heat_index(temperature: f64, relative_humidity: f64, alternate_coeffs: bool) -> f64 {
    let t = temperature;
    let rh = relative_humidity;

    let (c, crit) = if alternate_coeffs {
        (&HI_C_ALT, 70.0)
    } else {
        (&HI_C_DEFAULT, 79.0)
    };

    // Steadman initial
    let steadman = 61.0 + (t - 68.0) * 1.2 + rh * 0.094;
    let hi_initial = (steadman + t) / 2.0;

    if t < 40.0 {
        return t;
    }

    if hi_initial < crit {
        return hi_initial;
    }

    let mut hi = nws_eqn(t, rh, c);

    // Low humidity adjustment
    if !alternate_coeffs && rh <= 13.0 && t >= 80.0 && t <= 112.0 {
        let adj = ((13.0 - rh) / 4.0) * ((17.0 - (t - 95.0).abs()) / 17.0).sqrt();
        hi -= adj;
    }

    // High humidity adjustment
    if !alternate_coeffs && rh > 85.0 && t >= 80.0 && t <= 87.0 {
        let adj = ((rh - 85.0) / 10.0) * ((87.0 - t) / 5.0);
        hi += adj;
    }

    hi
}

// ============================================================
// Relative humidity
// ============================================================

/// Relative humidity over ice (Alduchov & Eskridge 1996).
/// t: temperature (K), w: mixing ratio (kg/kg), p: pressure (Pa).
#[inline]
pub fn relhum_ice(t: f64, w: f64, p: f64) -> f64 {
    const ES0: f64 = 6.1128;
    const A: f64 = 22.571;
    const B: f64 = 273.71;

    let est = ES0 * ((A * (t - T0)) / ((t - T0) + B)).exp();
    let qst = (EP * est) / ((p * 0.01) - ONEMEP * est);
    100.0 * (w / qst)
}

/// Relative humidity over water (Murray 1967, Magnus-Tetens).
/// t: temperature (K), w: mixing ratio (kg/kg), p: pressure (Pa).
#[inline]
pub fn relhum_water(t: f64, w: f64, p: f64) -> f64 {
    const ES0: f64 = 6.1128;
    const A: f64 = 17.269;
    const B: f64 = 35.86;

    let est = ES0 * ((A * (t - T0)) / (t - B)).exp();
    let qst = (EP * est) / ((p * 0.01) - ONEMEP * est);
    100.0 * (w / qst)
}

/// Relative humidity using NCL lookup table interpolation.
/// temperature: K, mixing_ratio: kg/kg, pressure: Pa.
#[inline]
pub fn relhum(temperature: f64, mixing_ratio: f64, pressure: f64) -> f64 {
    let t = temperature.clamp(RELHUM_MINTEMP, RELHUM_MAXTEMP);
    let it = (t - RELHUM_MINTEMP) as usize;
    let it = it.min(RELHUM_TABLE.len() - 2); // safety clamp
    let t2 = RELHUM_MINTEMP + it as f64;

    let es = (t2 + 1.0 - t) * RELHUM_TABLE[it] + (t - t2) * RELHUM_TABLE[it + 1];
    let es = es * 0.1; // convert to hPa

    let rh = (mixing_ratio * (pressure - 0.378 * es) / (0.622 * es)) * 100.0;
    rh.max(0.0001)
}

// ============================================================
// Maximum daylight hours (FAO-56)
// ============================================================

/// Maximum daylight hours for a given day of year and latitude.
/// jday: day of year (1-365), lat_deg: latitude in degrees.
#[inline]
pub fn max_daylight(jday: f64, lat_deg: f64) -> f64 {
    const PI2YR: f64 = 2.0 * std::f64::consts::PI / 365.0;
    const CON: f64 = 24.0 / std::f64::consts::PI;

    let sdec = 0.409 * (PI2YR * jday - 1.39).sin();
    let lat_rad = lat_deg.to_radians();
    let arg = -lat_rad.tan() * sdec.tan();
    // Clamp to [-1, 1] for arccos (polar regions)
    let ws = arg.clamp(-1.0, 1.0).acos();
    CON * ws
}

// ============================================================
// Saturation vapor pressure (Tetens, FAO-56)
// ============================================================

/// Saturation vapor pressure (kPa) from temperature in Fahrenheit.
/// Returns NaN for T_celsius <= 0.
#[inline]
pub fn saturation_vapor_pressure(temperature_f: f64) -> f64 {
    let tc = (temperature_f - 32.0) * 5.0 / 9.0;
    if tc > 0.0 {
        0.6108 * ((17.27 * tc) / (tc + 237.3)).exp()
    } else {
        f64::NAN
    }
}

/// Actual saturation vapor pressure (kPa) from dew point temperature (F).
#[inline]
pub fn actual_saturation_vapor_pressure(tdew_f: f64) -> f64 {
    saturation_vapor_pressure(tdew_f)
}

/// Slope of saturation vapor pressure curve (kPa/C) from temperature (F).
/// FAO-56 Equation 13.
#[inline]
pub fn saturation_vapor_pressure_slope(temperature_f: f64) -> f64 {
    let tc = (temperature_f - 32.0) * 5.0 / 9.0;
    if tc > 0.0 {
        let svp = 0.6108 * ((17.27 * tc) / (tc + 237.3)).exp();
        4096.0 * svp / ((tc + 237.3) * (tc + 237.3))
    } else {
        f64::NAN
    }
}

/// Psychrometric constant (kPa/C) from pressure (kPa). FAO-56 Equation 8.
#[inline]
pub fn psychrometric_constant(pressure_kpa: f64) -> f64 {
    0.66474e-3 * pressure_kpa
}

// ============================================================
// Delta pressure (pressure layer thickness)
// ============================================================

/// Compute pressure layer thickness for 1D pressure levels.
/// pressure_lev: sorted pressure levels (Pa), must be monotonic.
/// surface_pressure: scalar surface pressure (Pa).
/// pressure_top: top of atmosphere pressure (Pa), defaults to min(pressure_lev).
pub fn delta_pressure_1d(
    pressure_lev: &[f64],
    surface_pressure: f64,
    pressure_top: Option<f64>,
) -> Vec<f64> {
    let n = pressure_lev.len();
    if n == 0 {
        return vec![];
    }

    // Determine if decreasing (and flip if so)
    let decreasing = n > 1 && pressure_lev[0] > pressure_lev[n - 1];
    let plev: Vec<f64> = if decreasing {
        pressure_lev.iter().rev().copied().collect()
    } else {
        pressure_lev.to_vec()
    };

    let ptop = pressure_top.unwrap_or_else(|| plev.iter().cloned().fold(f64::INFINITY, f64::min));

    let mut dp = vec![0.0; n];

    // Find which levels are below surface pressure
    let mut last_valid = 0;
    for i in 0..n {
        if plev[i] <= surface_pressure {
            last_valid = i;
        }
    }

    if last_valid == 0 && n > 0 {
        dp[0] = surface_pressure - ptop;
    } else {
        // Top level
        dp[0] = (plev[0] + plev[1]) / 2.0 - ptop;

        // Middle levels
        for k in 1..last_valid {
            dp[k] = (plev[k + 1] - plev[k - 1]) / 2.0;
        }

        // Bottom level
        if last_valid > 0 {
            dp[last_valid] = surface_pressure - (plev[last_valid] + plev[last_valid - 1]) / 2.0;
        }
    }

    if decreasing {
        dp.reverse();
    }
    dp
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol || (a.is_nan() && b.is_nan())
    }

    #[test]
    fn test_dewtemp() {
        // 300K, 50% RH → dew point should be ~288K
        let td = dewtemp(300.0, 50.0);
        assert!(approx(td, 288.7, 1.0), "dewtemp={}", td);
    }

    #[test]
    fn test_heat_index_below_threshold() {
        assert_eq!(heat_index(35.0, 50.0, false), 35.0);
    }

    #[test]
    fn test_heat_index_normal() {
        let hi = heat_index(95.0, 50.0, false);
        assert!(hi > 95.0 && hi < 115.0, "heat_index={}", hi);
    }

    #[test]
    fn test_relhum_ice_basic() {
        let rh = relhum_ice(260.0, 0.001, 85000.0);
        assert!(rh > 0.0 && rh < 200.0, "relhum_ice={}", rh);
    }

    #[test]
    fn test_relhum_water_basic() {
        let rh = relhum_water(290.0, 0.01, 101325.0);
        assert!(rh > 0.0 && rh < 200.0, "relhum_water={}", rh);
    }

    #[test]
    fn test_relhum_table() {
        let rh = relhum(300.0, 0.01, 101325.0);
        assert!(rh > 0.0 && rh < 200.0, "relhum={}", rh);
    }

    #[test]
    fn test_max_daylight_equinox() {
        // Equinox at equator → ~12 hours
        let dl = max_daylight(80.0, 0.0); // ~March 21
        assert!(approx(dl, 12.0, 0.5), "max_daylight={}", dl);
    }

    #[test]
    fn test_svp() {
        // 68F = 20C → SVP ≈ 2.338 kPa
        let svp = saturation_vapor_pressure(68.0);
        assert!(approx(svp, 2.338, 0.01), "svp={}", svp);
    }

    #[test]
    fn test_svp_slope() {
        let s = saturation_vapor_pressure_slope(68.0);
        assert!(s > 0.0 && s < 1.0, "svp_slope={}", s);
    }

    #[test]
    fn test_psychrometric() {
        // At 101.3 kPa → γ ≈ 0.0673
        let g = psychrometric_constant(101.3);
        assert!(approx(g, 0.0673, 0.001), "psychrometric={}", g);
    }
}
