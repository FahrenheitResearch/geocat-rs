use geocat_rs::interpolation;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(temp_extrapolate_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(geo_height_extrapolate_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(pressure_at_hybrid_level_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(temp_extrapolate_array, m)?)?;
    m.add_function(wrap_pyfunction!(geo_height_extrapolate_array, m)?)?;
    m.add_function(wrap_pyfunction!(pressure_at_hybrid_levels_array, m)?)?;
    m.add_function(wrap_pyfunction!(delta_pressure_1d, m)?)?;
    m.add_function(wrap_pyfunction!(interpolate_columns, m)?)?;
    Ok(())
}

#[pyfunction]
fn temp_extrapolate_scalar(t_bot: f64, lev: f64, p_sfc: f64, ps: f64, phi_sfc: f64) -> f64 {
    interpolation::temp_extrapolate(t_bot, lev, p_sfc, ps, phi_sfc)
}

#[pyfunction]
fn geo_height_extrapolate_scalar(t_bot: f64, lev: f64, p_sfc: f64, ps: f64, phi_sfc: f64) -> f64 {
    interpolation::geo_height_extrapolate(t_bot, lev, p_sfc, ps, phi_sfc)
}

#[pyfunction]
fn pressure_at_hybrid_level_scalar(hya: f64, hyb: f64, psfc: f64, p0: f64) -> f64 {
    interpolation::pressure_at_hybrid_level(hya, hyb, psfc, p0)
}

/// Extrapolate temperature for arrays: one level against arrays of t_bot, p_sfc, ps, phi_sfc.
#[pyfunction]
fn temp_extrapolate_array<'py>(
    py: Python<'py>,
    t_bot: PyReadonlyArray1<'py, f64>,
    lev: f64,
    p_sfc: PyReadonlyArray1<'py, f64>,
    ps: PyReadonlyArray1<'py, f64>,
    phi_sfc: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let tb = t_bot.as_slice()?;
    let psfc = p_sfc.as_slice()?;
    let pss = ps.as_slice()?;
    let phi = phi_sfc.as_slice()?;
    let n = tb.len();
    if psfc.len() != n || pss.len() != n || phi.len() != n {
        return Err(PyValueError::new_err(format!(
            "All arrays must have the same length (got {}, {}, {}, {})",
            n, psfc.len(), pss.len(), phi.len()
        )));
    }
    let result: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| interpolation::temp_extrapolate(tb[i], lev, psfc[i], pss[i], phi[i]))
        .collect();
    Ok(PyArray1::from_vec(py, result))
}

/// Extrapolate geopotential height for arrays.
#[pyfunction]
fn geo_height_extrapolate_array<'py>(
    py: Python<'py>,
    t_bot: PyReadonlyArray1<'py, f64>,
    lev: f64,
    p_sfc: PyReadonlyArray1<'py, f64>,
    ps: PyReadonlyArray1<'py, f64>,
    phi_sfc: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let tb = t_bot.as_slice()?;
    let psfc = p_sfc.as_slice()?;
    let pss = ps.as_slice()?;
    let phi = phi_sfc.as_slice()?;
    let n = tb.len();
    if psfc.len() != n || pss.len() != n || phi.len() != n {
        return Err(PyValueError::new_err(format!(
            "All arrays must have the same length (got {}, {}, {}, {})",
            n, psfc.len(), pss.len(), phi.len()
        )));
    }
    let result: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| interpolation::geo_height_extrapolate(tb[i], lev, psfc[i], pss[i], phi[i]))
        .collect();
    Ok(PyArray1::from_vec(py, result))
}

/// Compute pressure at hybrid levels for all levels and surface pressures.
/// hya, hyb: 1D arrays of hybrid coefficients.
/// psfc: 1D array of surface pressures.
/// Returns flattened array of shape (len(hya) * len(psfc)).
#[pyfunction]
fn pressure_at_hybrid_levels_array<'py>(
    py: Python<'py>,
    hya: PyReadonlyArray1<'py, f64>,
    hyb: PyReadonlyArray1<'py, f64>,
    psfc: PyReadonlyArray1<'py, f64>,
    p0: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let hya = hya.as_slice()?;
    let hyb = hyb.as_slice()?;
    let psfc = psfc.as_slice()?;
    let nk = hya.len();
    let ns = psfc.len();
    let result: Vec<f64> = (0..nk * ns)
        .into_par_iter()
        .map(|idx| {
            let k = idx / ns;
            let s = idx % ns;
            interpolation::pressure_at_hybrid_level(hya[k], hyb[k], psfc[s], p0)
        })
        .collect();
    Ok(PyArray1::from_vec(py, result))
}

/// Batch vertical interpolation across many columns (rayon-parallel).
/// xp_flat and data_flat are column-major: (nlev_in * ncols).
/// Returns flattened (nlev_out * ncols) column-major.
#[pyfunction]
fn interpolate_columns<'py>(
    py: Python<'py>,
    xp_flat: PyReadonlyArray1<'py, f64>,
    data_flat: PyReadonlyArray1<'py, f64>,
    new_levels: PyReadonlyArray1<'py, f64>,
    nlev_in: usize,
    ncols: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xp = xp_flat.as_slice()?;
    let data = data_flat.as_slice()?;
    let levels = new_levels.as_slice()?;
    let result = geocat_rs::interpolation::interpolate_columns(xp, data, levels, nlev_in, ncols);
    Ok(PyArray1::from_vec(py, result))
}

/// Delta pressure for 1D pressure levels.
#[pyfunction]
fn delta_pressure_1d<'py>(
    py: Python<'py>,
    pressure_lev: PyReadonlyArray1<'py, f64>,
    surface_pressure: f64,
    pressure_top: Option<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let plev = pressure_lev.as_slice()?;
    let result = geocat_rs::meteorology::delta_pressure_1d(plev, surface_pressure, pressure_top);
    Ok(PyArray1::from_vec(py, result))
}
