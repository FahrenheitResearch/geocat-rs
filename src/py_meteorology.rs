use geocat_rs::meteorology;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;

/// Validate that two arrays have the same length.
fn check_len2(a: usize, b: usize, name_a: &str, name_b: &str) -> PyResult<()> {
    if a != b {
        Err(PyValueError::new_err(format!(
            "{} and {} must have the same length ({} != {})", name_a, name_b, a, b
        )))
    } else {
        Ok(())
    }
}

fn check_len3(a: usize, b: usize, c: usize, na: &str, nb: &str, nc: &str) -> PyResult<()> {
    if a != b || b != c {
        Err(PyValueError::new_err(format!(
            "{}, {}, and {} must have the same length ({}, {}, {})", na, nb, nc, a, b, c
        )))
    } else {
        Ok(())
    }
}

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dewtemp_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(heat_index_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(relhum_ice_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(relhum_water_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(relhum_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(max_daylight_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(saturation_vapor_pressure_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(saturation_vapor_pressure_slope_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(psychrometric_constant_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(dewtemp_array, m)?)?;
    m.add_function(wrap_pyfunction!(heat_index_array, m)?)?;
    m.add_function(wrap_pyfunction!(relhum_ice_array, m)?)?;
    m.add_function(wrap_pyfunction!(relhum_water_array, m)?)?;
    m.add_function(wrap_pyfunction!(relhum_array, m)?)?;
    m.add_function(wrap_pyfunction!(max_daylight_grid, m)?)?;
    m.add_function(wrap_pyfunction!(saturation_vapor_pressure_array, m)?)?;
    m.add_function(wrap_pyfunction!(saturation_vapor_pressure_slope_array, m)?)?;
    m.add_function(wrap_pyfunction!(psychrometric_constant_array, m)?)?;
    Ok(())
}

// ---- Scalar functions ----

#[pyfunction]
fn dewtemp_scalar(tk: f64, rh: f64) -> f64 { meteorology::dewtemp(tk, rh) }
#[pyfunction]
fn heat_index_scalar(temperature: f64, relative_humidity: f64, alternate_coeffs: bool) -> f64 {
    meteorology::heat_index(temperature, relative_humidity, alternate_coeffs)
}
#[pyfunction]
fn relhum_ice_scalar(t: f64, w: f64, p: f64) -> f64 { meteorology::relhum_ice(t, w, p) }
#[pyfunction]
fn relhum_water_scalar(t: f64, w: f64, p: f64) -> f64 { meteorology::relhum_water(t, w, p) }
#[pyfunction]
fn relhum_scalar(temperature: f64, mixing_ratio: f64, pressure: f64) -> f64 {
    meteorology::relhum(temperature, mixing_ratio, pressure)
}
#[pyfunction]
fn max_daylight_scalar(jday: f64, lat: f64) -> f64 { meteorology::max_daylight(jday, lat) }
#[pyfunction]
fn saturation_vapor_pressure_scalar(temperature_f: f64) -> f64 {
    meteorology::saturation_vapor_pressure(temperature_f)
}
#[pyfunction]
fn saturation_vapor_pressure_slope_scalar(temperature_f: f64) -> f64 {
    meteorology::saturation_vapor_pressure_slope(temperature_f)
}
#[pyfunction]
fn psychrometric_constant_scalar(pressure_kpa: f64) -> f64 {
    meteorology::psychrometric_constant(pressure_kpa)
}

// ---- Array functions (rayon-parallel, with input validation) ----

#[pyfunction]
fn dewtemp_array<'py>(
    py: Python<'py>,
    tk: PyReadonlyArray1<'py, f64>,
    rh: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let tk = tk.as_slice()?;
    let rh = rh.as_slice()?;
    check_len2(tk.len(), rh.len(), "tk", "rh")?;
    let result: Vec<f64> = (0..tk.len()).into_par_iter()
        .map(|i| meteorology::dewtemp(tk[i], rh[i])).collect();
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
fn heat_index_array<'py>(
    py: Python<'py>,
    temperature: PyReadonlyArray1<'py, f64>,
    relative_humidity: PyReadonlyArray1<'py, f64>,
    alternate_coeffs: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let t = temperature.as_slice()?;
    let rh = relative_humidity.as_slice()?;
    check_len2(t.len(), rh.len(), "temperature", "relative_humidity")?;
    let result: Vec<f64> = (0..t.len()).into_par_iter()
        .map(|i| meteorology::heat_index(t[i], rh[i], alternate_coeffs)).collect();
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
fn relhum_ice_array<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    w: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let t = t.as_slice()?;
    let w = w.as_slice()?;
    let p = p.as_slice()?;
    check_len3(t.len(), w.len(), p.len(), "t", "w", "p")?;
    let result: Vec<f64> = (0..t.len()).into_par_iter()
        .map(|i| meteorology::relhum_ice(t[i], w[i], p[i])).collect();
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
fn relhum_water_array<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    w: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let t = t.as_slice()?;
    let w = w.as_slice()?;
    let p = p.as_slice()?;
    check_len3(t.len(), w.len(), p.len(), "t", "w", "p")?;
    let result: Vec<f64> = (0..t.len()).into_par_iter()
        .map(|i| meteorology::relhum_water(t[i], w[i], p[i])).collect();
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
fn relhum_array<'py>(
    py: Python<'py>,
    temperature: PyReadonlyArray1<'py, f64>,
    mixing_ratio: PyReadonlyArray1<'py, f64>,
    pressure: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let t = temperature.as_slice()?;
    let w = mixing_ratio.as_slice()?;
    let p = pressure.as_slice()?;
    check_len3(t.len(), w.len(), p.len(), "temperature", "mixing_ratio", "pressure")?;
    let result: Vec<f64> = (0..t.len()).into_par_iter()
        .map(|i| meteorology::relhum(t[i], w[i], p[i])).collect();
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
fn max_daylight_grid<'py>(
    py: Python<'py>,
    jday: PyReadonlyArray1<'py, f64>,
    lat: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let jday = jday.as_slice()?;
    let lat = lat.as_slice()?;
    let nj = jday.len();
    let nl = lat.len();
    let result: Vec<f64> = (0..nj * nl).into_par_iter()
        .map(|idx| meteorology::max_daylight(jday[idx / nl], lat[idx % nl])).collect();
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
fn saturation_vapor_pressure_array<'py>(
    py: Python<'py>,
    temperature: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let t = temperature.as_slice()?;
    let result: Vec<f64> = t.par_iter()
        .map(|&ti| meteorology::saturation_vapor_pressure(ti)).collect();
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
fn saturation_vapor_pressure_slope_array<'py>(
    py: Python<'py>,
    temperature: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let t = temperature.as_slice()?;
    let result: Vec<f64> = t.par_iter()
        .map(|&ti| meteorology::saturation_vapor_pressure_slope(ti)).collect();
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
fn psychrometric_constant_array<'py>(
    py: Python<'py>,
    pressure: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = pressure.as_slice()?;
    let result: Vec<f64> = p.par_iter()
        .map(|&pi| meteorology::psychrometric_constant(pi)).collect();
    Ok(PyArray1::from_vec(py, result))
}
