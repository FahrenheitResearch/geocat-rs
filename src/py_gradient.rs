use geocat_rs::gradient;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rad_lat_wgs84_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(arc_lat_wgs84_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(arc_lon_wgs84_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(rad_lat_wgs84_array, m)?)?;
    m.add_function(wrap_pyfunction!(arc_lat_wgs84_array, m)?)?;
    m.add_function(wrap_pyfunction!(arc_lon_wgs84_array, m)?)?;
    Ok(())
}

#[pyfunction]
fn rad_lat_wgs84_scalar(lat: f64) -> f64 {
    gradient::rad_lat_wgs84(lat)
}

#[pyfunction]
fn arc_lat_wgs84_scalar(lat: f64) -> f64 {
    gradient::arc_lat_wgs84(lat)
}

#[pyfunction]
fn arc_lon_wgs84_scalar(lon: f64, lat: f64) -> f64 {
    gradient::arc_lon_wgs84(lon, lat)
}

#[pyfunction]
fn rad_lat_wgs84_array<'py>(
    py: Python<'py>,
    lat: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let lat = lat.as_slice()?;
    let result: Vec<f64> = lat.par_iter()
        .map(|&l| gradient::rad_lat_wgs84(l))
        .collect();
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
fn arc_lat_wgs84_array<'py>(
    py: Python<'py>,
    lat: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let lat = lat.as_slice()?;
    let result: Vec<f64> = lat.par_iter()
        .map(|&l| gradient::arc_lat_wgs84(l))
        .collect();
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
fn arc_lon_wgs84_array<'py>(
    py: Python<'py>,
    lon: PyReadonlyArray1<'py, f64>,
    lat: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let lon = lon.as_slice()?;
    let lat = lat.as_slice()?;
    let result: Vec<f64> = (0..lon.len())
        .into_par_iter()
        .map(|i| gradient::arc_lon_wgs84(lon[i], lat[i]))
        .collect();
    Ok(PyArray1::from_vec(py, result))
}
