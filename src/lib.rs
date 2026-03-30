use pyo3::prelude::*;

mod py_meteorology;
mod py_gradient;
mod py_interpolation;

#[pymodule]
fn _geocat_rs(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let met = PyModule::new(py, "meteorology")?;
    py_meteorology::register(py, &met)?;
    m.add_submodule(&met)?;

    let grad = PyModule::new(py, "gradient")?;
    py_gradient::register(py, &grad)?;
    m.add_submodule(&grad)?;

    let interp = PyModule::new(py, "interpolation")?;
    py_interpolation::register(py, &interp)?;
    m.add_submodule(&interp)?;

    Ok(())
}
