//! WGS84 gradient calculations from GeoCAT-comp.
//!
//! Taylor series approximations for radius, arc length, and gradients
//! on the WGS84 ellipsoid. Accurate to floating point precision.

const D2R: f64 = 1.74532925199432957692369e-02;

/// WGS84 radius at a given latitude (degrees). Degree-48 Taylor series.
#[inline]
pub fn rad_lat_wgs84(lat: f64) -> f64 {
    let l2 = lat * lat;
    // Horner-like evaluation of even-power polynomial
    // Coefficients from geocat.comp.gradient._rad_lat_wgs84
    8.05993093251779959604912e-107 * l2.powi(24)       // lat^48
    - 1.26581811418535723456176e-102 * l2.powi(23)     // lat^46
    - 9.10951565776720242021392e-98 * l2.powi(22)      // lat^44
    + 7.49821836126201522491765e-93 * l2.powi(21)      // lat^42
    - 2.27271626922827622448519e-88 * l2.powi(20)      // lat^40
    - 2.71379394439952534826763e-84 * l2.powi(19)      // lat^38
    + 6.14832871773468827219624e-79 * l2.powi(18)      // lat^36
    - 2.84993185787053811467259e-74 * l2.powi(17)      // lat^34
    + 3.04677525449067422497153e-70 * l2.powi(16)      // lat^32
    + 4.32285294849656618043972e-65 * l2.powi(15)      // lat^30
    - 3.09781963775152747156702e-60 * l2.powi(14)      // lat^28
    + 8.20657704323096444142572e-56 * l2.powi(13)      // lat^26
    + 2.18501284875664232841136e-51 * l2.powi(12)      // lat^24
    - 3.14445926697498412770696e-46 * l2.powi(11)      // lat^22
    + 1.33970531285992980064454e-41 * l2.powi(10)      // lat^20
    - 3.64729257084431972115219e-38 * l2.powi(9)       // lat^18
    - 3.19397401776292877914625e-32 * l2.powi(8)       // lat^16
    + 2.09755935705627800528960e-27 * l2.powi(7)       // lat^14
    - 3.93234660296994989005722e-23 * l2.powi(6)       // lat^12
    - 4.67346944715944948938249e-18 * l2.powi(5)       // lat^10
    + 5.23053397296821359502194e-13 * l2.powi(4)       // lat^8
    - 2.61499198791487361911660e-08 * l2.powi(3)       // lat^6
    + 6.57016735969856188450312e-04 * l2.powi(2)       // lat^4
    - 6.50322744547926518806010e00 * l2               // lat^2
    + 6378137.0
}

/// WGS84 arc length from equator to a given latitude (degrees). Degree-49 Taylor series.
#[inline]
pub fn arc_lat_wgs84(lat: f64) -> f64 {
    // Odd-power polynomial: lat * (c0 + c2*lat^2 + c4*lat^4 + ...)
    let l2 = lat * lat;
    lat * (
        111319.490793273572647713
        - 3.78342436432244021373642e-02 * l2
        + 2.29342105667605006614878e-06 * l2.powi(2)
        - 6.52003144319804471773524e-11 * l2.powi(3)
        + 1.01433377184128234589268e-15 * l2.powi(4)
        - 7.41522084948104995730556e-21 * l2.powi(5)
        - 5.27941504241845042397655e-26 * l2.powi(6)
        + 2.44062113577649370926265e-30 * l2.powi(7)
        - 3.27913899018323295027126e-35 * l2.powi(8)
        - 3.35038232340852202916191e-41 * l2.powi(9)
        + 1.11344136742221455419521e-44 * l2.powi(10)
        - 2.38613771319829866779682e-49 * l2.powi(11)
        + 1.52542673636737189511924e-54 * l2.powi(12)
        + 5.30488110085042076590693e-59 * l2.powi(13)
        - 1.86438456247248912270599e-63 * l2.powi(14)
        + 2.43380700099386902295507e-68 * l2.powi(15)
        + 1.61140181088334593373753e-73 * l2.powi(16)
        - 1.42116269649485608032749e-77 * l2.powi(17)
        + 2.90023188160517270041180e-82 * l2.powi(18)
        - 1.21447793719117046293341e-87 * l2.powi(19)
        - 9.67472728333544072524681e-92 * l2.powi(20)
        + 3.04345577761664668327703e-96 * l2.powi(21)
        - 3.53313425533365879608994e-101 * l2.powi(22)
        - 4.70057315402553703681995e-106 * l2.powi(23)
        + 2.87086392358719396475614e-110 * l2.powi(24)
    )
}

/// WGS84 arc length from prime meridian at a given longitude and latitude (degrees).
#[inline]
pub fn arc_lon_wgs84(lon: f64, lat: f64) -> f64 {
    rad_lat_wgs84(lat) * (lat * D2R).cos() * lon * D2R
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rad_lat_equator() {
        let r = rad_lat_wgs84(0.0);
        assert!((r - 6378137.0).abs() < 1.0, "equator radius = {}", r);
    }

    #[test]
    fn test_rad_lat_pole() {
        let r = rad_lat_wgs84(90.0);
        // WGS84 polar radius ≈ 6356752.3
        assert!((r - 6356752.3).abs() < 100.0, "pole radius = {}", r);
    }

    #[test]
    fn test_arc_lat_equator() {
        assert_eq!(arc_lat_wgs84(0.0), 0.0);
    }

    #[test]
    fn test_arc_lat_45() {
        let arc = arc_lat_wgs84(45.0);
        // ~4984 km from equator to 45N
        assert!((arc / 1000.0 - 4984.0).abs() < 50.0, "arc_45 = {} km", arc / 1000.0);
    }

    #[test]
    fn test_arc_lon_equator() {
        let arc = arc_lon_wgs84(1.0, 0.0);
        // 1 degree at equator ≈ 111319 m
        assert!((arc - 111319.0).abs() < 10.0, "arc_lon_equator = {}", arc);
    }
}
