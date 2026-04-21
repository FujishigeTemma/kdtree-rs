#[inline]
pub fn point(data: &[f64], ndim: usize, index: usize) -> &[f64] {
    let start = index * ndim;
    &data[start..start + ndim]
}
