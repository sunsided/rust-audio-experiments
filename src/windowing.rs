use num::traits::Zero;
use num::NumCast;
use std::iter;
use std::num::NonZeroUsize;
use std::ops::Add;

#[derive(Debug)]
pub struct Window<T>
where
    T: Zero,
{
    /// The vector of values.
    pub window: Vec<T>,
    /// The sum of values in the window.
    pub sum: Option<T>,
}

impl<T> Window<T>
where
    T: Zero,
{
    pub fn clear(&mut self) {
        self.window.clear();
        self.sum = None;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.window.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.window.iter()
    }
}

impl<T> Default for Window<T>
where
    T: Zero,
{
    fn default() -> Self {
        Self {
            window: Vec::default(),
            sum: None,
        }
    }
}

impl<T> From<Vec<T>> for Window<T>
where
    T: Zero + Add + Clone,
{
    fn from(vec: Vec<T>) -> Self {
        if vec.is_empty() {
            return Self {
                window: Vec::default(),
                sum: None,
            };
        }
        let sum = vec.iter().fold(T::zero(), |a, b| a.add(b.clone()));
        Window {
            window: vec,
            sum: Some(sum),
        }
    }
}

impl<T> From<Window<T>> for Vec<T>
    where
        T: Zero,
{
    fn from(value: Window<T>) -> Self {
        value.window
    }
}

pub trait WindowFunction<T>
where
    T: NumCast + Zero,
{
    fn to_vec(&self, window_size: NonZeroUsize) -> Window<T>;
}

/// A Hann window function, also known as a Raised Cosine window.
#[derive(Default, Copy, Clone)]
pub struct HannWindow;

impl<T> WindowFunction<T> for HannWindow
where
    T: NumCast + Zero + Clone,
{
    fn to_vec(&self, window_size: NonZeroUsize) -> Window<T> {
        let window: Vec<T> = apodize::hanning_iter(window_size.get())
            .map(NumCast::from)
            .map(|x| x.unwrap())
            .collect();
        window.into()
    }
}

/// A Hamming window function.
#[derive(Default, Copy, Clone)]
pub struct HammingWindow;

impl<T> WindowFunction<T> for HammingWindow
where
    T: NumCast + Zero + Clone,
{
    fn to_vec(&self, window_size: NonZeroUsize) -> Window<T> {
        let window: Vec<T> = apodize::hamming_iter(window_size.get())
            .map(NumCast::from)
            .map(|x| x.unwrap())
            .collect();
        window.into()
    }
}

/// A Blackman window function.
#[derive(Default, Copy, Clone)]
pub struct BlackmanWindow;

impl<T> WindowFunction<T> for BlackmanWindow
where
    T: NumCast + Zero + Clone,
{
    fn to_vec(&self, window_size: NonZeroUsize) -> Window<T> {
        let window: Vec<T> = apodize::blackman_iter(window_size.get())
            .map(NumCast::from)
            .map(|x| x.unwrap())
            .collect();
        window.into()
    }
}

/// A Nutall window function.
#[derive(Default, Copy, Clone)]
pub struct NuttallWindow;

impl<T> WindowFunction<T> for NuttallWindow
where
    T: NumCast + Zero + Clone,
{
    fn to_vec(&self, window_size: NonZeroUsize) -> Window<T> {
        let window: Vec<T> = apodize::nuttall_iter(window_size.get())
            .map(NumCast::from)
            .map(|x| x.unwrap())
            .collect();
        window.into()
    }
}

/// A Bartlett (triangular) window function.
#[derive(Default, Copy, Clone)]
pub struct BartlettWindow;

impl<T> WindowFunction<T> for BartlettWindow
where
    T: NumCast + Zero + Clone,
{
    fn to_vec(&self, window_size: NonZeroUsize) -> Window<T> {
        let window: Vec<T> = apodize::triangular_iter(window_size.get())
            .map(NumCast::from)
            .map(|x| x.unwrap())
            .collect();
        window.into()
    }
}

/// A Boxcar window function (i.e. similar to not using a window at all).
#[derive(Default, Copy, Clone)]
pub struct BoxcarWindow;

impl<T> WindowFunction<T> for BoxcarWindow
where
    T: NumCast + Zero + Clone,
{
    fn to_vec(&self, window_size: NonZeroUsize) -> Window<T> {
        let window: Vec<T> = iter::repeat(1.0)
            .take(window_size.get())
            .map(NumCast::from)
            .map(|x| x.unwrap())
            .collect();
        window.into()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn hann_apodization() {
        // Note: Test case migrated from apodize crate.

        // Create a Hann window of length 7.
        let window: Vec<f32> = HannWindow::default()
            .to_vec(NonZeroUsize::new(7).unwrap())
            .into();
        let expected = vec![
            0.0,
            0.24999999999999994,
            0.7499999999999999,
            1.0,
            0.7500000000000002,
            0.25,
            0.0,
        ];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);

        // Create test data and apply the window to it ("apodize").
        let data: Vec<f32> = vec![1., 2., 3., 4., 5., 6., 7.];
        let windowed_data = apply_window(data, window);

        let expected = vec![
            0.0,
            0.4999999999999999,
            2.2499999999999996,
            4.0,
            3.750000000000001,
            1.5,
            0.0,
        ];
        assert_ulps_eq!(windowed_data.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn hann_2() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = HannWindow::default()
            .to_vec(NonZeroUsize::new(2).unwrap())
            .into();
        let expected = vec![0.0, 0.0];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn hann_10() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = HannWindow::default()
            .to_vec(NonZeroUsize::new(10).unwrap())
            .into();
        let expected = vec![
            0.0,
            0.11697777844051094,
            0.4131759111665348,
            0.7499999999999999,
            0.9698463103929542,
            0.9698463103929542,
            0.7499999999999999,
            0.4131759111665348,
            0.11697777844051094,
            0.0,
        ];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn hann_11() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = HannWindow::default()
            .to_vec(NonZeroUsize::new(11).unwrap())
            .into();
        let expected = vec![
            0.0,
            0.09549150281252627,
            0.3454915028125263,
            0.6545084971874737,
            0.9045084971874737,
            1.0,
            0.9045084971874737,
            0.6545084971874737,
            0.3454915028125264,
            0.09549150281252633,
            0.0,
        ];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn hamming_2() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = HammingWindow::default()
            .to_vec(NonZeroUsize::new(2).unwrap())
            .into();
        let expected = vec![0.08000000000000002, 0.08000000000000002];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn hamming_3() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = HammingWindow::default()
            .to_vec(NonZeroUsize::new(3).unwrap())
            .into();
        let expected = vec![0.08000000000000002, 1.0, 0.08000000000000002];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn hamming_10() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = HammingWindow::default()
            .to_vec(NonZeroUsize::new(10).unwrap())
            .into();
        let expected = vec![
            0.08000000000000002,
            0.1876195561652701,
            0.46012183827321207,
            0.7699999999999999,
            0.9722586055615179,
            0.9722586055615179,
            0.7700000000000002,
            0.46012183827321224,
            0.1876195561652702,
            0.08000000000000002,
        ];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn blackman_2() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = BlackmanWindow::default()
            .to_vec(NonZeroUsize::new(2).unwrap())
            .into();
        let expected = vec![0.000060000000000004494, 0.000060000000000004494];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn blackman_3() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = BlackmanWindow::default()
            .to_vec(NonZeroUsize::new(3).unwrap())
            .into();
        let expected = vec![0.000060000000000004494, 1.0, 0.000060000000000004494];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn blackman_10() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = BlackmanWindow::default()
            .to_vec(NonZeroUsize::new(10).unwrap())
            .into();
        let expected = vec![
            0.000060000000000004494,
            0.015071173410218106,
            0.14703955786238146,
            0.5205749999999999,
            0.9316592687274005,
            0.9316592687274005,
            0.5205750000000003,
            0.1470395578623816,
            0.015071173410218144,
            0.000060000000000004494,
        ];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn nuttall_2() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = NuttallWindow::default()
            .to_vec(NonZeroUsize::new(2).unwrap())
            .into();
        let expected = vec![0.0, 0.0];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn nuttall_3() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = NuttallWindow::default()
            .to_vec(NonZeroUsize::new(3).unwrap())
            .into();
        let expected = vec![0.0, 1.0, 0.0];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn nuttall_10() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = NuttallWindow::default()
            .to_vec(NonZeroUsize::new(10).unwrap())
            .into();
        let expected = vec![
            0.0,
            0.013748631,
            0.14190082,
            0.51474607,
            0.9305606,
            0.93056047,
            0.51474595,
            0.14190066,
            0.013748631,
            0.0,
        ];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), epsilon = 0.000001);
    }

    #[test]
    fn bartlett_1() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = BartlettWindow::default()
            .to_vec(NonZeroUsize::new(1).unwrap())
            .into();
        let expected = vec![1.0];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn bartlett_2() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = BartlettWindow::default()
            .to_vec(NonZeroUsize::new(2).unwrap())
            .into();
        let expected = vec![0.5, 0.5];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn bartlett_3() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = BartlettWindow::default()
            .to_vec(NonZeroUsize::new(3).unwrap())
            .into();
        let expected = vec![0.3333333333333333, 1.0, 0.3333333333333333];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn bartlett_10() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = BartlettWindow::default()
            .to_vec(NonZeroUsize::new(10).unwrap())
            .into();
        let expected = vec![
            0.09999999999999998,
            0.30000000000000004,
            0.5,
            0.7,
            0.9,
            0.9,
            0.7,
            0.5,
            0.30000000000000004,
            0.09999999999999998,
        ];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn bartlett_11() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = BartlettWindow::default()
            .to_vec(NonZeroUsize::new(11).unwrap())
            .into();
        let expected = vec![
            0.09090909090909094,
            0.2727272727272727,
            0.4545454545454546,
            0.6363636363636364,
            0.8181818181818181,
            1.,
            0.8181818181818181,
            0.6363636363636364,
            0.4545454545454546,
            0.2727272727272727,
            0.09090909090909094,
        ];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    fn apply_window(data: Vec<f32>, window: Vec<f32>) -> Vec<f32> {
        let mut windowed_data = vec![0.; data.len()];

        for (windowed, (window, data)) in
            windowed_data.iter_mut().zip(window.iter().zip(data.iter()))
        {
            *windowed = *window * *data;
        }
        windowed_data
    }
}
