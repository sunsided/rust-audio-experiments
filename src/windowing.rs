use apodize;
use num::NumCast;
use std::iter;
use std::mem::MaybeUninit;

pub trait Window<T>
where
    T: NumCast,
{
    fn fill_vec(&self, window: &mut Vec<T>) {
        todo!(); // doesn't work like this:
                 // let mut window_uninit: Vec<MaybeUninit<T>> = unsafe { std::mem::transmute(window) };
                 // self.fill_vec_uninit(&mut window_uninit);
    }

    fn fill_vec_uninit(&self, window: &mut Vec<MaybeUninit<T>>);

    fn to_vec(&self, window_size: usize) -> Vec<T> {
        let mut window: Vec<MaybeUninit<T>> = Vec::with_capacity(window_size);
        unsafe {
            window.set_len(window_size);
        }
        self.fill_vec_uninit(&mut window);
        unsafe { std::mem::transmute(window) }
    }
}

/// A Hann window function, also known as a Raised Cosine window.
pub struct HannWindow;

impl<T> Window<T> for HannWindow
where
    T: NumCast,
{
    /*
    fn fill_vec(&self, window: &mut Vec<T>) {
        let window_size = window.len();
        for (x, w) in window.iter_mut().zip(apodize::hanning_iter(window_size)) {
            *x = NumCast::from(w).unwrap();
        }
    }
    */

    fn fill_vec_uninit(&self, window: &mut Vec<MaybeUninit<T>>) {
        let window_size = window.len();
        for (i, w) in apodize::hanning_iter(window_size).enumerate() {
            window[i].write(NumCast::from(w).unwrap());
        }
    }

    /*
    fn to_vec(&self, window_size: usize) -> Vec<T> {
        apodize::hanning_iter(window_size)
            .map(NumCast::from)
            .map(|x| x.unwrap())
            .collect()
    }
    */
}

impl Default for HannWindow {
    fn default() -> Self {
        Self {}
    }
}

/// A Hamming window function.
pub struct HammingWindow;

impl<T> Window<T> for HammingWindow
where
    T: NumCast,
{
    fn fill_vec(&self, window: &mut Vec<T>) {
        todo!()
    }

    fn fill_vec_uninit(&self, window: &mut Vec<MaybeUninit<T>>) {
        todo!()
    }

    fn to_vec(&self, window_size: usize) -> Vec<T> {
        apodize::hamming_iter(window_size)
            .map(NumCast::from)
            .map(|x| x.unwrap())
            .collect()
    }
}

impl Default for HammingWindow {
    fn default() -> Self {
        Self {}
    }
}

/// A Blackman window function.
pub struct BlackmanWindow;

impl<T> Window<T> for BlackmanWindow
where
    T: NumCast,
{
    fn fill_vec(&self, window: &mut Vec<T>) {
        todo!()
    }

    fn fill_vec_uninit(&self, window: &mut Vec<MaybeUninit<T>>) {
        todo!()
    }

    fn to_vec(&self, window_size: usize) -> Vec<T> {
        apodize::blackman_iter(window_size)
            .map(NumCast::from)
            .map(|x| x.unwrap())
            .collect()
    }
}

impl Default for BlackmanWindow {
    fn default() -> Self {
        Self {}
    }
}

/// A Nutall window function.
pub struct NuttallWindow;

impl<T> Window<T> for NuttallWindow
where
    T: NumCast,
{
    fn fill_vec(&self, window: &mut Vec<T>) {
        todo!()
    }

    fn fill_vec_uninit(&self, window: &mut Vec<MaybeUninit<T>>) {
        todo!()
    }

    fn to_vec(&self, window_size: usize) -> Vec<T> {
        apodize::nuttall_iter(window_size)
            .map(NumCast::from)
            .map(|x| x.unwrap())
            .collect()
    }
}

impl Default for NuttallWindow {
    fn default() -> Self {
        Self {}
    }
}

/// A Bartlett (triangular) window function.
pub struct BartlettWindow;

impl<T> Window<T> for BartlettWindow
where
    T: NumCast,
{
    fn fill_vec(&self, window: &mut Vec<T>) {
        todo!()
    }

    fn fill_vec_uninit(&self, window: &mut Vec<MaybeUninit<T>>) {
        todo!()
    }

    fn to_vec(&self, window_size: usize) -> Vec<T> {
        apodize::triangular_iter(window_size)
            .map(NumCast::from)
            .map(|x| x.unwrap())
            .collect()
    }
}

impl Default for BartlettWindow {
    fn default() -> Self {
        Self {}
    }
}

/// A Boxcar window function (i.e. similar to not using a window at all).
pub struct BoxcarWindow;

impl<T> Window<T> for BoxcarWindow
where
    T: NumCast,
{
    fn fill_vec(&self, window: &mut Vec<T>) {
        todo!()
    }

    fn fill_vec_uninit(&self, window: &mut Vec<MaybeUninit<T>>) {
        todo!()
    }

    fn to_vec(&self, window_size: usize) -> Vec<T> {
        iter::repeat(1.0)
            .take(window_size)
            .map(NumCast::from)
            .map(|x| x.unwrap())
            .collect()
    }
}

impl Default for BoxcarWindow {
    fn default() -> Self {
        Self {}
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
        let window: Vec<f32> = HannWindow::default().to_vec(7);
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
        let window: Vec<f32> = HannWindow::default().to_vec(2);
        let expected = vec![0.0, 0.0];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn hann_10() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = HannWindow::default().to_vec(10);
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
        let window: Vec<f32> = HannWindow::default().to_vec(11);
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
        let window: Vec<f32> = HammingWindow::default().to_vec(2);
        let expected = vec![0.08000000000000002, 0.08000000000000002];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn hamming_3() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = HammingWindow::default().to_vec(3);
        let expected = vec![0.08000000000000002, 1.0, 0.08000000000000002];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn hamming_10() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = HammingWindow::default().to_vec(10);
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
        let window: Vec<f32> = BlackmanWindow::default().to_vec(2);
        let expected = vec![0.000060000000000004494, 0.000060000000000004494];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn blackman_3() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = BlackmanWindow::default().to_vec(3);
        let expected = vec![0.000060000000000004494, 1.0, 0.000060000000000004494];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn blackman_10() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = BlackmanWindow::default().to_vec(10);
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
        let window: Vec<f32> = NuttallWindow::default().to_vec(2);
        let expected = vec![0.0, 0.0];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn nuttall_3() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = NuttallWindow::default().to_vec(3);
        let expected = vec![0.0, 1.0, 0.0];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn nuttall_10() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = NuttallWindow::default().to_vec(10);
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
        let window: Vec<f32> = BartlettWindow::default().to_vec(1);
        let expected = vec![1.0];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn bartlett_2() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = BartlettWindow::default().to_vec(2);
        let expected = vec![0.5, 0.5];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn bartlett_3() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = BartlettWindow::default().to_vec(3);
        let expected = vec![0.3333333333333333, 1.0, 0.3333333333333333];
        assert_ulps_eq!(window.as_slice(), expected.as_slice(), max_ulps = 10);
    }

    #[test]
    fn bartlett_10() {
        // Note: Test case migrated from apodize crate.
        let window: Vec<f32> = BartlettWindow::default().to_vec(10);
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
        let window: Vec<f32> = BartlettWindow::default().to_vec(11);
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
