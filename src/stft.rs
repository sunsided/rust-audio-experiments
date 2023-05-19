use crate::windowing::{Window, WindowFunction};
use num::traits::Zero;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftNum, FftPlanner};
use std::num::NonZeroUsize;
use std::sync::Arc;
use strider::{SliceRing, SliceRingImpl};

/// An implementation of the Short-Time (Fast) Fourier Transform.
pub struct ShortTimeFourierTransform<T>
where
    T: FftNum + num::Float,
{
    /// The size of the FFT, in time steps.
    pub fft_size: NonZeroUsize,
    /// The size of the time window, in time steps.
    pub window_size: NonZeroUsize,
    /// The number of time steps by which the time window
    /// will be shifted.
    pub step_size: NonZeroUsize,
    /// The size of the FFT, in time steps. The data window will be padded
    /// with zeros before the FFT is applied.
    fft: Arc<dyn Fft<T>>,
    /// The internal ring buffer used to store samples.
    samples: SliceRingImpl<T>,
    /// A buffer used to extract relevant samples from the ring buffer.
    real_input: Vec<T>,
    /// The FFT's complex input/output buffer.
    fft_input_output: Vec<Complex<T>>,
    /// The FFT's scratch buffer.
    fft_scratch: Vec<Complex<T>>,
    /// The window to apply.
    /// - If the window is of non-zero length, values will be multiplied with the [`real_input`]
    ///   before passing them on to the FFT.
    /// - If the window length is zero, the [`real_input`] values will be passed as-is.
    pub window: Window<T>,
}

#[allow(dead_code)]
impl<T> ShortTimeFourierTransform<T>
where
    T: FftNum + num::Float,
{
    /// Initializes a new [`ShortTimeFourierTransform`] instance.
    ///
    /// # Arguments
    /// * [`fft_size`] - The width of the FFT, in time steps. The data window will be clipped or zero-padded to this size before taking the FFT. Must be a positive number.
    /// * [`window_size`] - The size of the data window, in time steps. Must be a positive number.
    /// * [`step_size`] - The number of time steps by which the time window will be shifted; should be a positive number smaller than the [`window_size`] value.
    pub fn new(fft_size: NonZeroUsize, window_size: NonZeroUsize, step_size: NonZeroUsize) -> Self {
        let fft = FftPlanner::new().plan_fft_forward(fft_size.get());
        let scratch_len = fft.get_inplace_scratch_len();

        Self {
            fft,
            fft_size,
            window_size,
            step_size: step_size.clamp(unsafe { NonZeroUsize::new_unchecked(1) }, window_size),
            samples: SliceRingImpl::new(),
            real_input: std::iter::repeat(T::zero())
                .take(window_size.get())
                .collect(),
            fft_input_output: std::iter::repeat(Complex::<T>::zero())
                .take(fft_size.get())
                .collect(),
            fft_scratch: vec![Complex::<T>::zero(); scratch_len],
            window: Window::default(),
        }
    }

    /// Sets the window function to use.
    pub fn set_window(&mut self, window: Option<&dyn WindowFunction<T>>) {
        if let Some(window) = window {
            self.window = window.to_vec(self.window_size);
        } else {
            self.window.clear();
        }
    }

    /// Sets the window function to use.
    pub fn with_window(mut self, window: &dyn WindowFunction<T>) -> Self {
        self.set_window(Some(window));
        self
    }

    /// Appends samples to the internal buffer.
    pub fn append_samples(&mut self, input: &[T]) {
        self.samples.push_many_back(input);
    }

    /// Determines whether the internal buffer contains enough samples to compute the STFT.
    #[inline]
    pub fn contains_enough_to_compute(&self) -> bool {
        self.window_size.get() <= self.samples.len()
    }

    /// Moves on to the the next "slice" by
    /// dropping `self.step_size` samples from the internal buffer.
    pub fn move_to_next_column(&mut self) {
        self.samples.drop_many_front(self.step_size.get());
    }

    /// Determines the number of samples in the internal buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Determines whether the internal buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Determines the corresponding frequencies of a column of the spectrogram.
    ///
    /// # Arguments
    /// * [`sampling_rate`] - sampling frequency.
    pub fn frequencies(&self, sampling_rate: f64) -> Vec<f64> {
        let n = self.output_size();
        (0..n)
            .map(|f| (f as f64) / ((n - 0) as f64) * (sampling_rate / 2.))
            .collect()
    }

    /// Determines the corresponding time of first column of the spectrogram.
    pub fn first_time(&self, sampling_rate: f64) -> f64 {
        (self.window_size.get() as f64) / (sampling_rate * 2.)
    }

    /// Determines time interval between two adjacent columns of the spectrogram.
    pub fn time_interval(&self, sampling_rate: f64) -> f64 {
        (self.step_size.get() as f64) / sampling_rate
    }

    /// Determines the number of samples in the generated FFT.
    #[inline]
    pub fn output_size(&self) -> usize {
        // Since we're dealing with real input data, we only need a single-sided band.
        // We could be adding one here to get account for the DC component at index 0.
        self.fft_size.get() / 2
    }

    /// Computes the Short-Time Fourier Transformation on the current window
    /// and stores the complex results in the provided output buffer.
    ///
    /// # Panics
    /// Panics when `output.len() > self.output_size()`.
    pub fn compute_complex(&mut self, output: &mut [Complex<T>]) {
        assert!(output.len() <= self.output_size());
        self.fetch_window_and_compute_internal();
        for (dst, src) in output.iter_mut().zip(self.fft_input_output.iter()) {
            *dst = *src;
        }
    }

    /// Computes the Short-Time Fourier Transformation on the current window
    /// and stores the complex norms (i.e. magnitudes) in the provided output buffer.
    ///
    /// # Panics
    /// Panics when `self.output_size() != output.len()`.
    pub fn compute_magnitudes(&mut self, output: &mut [T]) {
        assert_eq!(output.len(), self.output_size());
        self.fetch_window_and_compute_internal();
        for (dst, src) in output.iter_mut().zip(self.fft_input_output.iter()) {
            *dst = src.norm();
        }
    }

    /// Computes the Single Side Band Spectrum of the Short-Time Fourier Transformation
    /// on the current window the normalized magnitudes in the provided output buffer.
    ///
    /// # Panics
    /// Panics when `self.output_size() != output.len()`.
    pub fn compute_ssb_spectrum(&mut self, output: &mut [T]) {
        assert_eq!(output.len(), self.output_size());
        self.fetch_window_and_compute_internal();

        // We either normalize by the window size (if no windowing function
        // is given) or by the sum of the window coefficients.
        let window_size = T::from(self.window_size.get() as f64).unwrap();
        let normalizer = T::from(1.).unwrap() / self.window.sum.unwrap_or(window_size);
        let two = T::from(2).unwrap();

        for (i, (dst, src)) in output
            .iter_mut()
            .zip(self.fft_input_output.iter())
            .enumerate()
        {
            let mut adjusted_value = src.norm() * normalizer.clone();
            // "Add" the mirrored spectrum onto the non-mirrored one; since values
            // are symmetric for real inputs, we can simply multiply by two.
            // We skip the zero-th bin as it contains the DC component.
            if i > 0 {
                adjusted_value = adjusted_value * two.clone();
            }
            *dst = adjusted_value;
        }
    }

    /// Fetches the current data slice, applies windowing and
    /// computes the complex output.
    fn fetch_window_and_compute_internal(&mut self) {
        assert!(self.contains_enough_to_compute());

        // Read from ring buffer into real_input.
        self.samples.read_many_front(&mut self.real_input);

        // Apply the window if it is non-empty.
        if !self.window.is_empty() {
            for (dst, src) in self.real_input.iter_mut().zip(self.window.iter()) {
                *dst = dst.clone() * src.clone();
            }
        }

        // If the window is non-zero in length, all values outside of the window
        // will be set to zero. Otherwise, all values outside of the input window will.
        let zero_after = if self.window.len() > 0 {
            self.window.len()
        } else {
            self.real_input.len()
        };

        // Copy the windowed real_input into the FFT input/output buffer.
        for (src, dst) in self.real_input.iter().zip(self.fft_input_output.iter_mut()) {
            dst.re = src.clone();
            dst.im = T::zero();
        }

        // Ensure the buffer is zero-padded when needed.
        for dst in self.fft_input_output.iter_mut().skip(zero_after) {
            dst.re = T::zero();
            dst.im = T::zero();
        }

        // Compute the FFT on the buffer.
        self.fft
            .process_with_scratch(&mut self.fft_input_output, &mut self.fft_scratch)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::windowing::HannWindow;
    use approx::{assert_relative_eq, assert_ulps_eq};
    use num::traits::FloatConst;
    use rand::prelude::*;

    #[test]
    pub fn output_size_is_half_of_fft_size() {
        let stft = ShortTimeFourierTransform::<f64>::new(
            NonZeroUsize::new(32).unwrap(),
            NonZeroUsize::new(8).unwrap(),
            NonZeroUsize::new(4).unwrap(),
        );

        assert_eq!(stft.output_size(), 16);
    }

    #[test]
    pub fn buffer_length_is_computed_correctly() {
        let mut stft = ShortTimeFourierTransform::<f64>::new(
            NonZeroUsize::new(8).unwrap(),
            NonZeroUsize::new(8).unwrap(),
            NonZeroUsize::new(4).unwrap(),
        );

        assert!(!stft.contains_enough_to_compute());
        assert_eq!(stft.len(), 0);
        assert!(stft.is_empty());

        stft.append_samples(&[500., 0., 100.]);
        assert_eq!(stft.len(), 3);
        assert!(!stft.is_empty());
        assert!(!stft.contains_enough_to_compute());

        stft.append_samples(&[500., 0., 100., 0.]);
        assert_eq!(stft.len(), 7);
        assert!(!stft.contains_enough_to_compute());

        stft.append_samples(&[500.]);
        assert!(stft.contains_enough_to_compute());
    }

    #[test]
    pub fn move_to_next_column_rolls_data_over() {
        let mut stft = ShortTimeFourierTransform::<f64>::new(
            NonZeroUsize::new(8).unwrap(),
            NonZeroUsize::new(8).unwrap(),
            NonZeroUsize::new(4).unwrap(),
        );

        stft.append_samples(&[500., 0., 100., 500., 0., 100.]);
        stft.move_to_next_column();
        assert_eq!(stft.len(), 2);
        assert!(!stft.is_empty());
    }

    #[test]
    pub fn move_to_next_column_may_empty_buffer() {
        let mut stft = ShortTimeFourierTransform::<f64>::new(
            NonZeroUsize::new(8).unwrap(),
            NonZeroUsize::new(8).unwrap(),
            NonZeroUsize::new(4).unwrap(),
        );

        stft.append_samples(&[500., 0., 100., 500., 0., 100., 0., 500.]);
        stft.move_to_next_column();
        assert_eq!(stft.len(), 4);
        assert!(!stft.is_empty());
        assert!(!stft.contains_enough_to_compute());

        stft.move_to_next_column();
        assert_eq!(stft.len(), 0);
        assert!(stft.is_empty());
        assert!(!stft.contains_enough_to_compute());
    }

    #[test]
    pub fn compute_complex_no_windowing() {
        let mut stft = ShortTimeFourierTransform::<f64>::new(
            NonZeroUsize::new(8).unwrap(),
            NonZeroUsize::new(8).unwrap(),
            NonZeroUsize::new(4).unwrap(),
        );

        stft.append_samples(&[500., 0., 100., 500., 0., 100., 0., 500.]);
        assert!(stft.contains_enough_to_compute());

        // Calculate the complex responses.
        let mut output = vec![Complex::zero(); stft.output_size()];
        stft.compute_complex(&mut output);
        let expected = vec![
            Complex::new(1700., 0.),
            Complex::new(429.289322, -29.289322),
            Complex::new(400., 900.),
            Complex::new(570.710678, 170.710678),
            /*
            Complex::new(-500., 0.),
            Complex::new(570.710678, -170.710678),
            Complex::new(400., -900.),
            Complex::new(429.289322, 29.289322),
            */
        ];
        assert_relative_eq!(
            output.as_slice(),
            expected.as_slice(),
            max_relative = 0.00001
        );

        // Repeat the calculation to ensure results are independent of the internal buffer.
        let mut output2 = vec![Complex::zero(); stft.output_size()];
        stft.compute_complex(&mut output2);
        assert_relative_eq!(
            output.as_slice(),
            output2.as_slice(),
            max_relative = 0.00001
        );
    }

    #[test]
    pub fn compute_magnitude_no_windowing() {
        let mut stft = ShortTimeFourierTransform::<f64>::new(
            NonZeroUsize::new(8).unwrap(),
            NonZeroUsize::new(8).unwrap(),
            NonZeroUsize::new(4).unwrap(),
        ); //.with_window(&HannWindow::default());

        stft.append_samples(&[500., 0., 100., 500., 0., 100., 0., 500.]);
        assert!(stft.contains_enough_to_compute());

        let _frequencies = stft.frequencies(8.);

        let mut output: Vec<f64> = vec![0.; stft.output_size()];
        stft.compute_magnitudes(&mut output);
        let expected = vec![
            1700.0,
            430.2873298827358,
            984.8857801796105,
            595.6952356216941,
            /*
            500.0,
            595.6952356216941,
            984.8857801796105,
            430.2873298827358,
            */
        ];
        assert_ulps_eq!(output.as_slice(), expected.as_slice(), max_ulps = 10);

        // Repeat the calculation to ensure results are independent of the internal buffer.
        let mut output2: Vec<f64> = vec![0.; stft.output_size()];
        stft.compute_magnitudes(&mut output2);
        assert_ulps_eq!(output.as_slice(), output2.as_slice(), max_ulps = 10);
    }

    #[test]
    pub fn it_works() {
        // from: https://medium.com/analytics-vidhya/breaking-down-confusions-over-fast-fourier-transform-fft-1561a029b1ab
        let sampling_rate = 1000.;
        let sampling_interval = 1. / sampling_rate;
        let num_time_points = 500usize;
        let time: Vec<f64> = (0..num_time_points)
            .map(|t| t as f64 * sampling_interval)
            .collect();

        // Define the frequencies.
        let f1 = 20.;
        let f2 = 220.;
        let f3 = 138.;

        // Define the amplitudes.
        let dc = 5.;
        let a1 = 12.;
        let a2 = 5.;
        let a3 = 4.;

        // Define the offsets/phases.
        let p1 = 0.3;
        let p2 = 0.5;
        let p3 = 1.5;

        // The noise amplitude.
        let noise_amplitude = 1.0;

        // Helper functions.
        #[inline]
        fn omega(frequency: f64) -> f64 {
            2. * f64::PI() * frequency
        }

        #[inline]
        fn sinusoid(t: f64, frequency: f64, amplitude: f64, phase: f64) -> f64 {
            amplitude * (omega(frequency) * t - phase).sin()
        }

        // Create the signal.
        let signal = time
            .iter()
            .map(|t| {
                dc + sinusoid(*t, f1, a1, p1) + sinusoid(*t, f2, a2, p2) + sinusoid(*t, f3, a3, p3)
            });

        fn gen_noise<R: Rng>(rng: &mut R, amplitude: f64)-> f64 {
            let value: f64 = rng.sample(rand::distributions::Standard);
            value * amplitude
        }

        let mut rng = StdRng::seed_from_u64(0);
        let signal: Vec<_> = signal
            .map(|x| x + gen_noise(&mut rng, noise_amplitude))
            .collect();

        // We can compute the full FFT here by using as many FFT bins as the window is wide, or more.
        // If the exact same number is used, the results are virtually perfect.
        let num_fft_samples = 500;
        let mut stft = ShortTimeFourierTransform::<f64>::new(
            NonZeroUsize::new(num_fft_samples).unwrap(),
            NonZeroUsize::new(num_time_points).unwrap(),
            NonZeroUsize::new(num_time_points).unwrap(),
        ).with_window(&HannWindow::default());

        stft.append_samples(&signal);

        let mut ssb_spectrum = vec![0.; stft.output_size()];
        stft.compute_ssb_spectrum(&mut ssb_spectrum);
        let ssb_spectrum = ssb_spectrum;

        let expected_bin_f1 = (f1 * num_fft_samples as f64 / sampling_rate) as usize;
        let expected_bin_f2 = (f2 * num_fft_samples as f64 / sampling_rate) as usize;
        let expected_bin_f3 = (f3 * num_fft_samples as f64 / sampling_rate) as usize;

        assert_relative_eq!(ssb_spectrum[0], dc, max_relative = 0.1);
        assert_relative_eq!(ssb_spectrum[expected_bin_f1], a1, max_relative = 0.01);
        assert_relative_eq!(ssb_spectrum[expected_bin_f2], a2, max_relative = 0.01);
        assert_relative_eq!(ssb_spectrum[expected_bin_f3], a3, max_relative = 0.01);

        let frequencies = stft.frequencies(sampling_rate);
        assert_relative_eq!(frequencies[0], 0.0, max_relative = 0.0);
        assert_relative_eq!(frequencies[expected_bin_f1], f1, max_relative = 0.0);
        assert_relative_eq!(frequencies[expected_bin_f2], f2, max_relative = 0.0);
        assert_relative_eq!(frequencies[expected_bin_f3], f3, max_relative = 0.0);
    }
}
