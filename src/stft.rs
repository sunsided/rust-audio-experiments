use num::traits::{Float, Signed, Zero};
use num::NumCast;
use rustfft::{Fft, FftDirection, FftNum, FftPlanner};

/// An implementation of the Short-Time (Fast) Fourier Transform.
pub struct ShortTimeFourierTransform {
    /// The size of the time window, in time steps.
    pub window_size: usize,
    /// The number of time steps by which the time window
    /// will be shifted.
    pub step_size: usize,
}

impl ShortTimeFourierTransform {
    pub fn new() -> Self {
        todo!()
    }
}
