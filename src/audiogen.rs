/// Generates "audio" from the specified frequency components.
///
/// ## Arguments
/// * `sample_rate` - The sampling rate in Hertz, e.g. `44100`.
/// * `duration` - The duration of the signal in seconds, e.g. `10.0`.
/// * `iter` - The audio components to generate, see [`Component`].
///
/// ## Returns
/// The audio vector generated.
pub fn generate_audio<C: IntoIterator<Item = Component>>(sample_rate: usize, duration: f64, iter: C) -> Vec<f64> {
    let sampling_interval = 1. / (sample_rate as f64);
    let sample_count = ((sample_rate as f64) * duration) as usize;

    let components: Vec<_> = iter.into_iter().collect();

    let time: Vec<f64> = (0..sample_count)
        .map(|t| t as f64 * sampling_interval)
        .collect();

    time
        .iter()
        .map(|&t| {
            components.iter().map(|c| c.sinusoid(t)).sum()
        }).collect()
}

#[inline]
fn omega(frequency: f64) -> f64 {
    2. * std::f64::consts::PI * frequency
}

#[inline]
pub(crate) fn sinusoid(t: f64, frequency: f64, amplitude: f64, phase: f64) -> f64 {
    // Using cos() instead of sin() so the DC calculation works, since cos(0.) = 1.
    amplitude * (omega(frequency) * t - phase).cos()
}

/// A frequency component.
pub struct Component {
    /// The frequency.
    frequency: f64,
    /// The amplitude.
    amplitude: f64,
    /// The phase.
    phase: f64
}

impl Component {
    pub const fn new_dc(amplitude: f64) -> Self {
        Self { frequency: 0.0, amplitude, phase: 0.0 }
    }

    pub const fn new(frequency: f64, amplitude: f64, phase: f64) -> Self {
        Self { frequency, amplitude, phase }
    }

    fn sinusoid(&self, t: f64) -> f64 {
        if self.frequency > 0.0 {
            sinusoid(t, self.frequency, self.amplitude, self.phase)
        }
        else {
            self.amplitude
        }
    }
}
