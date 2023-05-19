use std::num::NonZeroUsize;
use crate::audiogen::{Component, generate_audio};
use crate::stft::ShortTimeFourierTransform;

mod stft;
mod windowing;
mod audiogen;

fn main() {
    let sample_rate = 44100;
    let signal_duration = 10.0;
    let all_samples = generate_audio(sample_rate, signal_duration, [
        Component::new_dc(5.),
        Component::new(20.0, 12.0, 0.3),
        Component::new(220.0, 5.0, 0.5),
        Component::new(138.0, 4.0, 1.5),
    ]);

    println!("Sample rate: {} Hz", sample_rate);
    println!("Input signal duration: {} s", signal_duration);
    println!("Generated sample vector of length {}", all_samples.len());

    let num_fft_samples = NonZeroUsize::new(512).expect("input is nonzero");
    let window_size = NonZeroUsize::new(1024).expect("input is nonzero");
    let step_size = NonZeroUsize::new(64).expect("input is nonzero");
    let mut stft = ShortTimeFourierTransform::new(num_fft_samples, window_size, step_size);

    let mut spectrogram_column = vec![0.; stft.output_size()];

    // Number of columns: half of the FFT sample count.
    println!("Number of spectrogram columns: {}", spectrogram_column.len());

    // Highest detectable frequency: Half of the sampling rate. (Nyquist theorem.)
    println!("Highest detectable frequency: {} Hz", spectrogram_column.len() as f64 * sample_rate as f64 / (num_fft_samples.get() as f64));

    for some_samples in all_samples[..].chunks(128) {
        stft.append_samples(some_samples);
        while stft.contains_enough_to_compute() {
            stft.compute_ssb_spectrum(&mut spectrogram_column[..]);

            // TODO: Do something with the spectrum.

            stft.move_to_next_column();
        }
    }
}
