use stft::{WindowType, STFT};

fn main() {
    // let's generate ten seconds of fake audio
    let sample_rate: usize = 44100;
    let seconds: usize = 10;
    let sample_count = sample_rate * seconds;
    let all_samples = (0..sample_count).map(|x| x as f64).collect::<Vec<f64>>();

    // let's initialize our short-time fourier transform
    let window_type: WindowType = WindowType::Hanning;
    let window_size: usize = 1024;
    let step_size: usize = 512;
    let mut stft = STFT::<f64>::new(window_type, window_size, step_size);

    // we need a buffer to hold a computed column of the spectrogram
    let mut spectrogram_column: Vec<f64> = std::iter::repeat(0.).take(stft.output_size()).collect();

    // iterate over all the samples in chunks of 3000 samples.
    // in a real program you would probably read from something instead.
    for some_samples in (&all_samples[..]).chunks(3000) {
        // append the samples to the internal ringbuffer of the stft
        stft.append_samples(some_samples);

        // as long as there remain window_size samples in the internal
        // ringbuffer of the stft
        while stft.contains_enough_to_compute() {
            // compute one column of the stft by
            // taking the first window_size samples of the internal ringbuffer,
            // multiplying them with the window,
            // computing the fast fourier transform,
            // taking half of the symetric complex outputs,
            // computing the norm of the complex outputs and
            // taking the log10
            stft.compute_column(&mut spectrogram_column[..]);

            // here's where you would do something with the
            // spectrogram_column...

            // drop step_size samples from the internal ringbuffer of the stft
            // making a step of size step_size
            stft.move_to_next_column();
        }
    }
}
