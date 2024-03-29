use std::fs::File;
use std::io::BufReader;
use std::num::NonZeroUsize;
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use pixels::{Pixels, SurfaceTexture};
use rodio::{Decoder, Source};
use winit::{
    dpi::LogicalSize,
    event::Event,
    event_loop::EventLoop,
    window::WindowBuilder,
};
use winit::event::WindowEvent;
use winit::keyboard::KeyCode;
use winit_input_helper::WinitInputHelper;

use crate::stft::ShortTimeFourierTransform;

mod stft;
mod windowing;
mod audiogen;

const WIDTH: u32 = 400;
const HEIGHT: u32 = 300;

fn main() -> Result<()> {
    let paused = Arc::new(AtomicBool::new(false));
    let completed = Arc::new(AtomicBool::new(false));

    let file = File::open("audio/waiting-for-a-train.ogg")?;
    let decoder = Decoder::new(BufReader::new(file))?;

    let sample_rate = decoder.sample_rate() as usize;
    let display_max_frequency: f64 = 22050.0 * 0.25; // Hertz
    let display_max_frequency: f64 = display_max_frequency.min(sample_rate as f64 * 0.5);

    // Signal to synchronize the render thread with the STFT thread.
    let wake_up = Arc::new((Mutex::new(false), Condvar::new()));

    let spectrogram_column = calculate_stft_threaded(wake_up.clone(), completed.clone(), sample_rate, decoder);
    visualize(wake_up, paused, completed, spectrogram_column, sample_rate, display_max_frequency)
}

/// Calculates the STFT in a background thread.
fn calculate_stft_threaded(wake_up: Arc<(Mutex<bool>, Condvar)>, completed: Arc<AtomicBool>, sample_rate: usize, decoder: Decoder<BufReader<File>>) -> Arc<RwLock<Vec<f64>>> {
    let all_samples: Vec<f64> = decoder.into_iter().map(|i| i as f64 / (i16::MAX as f64)).collect();
    let signal_duration = all_samples.len() as f64 / sample_rate as f64;

    println!("Sample rate: {} Hz", sample_rate);
    println!("Input signal duration: {} s", signal_duration);
    println!("Generated sample vector of length {}", all_samples.len());

    let num_fft_samples = NonZeroUsize::new(1024).expect("input is nonzero");
    let window_size = NonZeroUsize::new(512).expect("input is nonzero");
    let step_size = NonZeroUsize::new(64).expect("input is nonzero");
    let mut stft = ShortTimeFourierTransform::new(num_fft_samples, window_size, step_size);

    let spectrogram_column = vec![0.; stft.output_size()];

    // Number of columns: half of the FFT sample count.
    println!("Number of spectrogram columns: {}", spectrogram_column.len());

    // Highest detectable frequency: Half of the sampling rate. (Nyquist theorem)
    println!("Highest detectable frequency: {} Hz", spectrogram_column.len() as f64 * sample_rate as f64 / (num_fft_samples.get() as f64));

    let spectrogram_column = Arc::new(RwLock::new(spectrogram_column));
    let spectrum = spectrogram_column.clone();

    let expected_loops_per_second = (sample_rate as f64) / step_size.get() as f64;
    let expected_duration_per_loop = Duration::from_secs_f64(1.0 / expected_loops_per_second);
    println!("Will process one audio frame every {:?}", expected_duration_per_loop);

    std::thread::spawn(move || {
        // Wait for the thread to be signaled.
        {
            let (lock, cvar) = &*wake_up;
            let mut calculate = lock.lock().unwrap();
            while !*calculate {
                calculate = cvar.wait(calculate).unwrap();
            }
        }

        let start = Instant::now();
        let mut count = 0;

        for some_samples in all_samples[..].chunks(1024) {
            stft.append_samples(some_samples);
            while stft.contains_enough_to_compute() {
                // Synchronize with the input buffer.
                let runtime = Instant::now() - start;
                let expected_time = expected_duration_per_loop * count;
                if runtime < expected_time {
                    let wait_time = expected_time - runtime;
                    std::thread::sleep(wait_time);
                }

                // Aggregate statistics.
                count += 1;

                let mut spectrum = spectrum.write().expect("lock acquired");
                stft.compute_ssb_spectrum(&mut spectrum[..]);
                drop(spectrum);

                stft.move_to_next_column();
            }
        }

        completed.store(true, Ordering::Relaxed);
        let duration = Instant::now() - start;
        println!("STFT calculation finished in {} loops after {:?}", count, duration);
    });
    spectrogram_column
}

/// Visualizes the STFT in a background thread.
fn visualize(wake_up: Arc<(Mutex<bool>, Condvar)>, paused: Arc<AtomicBool>, completed: Arc<AtomicBool>, spectrogram_column: Arc<RwLock<Vec<f64>>>, sample_rate: usize, display_max_frequency: f64) -> Result<()> {
    let event_loop = EventLoop::new()?;
    let mut input = WinitInputHelper::new();

    let window = {
        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        let scaled_size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        WindowBuilder::new()
            .with_title("STFT Visualization")
            .with_inner_size(scaled_size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(WIDTH, HEIGHT, surface_texture)?
    };

    let palette = colorgrad::magma();

    // The highest value observed in the spectrum.
    let mut max_value = 1.0f64;
    let mut stft_started = false;

    event_loop.run(move |event, target| {
        if let Event::WindowEvent { event: ref we, .. } = event {
            if let WindowEvent::RedrawRequested = we {
                let is_paused = paused.load(Ordering::Relaxed);
                let is_finished = completed.load(Ordering::Relaxed);

                if !is_paused && !is_finished {
                    let frame = pixels.frame_mut();
                    debug_assert_eq!(frame.len(), (WIDTH * HEIGHT * 4) as _);

                    // Move all pixels one row up.
                    let start: usize = WIDTH as usize * 4;
                    let end: usize = HEIGHT as usize * WIDTH as usize * 4;
                    frame.copy_within(start..end, 0);

                    let spectrum = spectrogram_column.read().expect("lock acquired");
                    let spectrum_len = spectrum.len() as f64;
                    let max_frequency_bin = spectrum_len * display_max_frequency / (sample_rate as f64 * 0.5);

                    for (i, pixel) in frame.chunks_exact_mut(4).skip((HEIGHT - 1) as usize * WIDTH as usize).enumerate() {

                        // TODO: Map the input range to the target range (0..display_max_frequency).
                        let position = (i as f64 / WIDTH as f64) * max_frequency_bin;
                        let position = position as usize;

                        max_value = max_value.max(spectrum.iter().fold(1.0, |max, &v| max.max(v)));
                        let value = spectrum[position] / max_value;

                        // Increase intensity.
                        let value = value.sqrt();

                        let color = palette.at(value);

                        let red = ((color.r * 255.0) as u8).max(0).min(255);
                        let green = ((color.g * 255.0) as u8).max(0).min(255);
                        let blue = ((color.b * 255.0) as u8).max(0).min(255);

                        pixel.copy_from_slice(&[red, green, blue, 0xff]);
                    }
                    drop(spectrum);
                }

                if let Err(err) = pixels.render() {
                    eprintln!("pixels.render: {:?}", err);
                    target.exit();
                    return;
                }
            }
        }

        // For everything else, for let winit_input_helper collect events to build its state.
        // It returns `true` when it is time to update our game state and request a redraw.
        if input.update(&event) {
            if input.key_pressed(KeyCode::Escape) || input.close_requested() || input.destroyed() {
                target.exit();
                return;
            }

            if input.key_pressed(KeyCode::Space) {
                paused.store(!paused.load(Ordering::Relaxed), Ordering::Relaxed);
            }

            // Resize the window
            if let Some(size) = input.window_resized() {
                if let Err(err) = pixels.resize_surface(size.width, size.height) {
                    eprintln!("pixels.resize_surface: {:?}", err);
                    target.exit();
                    return;
                }
            }

            // Signal STFT thread to start.
            if !stft_started {
                stft_started = true;
                let (lock, cvar) = &*wake_up;
                let mut calculate = lock.lock().unwrap();
                *calculate = true;
                cvar.notify_one();
            }

            window.request_redraw();
        }
    }).unwrap();

    Ok(())
}
