use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};
use crate::audiogen::{Component, generate_audio};
use crate::stft::ShortTimeFourierTransform;
use pixels::{Error, Pixels, SurfaceTexture};
use winit::{
    dpi::LogicalSize,
    event::{Event, VirtualKeyCode},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use winit_input_helper::WinitInputHelper;

mod stft;
mod windowing;
mod audiogen;

const WIDTH: u32 = 400;
const HEIGHT: u32 = 300;

fn main() -> Result<(), Error> {
    let paused = Arc::new(AtomicBool::new(false));
    let completed = Arc::new(AtomicBool::new(false));

    let sample_rate = 44100;
    let signal_duration = 10.0;

    // Signal to synchronize the render thread with the STFT thread.
    let wake_up = Arc::new((Mutex::new(false), Condvar::new()));

    let spectrogram_column = calculate_stft_threaded(wake_up.clone(), completed.clone(), sample_rate, signal_duration);
    visualize(wake_up, paused, completed, spectrogram_column)
}

/// Calculates the STFT in a background thread.
fn calculate_stft_threaded(wake_up: Arc<(Mutex<bool>, Condvar)>, completed: Arc<AtomicBool>, sample_rate: usize, signal_duration: f64) -> Arc<RwLock<Vec<f64>>> {
    let all_samples = generate_audio(sample_rate, signal_duration, [
        Component::new_dc(5.),
        Component::new(20.0, 12.0, 0.3),
        Component::new(220.0, 5.0, 0.5),
        Component::new(12000.0, 4.0, 1.5),
    ]);

    println!("Sample rate: {} Hz", sample_rate);
    println!("Input signal duration: {} s", signal_duration);
    println!("Generated sample vector of length {}", all_samples.len());

    let num_fft_samples = NonZeroUsize::new(WIDTH as usize * 2).expect("input is nonzero");
    let window_size = NonZeroUsize::new(1024).expect("input is nonzero");
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
        let duration  = Instant::now() - start;
        println!("STFT calculation finished in {} loops after {:?}", count, duration);
    });
    spectrogram_column
}

/// Visualizes the STFT in a background thread.
fn visualize(wake_up: Arc<(Mutex<bool>, Condvar)>, paused: Arc<AtomicBool>, completed: Arc<AtomicBool>, spectrogram_column: Arc<RwLock<Vec<f64>>>) -> Result<(), Error> {
    let event_loop = EventLoop::new();
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

    event_loop.run(move |event, _, control_flow| {
        if let Event::RedrawRequested(_) = event {
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
                for (i, pixel) in frame.chunks_exact_mut(4).skip((HEIGHT - 1) as usize * WIDTH as usize).enumerate() {
                    let position = i * spectrum.len() / WIDTH as usize;
                    debug_assert_eq!(position, i);

                    max_value = max_value.max(spectrum.iter().fold(1.0, |max, &v| max.max(v)));
                    let value = spectrum[position] / max_value;

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
                *control_flow = ControlFlow::Exit;
                return;
            }
        }

        // For everything else, for let winit_input_helper collect events to build its state.
        // It returns `true` when it is time to update our game state and request a redraw.
        if input.update(&event) {
            if input.key_pressed(VirtualKeyCode::Escape) || input.close_requested() || input.destroyed() {
                *control_flow = ControlFlow::Exit;
                return;
            }

            if input.key_pressed(VirtualKeyCode::Space) {
                paused.store(!paused.load(Ordering::Relaxed), Ordering::Relaxed);
            }

            // Resize the window
            if let Some(size) = input.window_resized() {
                if let Err(err) = pixels.resize_surface(size.width, size.height) {
                    eprintln!("pixels.resize_surface: {:?}", err);
                    *control_flow = ControlFlow::Exit;
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
    });
}
