use std::fs::File;
use std::io::BufReader;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};

use anyhow::Result;
use colorgrad::{BasisGradient, Gradient};
use pixels::{Pixels, SurfaceTexture};
use rodio::{Decoder, DeviceSinkBuilder, MixerDeviceSink, Player, Source};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::KeyCode;
use winit::window::{Window, WindowId};
use winit_input_helper::WinitInputHelper;

use ruststft::spectrum::magnitude_into;
use ruststft::{Complex, Scaling, Stft, Window as StftWindow};

const WIDTH: u32 = 400;
const HEIGHT: u32 = 300;

fn main() -> Result<()> {
    let paused = Arc::new(AtomicBool::new(false));
    let completed = Arc::new(AtomicBool::new(false));

    const AUDIO_PATH: &str = "audio/waiting-for-a-train.ogg";

    // Decoder used to analyze the signal (consumed by the STFT thread).
    let file = File::open(AUDIO_PATH)?;
    let decoder = Decoder::new(BufReader::new(file))?;

    let sample_rate = decoder.sample_rate().get() as usize;
    let display_max_frequency: f64 = 22050.0 * 0.25; // Hertz
    let display_max_frequency: f64 = display_max_frequency.min(sample_rate as f64 * 0.5);

    // Open the audio output and queue the same track, paused until the visualization starts.
    let stream_handle = DeviceSinkBuilder::open_default_sink()?;
    let player = Player::connect_new(stream_handle.mixer());
    player.set_volume(0.5);
    player.pause();
    let playback_file = File::open(AUDIO_PATH)?;
    player.append(Decoder::new(BufReader::new(playback_file))?);

    // Signal to synchronize the render thread with the STFT thread.
    let wake_up = Arc::new((Mutex::new(false), Condvar::new()));

    let spectrogram_column = calculate_stft_threaded(
        wake_up.clone(),
        completed.clone(),
        paused.clone(),
        sample_rate,
        decoder,
        stream_handle,
        player,
    );
    visualize(
        wake_up,
        paused,
        completed,
        spectrogram_column,
        sample_rate,
        display_max_frequency,
    )
}

/// Calculates the STFT in a background thread.
fn calculate_stft_threaded(
    wake_up: Arc<(Mutex<bool>, Condvar)>,
    completed: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    sample_rate: usize,
    decoder: Decoder<BufReader<File>>,
    stream_handle: MixerDeviceSink,
    player: Player,
) -> Arc<RwLock<Vec<f64>>> {
    // rodio 0.22 decodes to f32 samples already normalized to [-1, 1].
    let all_samples: Vec<f64> = decoder.into_iter().map(|s| s as f64).collect();
    let signal_duration = all_samples.len() as f64 / sample_rate as f64;

    println!("Sample rate: {} Hz", sample_rate);
    println!("Input signal duration: {} s", signal_duration);
    println!("Generated sample vector of length {}", all_samples.len());

    let num_fft_samples = 1024usize;
    let window_size = 512usize;
    let step_size = 64usize;
    let mut stft = Stft::builder()
        .window(StftWindow::<f64>::rectangular(window_size))
        .fft_size(num_fft_samples)
        .hop_size(step_size)
        .scaling(Scaling::Magnitude)
        .build()
        .expect("valid STFT configuration");

    let spectrogram_column = vec![0.; stft.n_freqs()];

    // Number of frequency bins: fft_size / 2 + 1 (DC … Nyquist).
    println!(
        "Number of spectrogram columns: {}",
        spectrogram_column.len()
    );

    // Highest detectable frequency: Half of the sampling rate. (Nyquist theorem)
    println!(
        "Highest detectable frequency: {} Hz",
        spectrogram_column.len() as f64 * sample_rate as f64 / (num_fft_samples as f64)
    );

    let spectrogram_column = Arc::new(RwLock::new(spectrogram_column));
    let spectrum = spectrogram_column.clone();

    let expected_loops_per_second = (sample_rate as f64) / step_size as f64;
    let expected_duration_per_loop = Duration::from_secs_f64(1.0 / expected_loops_per_second);
    println!(
        "Will process one audio frame every {:?}",
        expected_duration_per_loop
    );

    std::thread::spawn(move || {
        // Keep the audio output stream alive for the lifetime of this thread.
        let _audio_stream = stream_handle;

        // Wait for the thread to be signaled.
        {
            let (lock, cvar) = &*wake_up;
            let mut calculate = lock.lock().unwrap();
            while !*calculate {
                calculate = cvar.wait(calculate).unwrap();
            }
        }

        // Start audio playback in lock-step with the real-time-paced STFT loop below.
        player.play();

        let start = Instant::now();
        let mut count = 0;

        // Total time spent paused, excluded from the real-time pacing clock below.
        let mut paused_offset = Duration::ZERO;

        // Reused buffer for the complex spectrum of the current column.
        let mut complex_column = vec![Complex::new(0.0, 0.0); stft.n_freqs()];

        for some_samples in all_samples[..].chunks(1024) {
            stft.append(some_samples);
            while stft.ready() {
                // Honor the visual pause: also pause audio and freeze the pacing clock,
                // so audio and spectrogram resume together instead of fast-forwarding.
                if paused.load(Ordering::Relaxed) {
                    player.pause();
                    let pause_started = Instant::now();
                    while paused.load(Ordering::Relaxed) {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    player.play();
                    paused_offset += Instant::now() - pause_started;
                }

                // Synchronize with the input buffer.
                let runtime = Instant::now() - start - paused_offset;
                let expected_time = expected_duration_per_loop * count;
                if runtime < expected_time {
                    let wait_time = expected_time - runtime;
                    std::thread::sleep(wait_time);
                }

                // Aggregate statistics.
                count += 1;

                // Compute the complex column, then write its magnitudes into the shared buffer.
                stft.process_into(&mut complex_column)
                    .expect("valid column");

                let mut spectrum = spectrum.write().expect("lock acquired");
                magnitude_into(&complex_column, &mut spectrum[..]);
                drop(spectrum);

                stft.step();
            }
        }

        completed.store(true, Ordering::Relaxed);
        let duration = Instant::now() - start;
        println!(
            "STFT calculation finished in {} loops after {:?}",
            count, duration
        );
    });
    spectrogram_column
}

/// Visualizes the STFT, driving the winit 0.30 event loop.
fn visualize(
    wake_up: Arc<(Mutex<bool>, Condvar)>,
    paused: Arc<AtomicBool>,
    completed: Arc<AtomicBool>,
    spectrogram_column: Arc<RwLock<Vec<f64>>>,
    sample_rate: usize,
    display_max_frequency: f64,
) -> Result<()> {
    let event_loop = EventLoop::new()?;
    let mut app = App {
        wake_up,
        paused,
        completed,
        spectrogram_column,
        sample_rate,
        display_max_frequency,
        palette: colorgrad::preset::magma(),
        input: WinitInputHelper::new(),
        window: None,
        pixels: None,
        // The highest value observed in the spectrum.
        max_value: 1.0,
        stft_started: false,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

/// Holds the visualization state across the winit 0.30 [`ApplicationHandler`] callbacks.
struct App {
    wake_up: Arc<(Mutex<bool>, Condvar)>,
    paused: Arc<AtomicBool>,
    completed: Arc<AtomicBool>,
    spectrogram_column: Arc<RwLock<Vec<f64>>>,
    sample_rate: usize,
    display_max_frequency: f64,
    palette: BasisGradient,
    input: WinitInputHelper,
    /// The window, created on `resumed`. `Arc` so `pixels` can hold a `'static` surface.
    window: Option<Arc<Window>>,
    pixels: Option<Pixels<'static>>,
    max_value: f64,
    stft_started: bool,
}

impl App {
    /// Renders one spectrogram frame, scrolling the previous columns to the left.
    fn render(&mut self, event_loop: &ActiveEventLoop) {
        let Some(pixels) = self.pixels.as_mut() else {
            return;
        };

        let is_paused = self.paused.load(Ordering::Relaxed);
        let is_finished = self.completed.load(Ordering::Relaxed);

        if !is_paused && !is_finished {
            let frame = pixels.frame_mut();
            debug_assert_eq!(frame.len(), (WIDTH * HEIGHT * 4) as _);

            // Scroll one pixel column to the left; the newest column is drawn on the right.
            frame.copy_within(4.., 0);

            let spectrum = self.spectrogram_column.read().expect("lock acquired");
            let spectrum_len = spectrum.len() as f64;
            let max_frequency_bin =
                spectrum_len * self.display_max_frequency / (self.sample_rate as f64 * 0.5);

            self.max_value = self
                .max_value
                .max(spectrum.iter().fold(1.0, |max, &v| max.max(v)));

            // Draw the newest column into the rightmost pixel of every row:
            // low frequencies at the bottom, high frequencies at the top.
            let last_col = (WIDTH - 1) as usize;
            for y in 0..HEIGHT as usize {
                let freq_fraction = (HEIGHT as usize - 1 - y) as f64 / HEIGHT as f64;
                let position = ((freq_fraction * max_frequency_bin) as usize)
                    .min(spectrum.len().saturating_sub(1));

                // Increase intensity.
                let value = (spectrum[position] / self.max_value).sqrt();

                let [red, green, blue, _] = self.palette.at(value as f32).to_rgba8();
                let idx = (y * WIDTH as usize + last_col) * 4;
                frame[idx..idx + 4].copy_from_slice(&[red, green, blue, 0xff]);
            }
            drop(spectrum);
        }

        if let Err(err) = pixels.render() {
            eprintln!("pixels.render: {:?}", err);
            event_loop.exit();
        }
    }
}

impl ApplicationHandler for App {
    fn new_events(&mut self, _event_loop: &ActiveEventLoop, _cause: winit::event::StartCause) {
        self.input.step();
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        let attributes = Window::default_attributes()
            .with_title("STFT Visualization")
            .with_inner_size(size)
            .with_min_inner_size(size);
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .expect("window created"),
        );

        let window_size = window.inner_size();
        let surface_texture =
            SurfaceTexture::new(window_size.width, window_size.height, window.clone());
        let pixels = Pixels::new(WIDTH, HEIGHT, surface_texture).expect("pixels created");

        self.window = Some(window);
        self.pixels = Some(pixels);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        self.input.process_window_event(&event);

        match event {
            WindowEvent::RedrawRequested => self.render(event_loop),
            WindowEvent::Resized(size) => {
                if let Some(pixels) = self.pixels.as_mut() {
                    if let Err(err) = pixels.resize_surface(size.width, size.height) {
                        eprintln!("pixels.resize_surface: {:?}", err);
                        event_loop.exit();
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.input.end_step();

        if self.input.key_pressed(KeyCode::Escape)
            || self.input.close_requested()
            || self.input.destroyed()
        {
            event_loop.exit();
            return;
        }

        if self.input.key_pressed(KeyCode::Space) {
            self.paused
                .store(!self.paused.load(Ordering::Relaxed), Ordering::Relaxed);
        }

        // Signal STFT thread to start.
        if !self.stft_started {
            self.stft_started = true;
            let (lock, cvar) = &*self.wake_up;
            let mut calculate = lock.lock().unwrap();
            *calculate = true;
            cvar.notify_one();
        }

        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}
