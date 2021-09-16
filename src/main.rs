use rodio::{OutputStream, Sink};
use rodio::source::SineWave;
use std::{thread, time};

fn main() {
    let (_stream, stream_handle) = OutputStream::try_default().unwrap();
    let period = time::Duration::from_millis(1000);
    for i in [932, 587, 523, 466].iter() {
        let sink = Sink::try_new(&stream_handle).unwrap();
        sink.set_volume(0.1);
        sink.append(SineWave::new(*i));
        thread::sleep(period);
        sink.stop();
    }
}

