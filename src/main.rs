use rodio::{OutputStream, Sink, Source};
use rodio::source::SineWave;
use std::{thread, time};

fn main() {
    let (_stream, stream_handle) = OutputStream::try_default().unwrap();

    let period = time::Duration::from_millis(1000);

    let sink = Sink::try_new(&stream_handle).unwrap();
    sink.set_volume(0.1);

    for i in [932, 587, 523, 466].iter() {
        let source = SineWave::new(*i)
            .take_duration(period);
        sink.append(source);
    }

    thread::sleep(4*period);
    sink.stop();
}

