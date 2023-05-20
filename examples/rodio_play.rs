use rodio::{OutputStream, Sink};
use std::io::BufReader;

fn main() {
    let (_stream, stream_handle) = OutputStream::try_default().unwrap();

    let sink = Sink::try_new(&stream_handle).unwrap();
    sink.set_volume(0.1);

    let file = std::fs::File::open("audio/woodpecker.ogg").unwrap();
    sink.append(rodio::Decoder::new(BufReader::new(file)).unwrap());

    sink.sleep_until_end();
}
