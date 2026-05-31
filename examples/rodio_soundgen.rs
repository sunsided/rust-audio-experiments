use rodio::source::SineWave;
use rodio::Source;
use std::{thread, time};

fn main() {
    // rodio 0.22: open the default device sink, then queue sources on a Player.
    let stream_handle = rodio::DeviceSinkBuilder::open_default_sink().unwrap();
    let player = rodio::Player::connect_new(stream_handle.mixer());
    player.set_volume(0.1);

    let period = time::Duration::from_millis(1000);

    for i in [932, 587, 523, 466].iter() {
        let source = SineWave::new(*i as _).take_duration(period);
        player.append(source);
    }

    thread::sleep(4 * period);
    player.stop();
}
