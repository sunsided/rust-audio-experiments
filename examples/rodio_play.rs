use std::io::BufReader;

fn main() {
    // rodio 0.22: open the default device sink, then drive it through a Player.
    let stream_handle = rodio::DeviceSinkBuilder::open_default_sink().unwrap();
    let player = rodio::Player::connect_new(stream_handle.mixer());
    player.set_volume(0.1);

    let file = std::fs::File::open("audio/woodpecker.ogg").unwrap();
    player.append(rodio::Decoder::new(BufReader::new(file)).unwrap());

    player.sleep_until_end();
}
