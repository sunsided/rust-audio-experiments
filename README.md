# 🔉 Audio & Short-Time Fourier Transforms

This project experiments with calculation and visualizations of STFTs in Rust.
Here is a spectral visualization of the beginning of the song _[Waiting for a Train](audio/waiting-for-a-train.ogg)_:

<p align="center" style="text-align: center">
    <img src="images/stft.png" alt="Spectrum of a Short-Term Fourier Transform" />
</p>

To run the application, execute:

```
cargo run --release
```

---

## Bucket list

- [x] Implement an example with multiple overlaid frequencies.
- [x] Implement simple audio generation example.
- [x] Implement different windowing functions.
- [x] ~~Implement~~ Add audio file o a frequency sweep to better visualize the STFT.
- [ ] Map microphone input to the display.
