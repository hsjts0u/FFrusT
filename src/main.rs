mod pfft;
mod rfft;
mod sfft;
mod utils;

use std::time::Instant;

fn main() {
    let mut reader = hound::WavReader::open("audio/rr.wav").unwrap();
    let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();

    let mut complex_data = utils::initialize(&samples);
    let now = Instant::now();
    sfft::fft(&mut complex_data);
    sfft::ifft(&mut complex_data);
    let elapsed = now.elapsed();
    println!("Serial Elapsed: {:.2?}", elapsed);

    let mut complex_data = utils::initialize(&samples);
    let now = Instant::now();
    rfft::fft(&mut complex_data);
    rfft::ifft(&mut complex_data);
    let elapsed = now.elapsed();
    println!("Recursive Elapsed: {:.2?}", elapsed);

    let mut complex_data = utils::initialize(&samples);
    let now = Instant::now();
    pfft::rayon_fft(&mut complex_data);
    pfft::rayon_ifft(&mut complex_data);
    let elapsed = now.elapsed();
    println!("Rayon Elapsed: {:.2?}", elapsed);

    let mut complex_data = utils::initialize(&samples);
    let now = Instant::now();
    pfft::simd_fft(&mut complex_data);
    pfft::simd_ifft(&mut complex_data);
    let elapsed = now.elapsed();
    println!("SIMD Elapsed: {:.2?}", elapsed);

    let mut complex_data = utils::initialize(&samples);
    let now = Instant::now();
    pfft::rayon_simd_fft(&mut complex_data);
    pfft::rayon_simd_ifft(&mut complex_data);
    let elapsed = now.elapsed();
    println!("Rayon SIMD Elapsed: {:.2?}", elapsed);
}
