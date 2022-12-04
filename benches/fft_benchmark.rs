use criterion::{criterion_group, criterion_main, Criterion};

use ffrust::sfft;
use ffrust::pfft;
use ffrust::utils;

fn test_fft(mode: u8) {
    let mut reader = hound::WavReader::open("audio/sine.wav").unwrap();
    let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();

    let mut complex_data = utils::initialize(&samples);
    
    sfft::fft(&mut complex_data);
    sfft::ifft(&mut complex_data);

    match mode {
        0u8 => {
            sfft::fft(&mut complex_data);
            sfft::ifft(&mut complex_data);
        },
        1u8 => {
            pfft::rayon_fft(&mut complex_data);
            pfft::rayon_ifft(&mut complex_data);
        },
        2u8 => {
            pfft::simd_fft(&mut complex_data);
            pfft::simd_ifft(&mut complex_data);
        },
        3u8 => {
            pfft::rayon_simd_fft(&mut complex_data);
            pfft::rayon_simd_fft(&mut complex_data);
        },
        _ => { }
    }

    let new_samples: Vec<i16> = complex_data.iter().map(|c| c.re as i16).collect();

    let tolerance = -1..2;

    for i in 0..samples.len() {
        let diff = samples[i] - new_samples[i];
        assert!(tolerance.contains(&diff));
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("serial fft", |b| b.iter(|| test_fft(0)));
    c.bench_function("parallel fft -- rayon", |b| b.iter(|| test_fft(1)));
    c.bench_function("parallel fft -- SIMD", |b| b.iter(|| test_fft(2)));
    c.bench_function("parallel fft -- rayon & SIMD", |b| b.iter(|| test_fft(3)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

