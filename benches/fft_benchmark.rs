use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

use ffrust::pfft;
use ffrust::sfft;
use ffrust::utils;

fn bench_fft(mode: u8, samples: &mut Vec<i16>) {
    let mut complex_data = utils::initialize(&samples);

    match mode {
        0u8 => {
            sfft::fft(&mut complex_data);
            sfft::ifft(&mut complex_data);
        }
        1u8 => {
            pfft::rayon_fft(&mut complex_data);
            pfft::rayon_ifft(&mut complex_data);
        }
        2u8 => {
            pfft::simd_fft(&mut complex_data);
            pfft::simd_ifft(&mut complex_data);
        }
        3u8 => {
            pfft::rayon_simd_fft(&mut complex_data);
            pfft::rayon_simd_fft(&mut complex_data);
        }
        _ => {}
    }
}

fn criterion_with_custom_sample_size() -> Criterion {
    Criterion::default().with_output_color(true).sample_size(20)
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut reader = hound::WavReader::open("audio/rr.wav").unwrap();
    let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();

    *c = criterion_with_custom_sample_size();

    c.bench_function("serial fft", |b| {
        b.iter_batched(
            || samples.clone(),
            |mut samples| bench_fft(0, &mut samples),
            BatchSize::SmallInput,
        )
    });
    c.bench_function("parallel fft -- rayon", |b| {
        b.iter_batched(
            || samples.clone(),
            |mut samples| bench_fft(1, &mut samples),
            BatchSize::SmallInput,
        )
    });
    c.bench_function("parallel fft -- SIMD", |b| {
        b.iter_batched(
            || samples.clone(),
            |mut samples| bench_fft(2, &mut samples),
            BatchSize::SmallInput,
        )
    });
    c.bench_function("parallel fft -- rayon & SIMD", |b| {
        b.iter_batched(
            || samples.clone(),
            |mut samples| bench_fft(3, &mut samples),
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
