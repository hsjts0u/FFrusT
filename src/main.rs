mod pfft;
mod rfft;
mod sfft;
mod utils;

use std::time::Instant;

macro_rules! bench_fft {
    ($fft: stmt, $ifft: stmt, $value: expr, $ver: expr) => {
        print!("{} Elapsed Time: ", $ver);

        for _ in 0..$value {
            let now = Instant::now();
            $fft
            $ifft
            let elapsed = now.elapsed();
            print!("{:.2?} ", elapsed);
        }
        println!();
    };
}

fn main() {
    let mut reader = hound::WavReader::open("audio/rr.wav").unwrap();
    let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();
    let bench_niter = 1;

    let mut complex_data = utils::initialize(&samples);
    bench_fft!(
        sfft::fft(&mut complex_data),
        sfft::ifft(&mut complex_data),
        bench_niter,
        "Serial"
    );

    let mut complex_data = utils::initialize(&samples);
    bench_fft!(
        rfft::fft(&mut complex_data),
        rfft::ifft(&mut complex_data),
        bench_niter,
        "Recursive"
    );

    let mut complex_data = utils::initialize(&samples);
    bench_fft!(
        pfft::rayon_fft(&mut complex_data),
        pfft::rayon_ifft(&mut complex_data),
        bench_niter,
        "Rayon"
    );

    let mut complex_data = utils::initialize(&samples);
    bench_fft!(
        pfft::simd_fft(&mut complex_data),
        pfft::simd_ifft(&mut complex_data),
        bench_niter,
        "SIMD"
    );

    let mut complex_data = utils::initialize(&samples);
    bench_fft!(
        pfft::rayon_simd_fft(&mut complex_data),
        pfft::rayon_simd_ifft(&mut complex_data),
        bench_niter,
        "Rayon SIMD"
    );
}
