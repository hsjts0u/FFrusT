mod intrinfft;
mod pfft;
mod rfft;
mod sfft;
mod splitfft;
mod utils;

use num_complex::Complex;
use rustfft::FftPlanner;
use std::env;
use std::time::Duration;
use std::time::Instant;

fn bench_fft(
    fft: fn(&mut Vec<Complex<f64>>),
    ifft: fn(&mut Vec<Complex<f64>>),
    bench: &str,
    version: &str,
    niter: u64,
) -> Vec<Duration> {
    let mut reader = hound::WavReader::open(bench).unwrap();
    let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();
    let mut complex_data = utils::initialize(&samples);

    let mut result = Vec::new();

    print!("{} Elapsed Time: ", version);
    for _ in 0..niter {
        let now = Instant::now();
        fft(&mut complex_data);
        ifft(&mut complex_data);
        let elapsed = now.elapsed();
        print!("{:.2?} ", elapsed);
        result.push(elapsed);
    }
    println!();
    return result;
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let output_filename = if args.len() > 1 {
        &args[1]
    } else {
        "result.csv"
    };

    let bench_niter = 2;
    let benchname = "audio/rr.wav";

    let mut result = bench_fft(sfft::fft, sfft::ifft, benchname, "Serial", bench_niter);
    utils::dump_result("serial", &mut result, output_filename).unwrap();

    let mut result = bench_fft(rfft::fft, rfft::ifft, benchname, "Recursive", bench_niter);
    utils::dump_result("recursive", &mut result, output_filename).unwrap();

    let mut result = bench_fft(
        pfft::rayon_fft,
        pfft::rayon_ifft,
        benchname,
        "Rayon",
        bench_niter,
    );
    utils::dump_result("rayon", &mut result, output_filename).unwrap();

    let mut result = bench_fft(
        intrinfft::simd_fft,
        intrinfft::simd_ifft,
        benchname,
        "SIMD",
        bench_niter,
    );
    utils::dump_result("simd", &mut result, output_filename).unwrap();

    let mut result = bench_fft(
        intrinfft::rayon_simd_fft,
        intrinfft::rayon_simd_ifft,
        benchname,
        "Rayon SIMD",
        bench_niter,
    );
    utils::dump_result("rayon simd", &mut result, output_filename).unwrap();

    let mut result = bench_fft(
        splitfft::fft,
        splitfft::ifft,
        benchname,
        "Split FFT",
        bench_niter,
    );
    utils::dump_result("split fft", &mut result, output_filename).unwrap();

    let mut reader = hound::WavReader::open(benchname).unwrap();
    let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();
    let mut complex_data = utils::initialize(&samples);
    let len = complex_data.len();

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(len);
    let ifft = planner.plan_fft_inverse(len);

    let mut result = Vec::new();

    print!("RustFFT Elapsed Time: ");
    for _ in 0..bench_niter {
        let now = Instant::now();
        fft.process(&mut complex_data);
        ifft.process(&mut complex_data);
        let elapsed = now.elapsed();
        print!("{:.2?} ", elapsed);
        result.push(elapsed);
    }
    println!();
    utils::dump_result("rustfft", &mut result, output_filename).unwrap();
}
