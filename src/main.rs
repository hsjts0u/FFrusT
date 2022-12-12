mod intrinfft;
mod pfft;
mod rfft;
mod sfft;
mod utils;

use rustfft::FftPlanner;
use std::time::Instant;
use std::env;

// macro_rules! bench_fft {
//     ($fft: stmt, $ifft: stmt, $value: expr, $ver: expr) => {
//         print!("{} Elapsed Time: ", $ver);

//         for _ in 0..$value {
//             let now = Instant::now();
//             $fft
//             $ifft
//             let elapsed = now.elapsed();
//             print!("{:.2?} ", elapsed);
//         }
//         println!();
//     };
// }

fn main() {
    let args: Vec<String> = env::args().collect();
    let output_filename = if args.len() < 1 { &args[1] } else { "result.csv" };

    let mut reader = hound::WavReader::open("audio/rr.wav").unwrap();
    let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();
    let bench_niter = 2;

    let mut complex_data = utils::initialize(&samples);
    let mut result = Vec::new();
    print!("Serial Elapsed Time: ");
    for _ in 0..bench_niter {
        let now = Instant::now();
        sfft::fft(&mut complex_data);
        sfft::ifft(&mut complex_data);
        let elapsed = now.elapsed();
        print!("{:.2?} ", elapsed);
        result.push(elapsed);
    }
    println!();
    utils::dump_result("serial", &mut result, output_filename).unwrap();

    let mut complex_data = utils::initialize(&samples);
    let mut result = Vec::new();
    print!("Recursive Elapsed Time: ");
    for _ in 0..bench_niter {
        let now = Instant::now();
        rfft::fft(&mut complex_data);
        rfft::ifft(&mut complex_data);
        let elapsed = now.elapsed();
        print!("{:.2?} ", elapsed);
        result.push(elapsed);
    }
    println!();
    utils::dump_result("recursive", &mut result, output_filename).unwrap();

    let mut complex_data = utils::initialize(&samples);
    let mut result = Vec::new();
    print!("Rayon Elapsed Time: ");
    for _ in 0..bench_niter {
        let now = Instant::now();
        pfft::rayon_fft(&mut complex_data);
        pfft::rayon_ifft(&mut complex_data);
        let elapsed = now.elapsed();
        print!("{:.2?} ", elapsed);
        result.push(elapsed);
    }
    println!();
    utils::dump_result("rayon", &mut result, output_filename).unwrap();

    let mut complex_data = utils::initialize(&samples);
    let mut result = Vec::new();
    print!("SIMD Elapsed Time: ");
    for _ in 0..bench_niter {
        let now = Instant::now();
        intrinfft::simd_fft(&mut complex_data);
        intrinfft::simd_ifft(&mut complex_data);
        let elapsed = now.elapsed();
        print!("{:.2?} ", elapsed);
        result.push(elapsed);
    }
    println!();
    utils::dump_result("simd", &mut result, output_filename).unwrap();

    let mut complex_data = utils::initialize(&samples);
    let mut result = Vec::new();
    print!("Rayon SIMD Elapsed Time: ");
    for _ in 0..bench_niter {
        let now = Instant::now();
        intrinfft::rayon_simd_fft(&mut complex_data);
        intrinfft::rayon_simd_ifft(&mut complex_data);
        let elapsed = now.elapsed();
        print!("{:.2?} ", elapsed);
        result.push(elapsed);
    }
    println!();
    utils::dump_result("rayon simd", &mut result, output_filename).unwrap();

    let mut planner = FftPlanner::<f64>::new();
    let mut complex_data = utils::initialize(&samples);
    let len = complex_data.len();
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
