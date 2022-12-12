use ffrust::intrinfft;
use ffrust::pfft;
use ffrust::rfft;
use ffrust::sfft;
use ffrust::utils;
use num_complex::Complex;

fn run_fft_ifft(fft: fn(&mut Vec<Complex<f64>>), ifft: fn(&mut Vec<Complex<f64>>)) {
    let mut reader = hound::WavReader::open("audio/sine.wav").unwrap();
    let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();
    let mut data = utils::initialize(&samples);

    fft(&mut data);
    ifft(&mut data);

    let new_samples: Vec<i16> = data.iter().map(|c| c.re as i16).collect();

    let tolerance = -1..2;

    for i in 0..samples.len() {
        let diff = samples[i] - new_samples[i];
        assert!(tolerance.contains(&diff));
    }
}

#[test]
fn test_sfft() {
    run_fft_ifft(sfft::fft, sfft::ifft);
}

#[test]
fn test_rfft() {
    run_fft_ifft(rfft::fft, rfft::ifft);
}

#[test]
fn test_rayon_fft() {
    run_fft_ifft(pfft::rayon_fft, pfft::rayon_ifft);
}

#[test]
fn test_simd_fft() {
    run_fft_ifft(intrinfft::simd_fft, intrinfft::simd_ifft);
}

#[test]
fn test_rayon_simd_fft() {
    run_fft_ifft(intrinfft::rayon_simd_fft, intrinfft::rayon_simd_ifft);
}
