use ffrust::freq_scale;
use ffrust::intrinfft;
use ffrust::pfft;
use ffrust::rfft;
use ffrust::sfft;
use ffrust::splitfft;
use ffrust::utils;
use num_complex::Complex;

fn run_fft_ifft(fft: fn(&mut Vec<Complex<f64>>), ifft: fn(&mut Vec<Complex<f64>>), filename: &str) {
    let mut reader = hound::WavReader::open("audio/sine.wav").unwrap();
    let spec = reader.spec();
    let mut writer = hound::WavWriter::create(filename, spec).unwrap();

    let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();
    let mut data = utils::initialize(&samples);
    let mut scale_data = data.clone();

    fft(&mut data);
    ifft(&mut data);

    fft(&mut scale_data);
    freq_scale::scaledown(&mut scale_data, spec.sample_rate, 441);
    ifft(&mut scale_data);

    let new_samples: Vec<i16> = data.iter().map(|c| c.re as i16).collect();
    let down_samples: Vec<i16> = scale_data.iter().map(|c| c.re as i16).collect();

    let tolerance = -1..2;

    for i in 0..samples.len() {
        writer.write_sample(down_samples[i]).unwrap();
    }

    for i in 0..samples.len() {
        let diff = samples[i] - new_samples[i];
        assert!(tolerance.contains(&diff));
    }
}

#[test]
fn test_sfft() {
    run_fft_ifft(sfft::fft, sfft::ifft, "serial_recon.wav");
}

#[test]
fn test_rfft() {
    run_fft_ifft(rfft::fft, rfft::ifft, "recursive_recon.wav");
}

#[test]
fn test_rayon_fft() {
    run_fft_ifft(pfft::rayon_fft, pfft::rayon_ifft, "rayon_recon.wav");
}

#[test]
fn test_simd_fft() {
    run_fft_ifft(intrinfft::simd_fft, intrinfft::simd_ifft, "simd_recon.wav");
}

#[test]
fn test_rayon_simd_fft() {
    run_fft_ifft(
        intrinfft::rayon_simd_fft,
        intrinfft::rayon_simd_ifft,
        "rayon_simd_recon.wav",
    );
}

#[test]
fn test_split_fft() {
    run_fft_ifft(splitfft::fft, splitfft::ifft, "split_recon.wav");
}
