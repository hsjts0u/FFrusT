pub mod pfft;
pub mod sfft;
pub mod utils;

pub fn test_fft(mode: u8) {
    let mut reader = hound::WavReader::open("audio/sine.wav").unwrap();
    let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();

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

    let new_samples: Vec<i16> = complex_data.iter().map(|c| c.re as i16).collect();

    let tolerance = -1..2;

    for i in 0..samples.len() {
        let diff = samples[i] - new_samples[i];
        assert!(tolerance.contains(&diff));
    }
}

#[cfg(test)]
mod tests {
    use crate::test_fft;

    #[test]
    fn sfft_correctness() {
        test_fft(0)
    }

    #[test]
    fn pfft_rayon_correctness() {
        test_fft(1)
    }

    #[test]
    fn pfft_simd_correctness() {
        test_fft(2)
    }

    #[test]
    fn pfft_rayon_simd_correctness() {
        test_fft(3)
    }
}
