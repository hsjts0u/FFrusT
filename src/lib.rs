#![feature(test)]

extern crate test;
pub mod pfft;
pub mod sfft;
pub mod utils;

#[cfg(test)]
mod tests {
    use super::*;
    use hound;
    use test::Bencher;

    #[test]
    fn test_fft() {
        let mut reader = hound::WavReader::open("audio/sine.wav").unwrap();
        let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();

        let mut complex_data = utils::initialize(&samples);

        sfft::fft(&mut complex_data);
        sfft::ifft(&mut complex_data);

        let new_samples: Vec<i16> = complex_data.iter().map(|c| c.re as i16).collect();

        let tolerance = -1..2;
        for i in 0..samples.len() {
            let diff = samples[i] - new_samples[i];
            assert!(tolerance.contains(&diff));
        }
    }

    #[bench]
    fn bench_fft(b: &mut Bencher) {
        b.iter(|| test_fft());
    }
}
