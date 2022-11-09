mod sfft;

use hound;
use num_complex as nc;

fn main() {

    let mut reader = hound::WavReader::open("audio/rr.wav").unwrap();
    let samples : Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();
    
    let mut sample_complex = Vec::<nc::Complex32>::new();

    for i in samples.iter() {
        println!("{}", i);
        break
    }

    for idx in 10100..10116 {
        sample_complex.push(
            nc::Complex32::new(samples[idx] as f32, 0f32)
        );
        println!("{}", samples[idx] as f32);
    }
    
    sfft::fft(&mut sample_complex[..], 16);
    sfft::ifft(&mut sample_complex[..], 16);

    for i in sample_complex.iter() {
        println!("{} {}", i.re, i.im)
        
    }
}
