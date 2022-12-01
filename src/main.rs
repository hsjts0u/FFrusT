use hound;
use std::time::Instant;

mod pfft;
mod rfft;
mod sfft;
mod utils;
mod freq_scale;

fn main() {
    let mut reader = hound::WavReader::open("audio/sine.wav").unwrap();
    let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();

    let mut complex_data = utils::initialize(&samples);

    pfft::fft(&mut complex_data);
    //for i in 100..800 {
        freq_scale::scaledown(&mut complex_data, reader.spec().sample_rate, 441);
    //}
    
    let start = Instant::now();
    pfft::ifft(&mut complex_data);
    let elapsed = start.elapsed();
    println!("Time elapsed: {} ms", elapsed.as_millis());

    let spec = reader.spec();
    let mut writer = hound::WavWriter::create("serial_reconstruct.wav", spec).unwrap();
    for s in 0..samples.len() {
        writer.write_sample(complex_data[s].re as i16).unwrap();
    }
}
