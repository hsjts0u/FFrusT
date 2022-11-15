use hound;

mod sfft;
mod utils;

fn main() {
    let mut reader = hound::WavReader::open("audio/sine.wav").unwrap();
    let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();

    let mut complex_data = utils::initialize(&samples);

    sfft::fft(&mut complex_data);
    sfft::ifft(&mut complex_data);

    let spec = reader.spec();
    let mut writer = hound::WavWriter::create("serial_reconstruct.wav", spec).unwrap();
    for s in 0..samples.len() {
        writer.write_sample(complex_data[s].re as i16).unwrap();
    }
}
