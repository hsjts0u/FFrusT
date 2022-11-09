use hound;

fn main() {

    let mut reader = hound::WavReader::open("audio/rr.wav").unwrap();
    let samples : Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();
    
    for i in samples.iter() {
        println!("{}", i);
    }

}
