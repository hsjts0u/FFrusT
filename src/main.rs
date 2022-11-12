use hound;
use num_complex::Complex;
use std::f32::consts::PI;

fn reverse_bits(n: u64, bits: u64) -> u64 {
    let mut result = 0;
    for i in 0..bits {
        result = (result << 1) + (n >> i & 1);
    }

    result
}

fn reorder(complex_data: &mut Vec<Complex<f32>>, log2n: u64) {
    for i in 0..complex_data.len() as u64 {
        let x = reverse_bits(i, log2n);
        if x > i {
            complex_data.swap(i as usize, x as usize);
        }
    }
}

fn fft(complex_data: &mut Vec<Complex<f32>>, inv: bool) {

    let sign = if inv { 1.0 } else { -1.0 };
    let mut log2n: u64 = 0;
    let mut n = complex_data.len();
    while n != 1 {
        log2n += 1;
        n >>= 1;
    }

    reorder(complex_data, log2n);

    for i in 1..=log2n {
        
        let m = 1 << i;
        let deltawn = Complex {
            re: f32::cos(2.0 * PI / m as f32),
            im: sign * f32::sin(2.0 * PI / m as f32),
        };
        
        for k in (0..complex_data.len()).step_by(m) {
            
            let mut wn = Complex{
                re: 1.0,
                im: 0.0
            };

            for j in 0..m/2 {
                let t = wn * complex_data[k + j + m/2];
                let u = complex_data[k + j];
                complex_data[k + j] = u + t;
                complex_data[k + j + m/2] = u - t;
                wn = wn * deltawn;
            }

        }

    }
}

fn initialize(real_data: &Vec<i16>) -> Vec<Complex<f32>> {
    let mut len = real_data.len();
    if len & (len - 1) != 0 {
        let mut base_two = 1;
        while base_two < len {
            base_two <<= 1;
        }
        len = base_two;
    }

    let mut a = Vec::with_capacity(len);

    for i in 0..len {
        let point = Complex {
            re: if i < real_data.len() { real_data[i] as f32 } else { 0.0 },
            im: 0.0,
        };
        a.push(point);
    }

    a
}

fn main() {
    let mut reader = hound::WavReader::open("audio/sine.wav").unwrap();
    let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();

    let mut complex_data = initialize(&samples);

    fft(&mut complex_data, false);
    fft(&mut complex_data, true);

    for s in 0..samples.len() {
        println!("samples: {}, cd: {}", samples[s], (complex_data[s].re / complex_data.len() as f32) as i16);
    }

    let spec = reader.spec();
    let mut writer = hound::WavWriter::create("reconstruct.wav", spec).unwrap();
    for s in 0..samples.len() {
        let a = (complex_data[s].re / complex_data.len() as f32) as i16;
        writer.write_sample(a).unwrap();
    }
}
