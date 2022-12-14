use num_complex::Complex;
use std::f64::consts::PI;

fn reverse_bits(n: u64, bits: u64) -> u64 {
    let mut result = 0;
    for i in 0..bits {
        result = (result << 1) + (n >> i & 1);
    }

    result
}

fn reorder(complex_data: &mut Vec<Complex<f64>>, log2n: u64) {
    for i in 0..complex_data.len() as u64 {
        let x = reverse_bits(i, log2n);
        if x > i {
            complex_data.swap(i as usize, x as usize);
        }
    }
}

fn cooley_tukey(complex_data: &mut Vec<Complex<f64>>, inv: bool) {
    let sign = if inv { 1.0 } else { -1.0 };
    let mut log2n: u64 = 0;
    let n = complex_data.len();
    while (1 << log2n) < n {
        log2n += 1;
    }

    reorder(complex_data, log2n);

    for i in 1..=log2n {
        let m = 1 << i;
        let deltawn = Complex {
            re: f64::cos(2.0 * PI / m as f64),
            im: sign * f64::sin(2.0 * PI / m as f64),
        };

        for k in (0..complex_data.len()).step_by(m) {
            let mut wn = Complex { re: 1.0, im: 0.0 };

            for j in 0..m / 2 {
                let t = wn * complex_data[k + j + m / 2];
                let u = complex_data[k + j];
                complex_data[k + j] = u + t;
                complex_data[k + j + m / 2] = u - t;
                wn = wn * deltawn;
            }
        }
    }
}

pub fn fft(complex_data: &mut Vec<Complex<f64>>) {
    cooley_tukey(complex_data, false);
}

pub fn ifft(complex_data: &mut Vec<Complex<f64>>) {
    cooley_tukey(complex_data, true);
    let n = complex_data.len() as f64;
    complex_data.iter_mut().for_each(|x| *x = *x / n);
}
