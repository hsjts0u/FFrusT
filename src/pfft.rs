use num_complex::Complex;
use std::f64::consts::PI;

fn rfft(complex_data: &mut Vec<Complex<f64>>, inv: bool) {
    let n = complex_data.len();
    if n == 1 {
        return;
    }

    let sign = if inv { 1.0 } else { -1.0 };

    let mut a0 = Vec::with_capacity(n / 2);
    let mut a1 = Vec::with_capacity(n / 2);

    for i in 0..n / 2 {
        a0.push(complex_data[2 * i]);
        a1.push(complex_data[2 * i + 1]);
    }

    rayon::join(|| rfft(&mut a0, inv), || rfft(&mut a1, inv));

    let mut w = Complex { re: 1.0, im: 0.0 };
    let wn = Complex {
        re: f64::cos(sign * 2.0 * PI / n as f64),
        im: f64::sin(sign * 2.0 * PI / n as f64),
    };

    for i in 0..n / 2 {
        complex_data[i] = a0[i] + w * a1[i];
        complex_data[i + n / 2] = a0[i] - w * a1[i];
        if inv {
            complex_data[i] /= 2.0;
            complex_data[i + n / 2] /= 2.0;
        }
        w *= wn;
    }
}

pub fn rayon_fft(complex_data: &mut Vec<Complex<f64>>) {
    rfft(complex_data, false);
}

pub fn rayon_ifft(complex_data: &mut Vec<Complex<f64>>) {
    rfft(complex_data, true);
}
