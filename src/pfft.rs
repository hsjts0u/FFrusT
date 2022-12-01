use num_complex::Complex;
use rayon::prelude::*;
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

#[inline]
fn comp_mult_re(a_re: f64, a_im: f64, b_re: f64, b_im: f64) -> f64 {
    return (a_re * b_re) - (a_im * b_im);
}

#[inline]
fn comp_mult_im(a_re: f64, a_im: f64, b_re: f64, b_im: f64) -> f64 {
    return (a_re * b_im) + (a_im * b_re);
}

fn simd_rfft(data_re: &mut Vec<f64>, data_im: &mut Vec<f64>, inv: bool) {
    if data_re.len() != data_im.len() {
        return;
    }

    let n = data_re.len();
    if n == 1 {
        return;
    }

    let sign = if inv { 1.0 } else { -1.0 };

    let mut a0_re = Vec::with_capacity(n / 2);
    let mut a0_im = Vec::with_capacity(n / 2);
    let mut a1_re = Vec::with_capacity(n / 2);
    let mut a1_im = Vec::with_capacity(n / 2);

    for i in 0..n / 2 {
        a0_re.push(data_re[2 * i]);
        a0_im.push(data_im[2 * i]);
        a1_re.push(data_re[2 * i + 1]);
        a1_im.push(data_im[2 * i + 1]);
    }

    rayon::join(
        || simd_rfft(&mut a0_re, &mut a0_im, inv),
        || simd_rfft(&mut a1_re, &mut a1_im, inv),
    );

    let mut w_re = 1.0;
    let mut w_im = 0.0;
    let wn_re = f64::cos(sign * 2.0 * PI / n as f64);
    let wn_im = f64::sin(sign * 2.0 * PI / n as f64);

    for i in 0..n / 2 {
        data_re[i] = a0_re[i] + comp_mult_re(w_re, w_im, a1_re[i], a1_im[i]);
        data_im[i] = a0_im[i] + comp_mult_im(w_re, w_im, a1_re[i], a1_im[i]);
        data_re[i + n / 2] = a0_re[i] - comp_mult_re(w_re, w_im, a1_re[i], a1_im[i]);
        data_im[i + n / 2] = a0_im[i] - comp_mult_im(w_re, w_im, a1_re[i], a1_im[i]);
        if inv {
            data_re[i] /= 2.0;
            data_im[i] /= 2.0;
            data_re[i + n / 2] /= 2.0;
            data_im[i + n / 2] /= 2.0;
        }
        let wt_re = w_re;
        let wt_im = w_im;
        w_re = comp_mult_re(wt_re, wt_im, wn_re, wn_im);
        w_im = comp_mult_im(wt_re, wt_im, wn_re, wn_im);
    }
}

pub fn fft(complex_data: &mut Vec<Complex<f64>>) {
    rfft(complex_data, false);
}

pub fn ifft(complex_data: &mut Vec<Complex<f64>>) {
    rfft(complex_data, true);
}

pub fn simd_fft(complex_data: &mut Vec<Complex<f64>>) {
    let mut data_re: Vec<f64> = complex_data.into_iter().map(|x| x.re).collect();
    let mut data_im: Vec<f64> = complex_data.into_iter().map(|x| x.im).collect();
    simd_rfft(&mut data_re, &mut data_im, false);
}

pub fn simd_ifft(complex_data: &mut Vec<Complex<f64>>) {
    let mut data_re: Vec<f64> = complex_data.into_iter().map(|x| x.re).collect();
    let mut data_im: Vec<f64> = complex_data.into_iter().map(|x| x.im).collect();
    simd_rfft(&mut data_re, &mut data_im, true);
}
