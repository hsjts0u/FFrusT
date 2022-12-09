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

#[inline(always)]
fn comp_mult_re(a_re: f64, a_im: f64, b_re: f64, b_im: f64) -> f64 {
    return (a_re * b_re) - (a_im * b_im);
}

#[inline(always)]
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

    let mut a0_re: Vec<f64> = vec![0.0; n / 2];
    let mut a0_im: Vec<f64> = vec![0.0; n / 2];
    let mut a1_re: Vec<f64> = vec![0.0; n / 2];
    let mut a1_im: Vec<f64> = vec![0.0; n / 2];

    for i in 0..n / 2 {
        a0_re[i] = data_re[2 * i];
        a0_im[i] = data_im[2 * i];
        a1_re[i] = data_re[2 * i + 1];
        a1_im[i] = data_im[2 * i + 1];
    }

    simd_rfft(&mut a0_re, &mut a0_im, inv);
    simd_rfft(&mut a1_re, &mut a1_im, inv);

    let wn = Complex {
        re: f64::cos(sign * 2.0 * PI / n as f64),
        im: f64::sin(sign * 2.0 * PI / n as f64),
    };

    let mut warr: Vec<Complex<f64>> = Vec::new();
    warr.resize(n / 2, Complex::default());

    warr[0] = Complex { re: 1.0, im: 0.0 };
    for i in 1..n / 2 {
        warr[i] = warr[i - 1] * wn;
    }

    for i in 0..n / 2 {
        data_re[i] = a0_re[i] + comp_mult_re(warr[i].re, warr[i].im, a1_re[i], a1_im[i]);
        data_im[i] = a0_im[i] + comp_mult_im(warr[i].re, warr[i].im, a1_re[i], a1_im[i]);
        data_re[i + n / 2] = a0_re[i] - comp_mult_re(warr[i].re, warr[i].im, a1_re[i], a1_im[i]);
        data_im[i + n / 2] = a0_im[i] - comp_mult_im(warr[i].re, warr[i].im, a1_re[i], a1_im[i]);
        if inv {
            data_re[i] /= 2.0;
            data_im[i] /= 2.0;
            data_re[i + n / 2] /= 2.0;
            data_im[i + n / 2] /= 2.0;
        }
    }
}

fn rayon_simd_rfft(data_re: &mut Vec<f64>, data_im: &mut Vec<f64>, inv: bool) {
    if data_re.len() != data_im.len() {
        return;
    }

    let n = data_re.len();
    if n == 1 {
        return;
    }

    let sign = if inv { 1.0 } else { -1.0 };

    let mut a0_re: Vec<f64> = vec![0.0; n / 2];
    let mut a0_im: Vec<f64> = vec![0.0; n / 2];
    let mut a1_re: Vec<f64> = vec![0.0; n / 2];
    let mut a1_im: Vec<f64> = vec![0.0; n / 2];

    for i in 0..n / 2 {
        a0_re[i] = data_re[2 * i];
        a0_im[i] = data_im[2 * i];
        a1_re[i] = data_re[2 * i + 1];
        a1_im[i] = data_im[2 * i + 1];
    }

    rayon::join(
        || rayon_simd_rfft(&mut a0_re, &mut a0_im, inv),
        || rayon_simd_rfft(&mut a1_re, &mut a1_im, inv),
    );

    let wn = Complex {
        re: f64::cos(sign * 2.0 * PI / n as f64),
        im: f64::sin(sign * 2.0 * PI / n as f64),
    };

    let mut warr: Vec<Complex<f64>> = Vec::new();
    warr.resize(n / 2, Complex::default());

    warr[0] = Complex { re: 1.0, im: 0.0 };
    for i in 1..n / 2 {
        warr[i] = warr[i - 1] * wn;
    }

    for i in 0..n / 2 {
        data_re[i] = a0_re[i] + comp_mult_re(warr[i].re, warr[i].im, a1_re[i], a1_im[i]);
        data_im[i] = a0_im[i] + comp_mult_im(warr[i].re, warr[i].im, a1_re[i], a1_im[i]);
        data_re[i + n / 2] = a0_re[i] - comp_mult_re(warr[i].re, warr[i].im, a1_re[i], a1_im[i]);
        data_im[i + n / 2] = a0_im[i] - comp_mult_im(warr[i].re, warr[i].im, a1_re[i], a1_im[i]);
        if inv {
            data_re[i] /= 2.0;
            data_im[i] /= 2.0;
            data_re[i + n / 2] /= 2.0;
            data_im[i + n / 2] /= 2.0;
        }
    }
}

pub fn rayon_fft(complex_data: &mut Vec<Complex<f64>>) {
    rfft(complex_data, false);
}

pub fn rayon_ifft(complex_data: &mut Vec<Complex<f64>>) {
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

pub fn rayon_simd_fft(complex_data: &mut Vec<Complex<f64>>) {
    let mut data_re: Vec<f64> = complex_data.into_iter().map(|x| x.re).collect();
    let mut data_im: Vec<f64> = complex_data.into_iter().map(|x| x.im).collect();
    rayon_simd_rfft(&mut data_re, &mut data_im, false);
}

pub fn rayon_simd_ifft(complex_data: &mut Vec<Complex<f64>>) {
    let mut data_re: Vec<f64> = complex_data.into_iter().map(|x| x.re).collect();
    let mut data_im: Vec<f64> = complex_data.into_iter().map(|x| x.im).collect();
    rayon_simd_rfft(&mut data_re, &mut data_im, true);
}
