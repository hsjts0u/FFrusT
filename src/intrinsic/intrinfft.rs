use num_complex::Complex;
use std::arch::x86_64::*;
use std::f64::consts::PI;
use std::mem;

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

    let mut w = Complex { re: 1.0, im: 0.0 };
    let wn = Complex {
        re: f64::cos(sign * 2.0 * PI / n as f64),
        im: f64::sin(sign * 2.0 * PI / n as f64),
    };

    for i in 0..n / 2 {
        unsafe {
            let v1 = _mm256_setr_pd(w.re, w.im, w.re, w.im);
            let v2 = _mm256_setr_pd(a1_re[i], a1_im[i], wn.re, wn.im);
            let neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
            let v3 = _mm256_mul_pd(v1, v2);
            let v2 = _mm256_permute_pd::<0x5>(v2);
            let v2 = _mm256_mul_pd(v2, neg);
            let v4 = _mm256_mul_pd(v1, v2);
            let v1 = _mm256_hsub_pd(v3, v4);
            let a: (f64, f64, f64, f64) = mem::transmute(v1);

            w.re = a.2;
            w.im = a.3;

            let pv = _mm256_setr_pd(a0_re[i], a0_im[i], a0_re[i], a0_im[i]);
            let qv = _mm256_setr_pd(a.0, a.1, -a.0, -a.1);
            let mut s = _mm256_add_pd(pv, qv);
            if inv {
                let d = _mm256_set1_pd(2.0);
                s = _mm256_div_pd(s, d);
            }
            let sum: (f64, f64, f64, f64) = mem::transmute(s);

            data_re[i] = sum.0;
            data_im[i] = sum.1;
            data_re[i + n / 2] = sum.2;
            data_im[i + n / 2] = sum.3;
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

    let mut w = Complex { re: 1.0, im: 0.0 };
    let wn = Complex {
        re: f64::cos(sign * 2.0 * PI / n as f64),
        im: f64::sin(sign * 2.0 * PI / n as f64),
    };

    for i in 0..n / 2 {
        unsafe {
            let v1 = _mm256_setr_pd(w.re, w.im, w.re, w.im);
            let v2 = _mm256_setr_pd(a1_re[i], a1_im[i], wn.re, wn.im);
            let neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
            let v3 = _mm256_mul_pd(v1, v2);
            let v2 = _mm256_permute_pd::<0x5>(v2);
            let v2 = _mm256_mul_pd(v2, neg);
            let v4 = _mm256_mul_pd(v1, v2);
            let v1 = _mm256_hsub_pd(v3, v4);
            let a: (f64, f64, f64, f64) = mem::transmute(v1);

            w.re = a.2;
            w.im = a.3;

            let pv = _mm256_setr_pd(a0_re[i], a0_im[i], a0_re[i], a0_im[i]);
            let qv = _mm256_setr_pd(a.0, a.1, -a.0, -a.1);
            let mut s = _mm256_add_pd(pv, qv);
            if inv {
                let d = _mm256_set1_pd(2.0);
                s = _mm256_div_pd(s, d);
            }
            let sum: (f64, f64, f64, f64) = mem::transmute(s);

            data_re[i] = sum.0;
            data_im[i] = sum.1;
            data_re[i + n / 2] = sum.2;
            data_im[i + n / 2] = sum.3;
        }
    }
}

pub fn simd_fft(complex_data: &mut Vec<Complex<f64>>) {
    let mut data_re: Vec<f64> = complex_data.into_iter().map(|x| x.re).collect();
    let mut data_im: Vec<f64> = complex_data.into_iter().map(|x| x.im).collect();
    simd_rfft(&mut data_re, &mut data_im, false);
    for i in 0..complex_data.len() {
        complex_data[i].re = data_re[i];
        complex_data[i].im = data_im[i];
    }
}

pub fn simd_ifft(complex_data: &mut Vec<Complex<f64>>) {
    let mut data_re: Vec<f64> = complex_data.into_iter().map(|x| x.re).collect();
    let mut data_im: Vec<f64> = complex_data.into_iter().map(|x| x.im).collect();
    simd_rfft(&mut data_re, &mut data_im, true);
    for i in 0..complex_data.len() {
        complex_data[i].re = data_re[i];
        complex_data[i].im = data_im[i];
    }
}

pub fn rayon_simd_fft(complex_data: &mut Vec<Complex<f64>>) {
    let mut data_re: Vec<f64> = complex_data.into_iter().map(|x| x.re).collect();
    let mut data_im: Vec<f64> = complex_data.into_iter().map(|x| x.im).collect();
    rayon_simd_rfft(&mut data_re, &mut data_im, false);
    for i in 0..complex_data.len() {
        complex_data[i].re = data_re[i];
        complex_data[i].im = data_im[i];
    }
}

pub fn rayon_simd_ifft(complex_data: &mut Vec<Complex<f64>>) {
    let mut data_re: Vec<f64> = complex_data.into_iter().map(|x| x.re).collect();
    let mut data_im: Vec<f64> = complex_data.into_iter().map(|x| x.im).collect();
    rayon_simd_rfft(&mut data_re, &mut data_im, true);
    for i in 0..complex_data.len() {
        complex_data[i].re = data_re[i];
        complex_data[i].im = data_im[i];
    }
}
