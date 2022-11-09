use std::f32::consts::PI;
use num_complex as nc;

pub fn bitrp(x: &mut[nc::Complex<f32>], n: i64) {
    let mut i = 1;
    let mut p = 0;
    while i < n {
        p += 1;
        i *= 2;
    }
    for i in 1..n {
        let mut a = i;
        let mut b = 0;

        for _ in 0..p {
            b = (b << 1) + (a & 1);
            a >>= 1;
        }
        if b > i {
            x.swap(i.try_into().unwrap(),
                   b.try_into().unwrap());
        }
    }
}

pub fn fft(x: &mut [nc::Complex<f32>], n: i64) {
    let N: usize = n as usize; // not sure about the correctness
    let mut w = vec![nc::Complex::new(1.0, 0.0); N / 2];

    bitrp(&mut x[..], n);

    let arg = -2.0 * PI / (n as f32);
    let mut t = nc::Complex::new(
        arg.cos(), arg.sin()
    );

    for j in 1_usize..(n as usize / 2) {
        w[j].re = w[j - 1].re * t.re - w[j - 1].im * t.im;
        w[j].im = w[j - 1].re * t.im + w[j - 1].im * t.re;
    }

    let mut m = 2_usize;

    while m <= n.try_into().unwrap() {
        for k in (0_usize..n as usize).step_by(m) {
            for j in 0_usize..(m / 2) {
                let idx1: usize = k + j;
                let idx2: usize = idx1 + m / 2;
                let tt: usize = (n as usize) * j / m;

                t.re = w[tt].re * x[idx2].re - w[tt].im * x[idx2].im;
                t.im = w[tt].re * x[idx2].im + w[tt].im * x[idx2].re;

                let u = x[idx1].clone();

                x[idx1] = u + t;
                x[idx2] = u - t;
            }
        }
        m *= 2;
    }
}

pub fn ifft(x: &mut [nc::Complex<f32>], n: i64) {
    let N: usize = n as usize; // not sure about the correctness
    let mut w = vec![nc::Complex::new(1.0, 0.0); N / 2];

    bitrp(&mut x[..], n);

    let arg = 2.0 * PI / (n as f32);
    let mut t = nc::Complex::new(
        arg.cos(), arg.sin()
    );

    for j in 1_usize..(n as usize / 2) {
        w[j].re = w[j - 1].re * t.re - w[j - 1].im * t.im;
        w[j].im = w[j - 1].re * t.im + w[j - 1].im * t.re;
    }

    let mut m = 2_usize;

    while m <= n.try_into().unwrap() {
        for k in (0_usize..n as usize).step_by(m) {
            for j in 0_usize..(m / 2) {
                let idx1: usize = k + j;
                let idx2: usize = idx1 + m / 2;
                let tt: usize = (n as usize) * j / m;

                t.re = w[tt].re * x[idx2].re - w[tt].im * x[idx2].im;
                t.im = w[tt].re * x[idx2].im + w[tt].im * x[idx2].re;

                let u = x[idx1].clone();

                x[idx1] = u + t;
                x[idx2] = u - t;
            }
        }
        m *= 2;
    }
    for j in 0_usize..n as usize {
        x[j] = x[j].unscale(n as f32);
    }
}
