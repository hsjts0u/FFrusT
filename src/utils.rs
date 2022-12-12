use std::io::prelude::*;
use std::{fs::OpenOptions, time::Duration};

use num_complex::Complex;

pub fn initialize<T>(real_data: &Vec<T>) -> Vec<Complex<f64>>
where
    T: Into<f64> + Copy,
{
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
            re: if i < real_data.len() {
                real_data[i].into()
            } else {
                0.0
            },
            im: 0.0,
        };
        a.push(point);
    }

    a
}

pub fn dump_result(complex_data: &mut Vec<Duration>, filename: &str) -> std::io::Result<()> {
    let mut buf = OpenOptions::new()
        .append(true)
        .create(true)
        .open(filename)?;
    for (idx, t) in complex_data.iter().enumerate() {
        buf.write_fmt(format_args!("{}", &t.as_secs_f32()))?;
        if idx != complex_data.len() - 1 {
            buf.write(b", ")?;
        } else {
            buf.write(b"\n")?;
        }
    }
    Ok(())
}
