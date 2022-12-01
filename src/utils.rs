use num_complex::Complex;

pub fn initialize<T>(real_data: &Vec<T>) -> Vec<Complex<f32>>
where
    T: Into<f32> + Copy,
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
