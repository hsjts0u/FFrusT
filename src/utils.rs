use num_complex::Complex;

pub fn initialize(real_data: &Vec<i16>) -> Vec<Complex<f32>> {
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
                real_data[i] as f32
            } else {
                0.0
            },
            im: 0.0,
        };
        a.push(point);
    }

    a
}
