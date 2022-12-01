use num_complex::Complex;

pub fn scaledown(complex_data: &mut Vec<Complex<f32>>, sample_rate: u32, freq: i32) {
    let points_per_freq = complex_data.len() as f32 / sample_rate as f32;

    let target_idx = points_per_freq * freq as f32;
    let target_idx_n = complex_data.len() - target_idx as usize;

    let start = target_idx as usize - 100;
    let end = target_idx as usize + 100;
    let start_n = target_idx_n - 100;
    let end_n = target_idx_n + 100;

    for i in start..end {
        complex_data[i].re = 0.0;
        complex_data[i].im = 0.0;
    }

    for i in start_n..end_n {
        complex_data[i].re = 0.0;
        complex_data[i].im = 0.0;
    }

}