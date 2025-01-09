pub fn sample_checkerboard(uv: &[f32; 2], row: usize, col: usize) -> [f32; 3] {
    let u = uv[0];
    let v = uv[1];

    let d = (u * col as f32).floor() + (v * row as f32).floor();

    if d as usize % 2 == 0 {
        [0.; 3]
    } else {
        [1.; 3]
    }
}
