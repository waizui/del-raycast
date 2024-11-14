
fn main() -> anyhow::Result<()>{
    let camera_fov = 20.0;
    let transform_cam_lcl2glbl = del_geo_core::mat4_col_major::from_translate(&[0., 0., -5.]);
    let img_shape = (400, 300);
    let mut img = Vec::<image::Rgb<f32>>::new();
    img.resize(img_shape.0 * img_shape.1, image::Rgb([0_f32; 3]));
    for ih in 0..img_shape.1 {
        for iw in 0..img_shape.0 {
            let (ray_org, ray_dir) = del_raycast::cam_pbrt::cast_ray(
                iw,
                ih,
                img_shape,
                camera_fov,
                transform_cam_lcl2glbl,
            );
            // dbg!(ray_org, ray_dir);
            img[ih * img_shape.0 + iw] = image::Rgb([0f32; 3]);
            // dbg!(v);
        }
    }
    {
        use image::codecs::hdr::HdrEncoder;
        let file = std::fs::File::create("target/04_env_light.hdr").unwrap();
        let enc = HdrEncoder::new(file);
        let _ = enc.encode(&img, img_shape.0, img_shape.1);
    }
    Ok(())
}