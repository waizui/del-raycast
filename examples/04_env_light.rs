use core::str;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};

use anyhow::anyhow;

pub struct PFM {
    pub data: Vec<f32>,
    pub w: usize,
    pub h: usize,
    pub channels: usize,
    pub little_endian: bool,
}

impl PFM {
    pub fn pixel(&self, iw: usize, ih: usize) -> Option<&[f32]> {
        if iw >= self.w || ih >= self.h {
            return None;
        }
        // top left is origin, slightly different from standard pfm definition
        let start = ((self.h - ih - 1) * self.w + iw) * self.channels;
        // uncomment this for bottom left origin
        // let start = (ih  * self.w + iw) * self.channels;
        Some(&self.data[start..start + self.channels])
    }

    pub fn rgb(&self, iw: usize, ih: usize) -> Option<[f32; 3]> {
        if self.channels != 3 {
            return None;
        }
        self.pixel(iw, ih).map(|p| [p[0], p[1], p[2]])
    }
}

fn read_pfm(path: &str) -> anyhow::Result<PFM> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut header = String::new();
    reader.read_line(&mut header)?;
    let header = header.trim();

    let channels = match header {
        "PF" => 3, // color image
        "Pf" => 1, // grayscale image
        _ => return Err(anyhow!("Invalid header of PFM")),
    };

    let mut dimensions = String::new();
    reader.read_line(&mut dimensions)?;
    let dims: Vec<&str> = dimensions.trim().split_whitespace().collect();

    if dims.len() != 2 {
        return Err(anyhow!("Invalid dimensions format"));
    }

    let w: usize = dims[0]
        .parse()
        .map_err(|_| anyhow!("Invalid width: {}", dims[0]))?;
    let h: usize = dims[1]
        .parse()
        .map_err(|_| anyhow!("Invalid height: {}", dims[1]))?;

    // scale factor, little endian if scale is negative
    let mut scale_line = String::new();
    reader.read_line(&mut scale_line)?;
    let scale: f32 = scale_line
        .trim()
        .parse()
        .map_err(|_| anyhow!("Invalid scale factor: {}", scale_line.trim()))?;

    let little_endian = scale < 0.0;

    // binary data
    let data_size = w * h * channels;
    let mut buffer = vec![0u8; data_size * 4]; // 4 bytes per f32
    reader.read_exact(&mut buffer)?;

    // convert bytes to f32 values
    let mut data = Vec::with_capacity(data_size);
    for chunk in buffer.chunks_exact(4) {
        let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
        let value = if little_endian {
            f32::from_le_bytes(bytes)
        } else {
            f32::from_be_bytes(bytes)
        };
        data.push(value);
    }

    Ok(PFM {
        data,
        w,
        h,
        channels,
        little_endian: true,
    })
}

fn main() -> anyhow::Result<()> {
    let pfm = read_pfm("examples/asset/material-testball/textures/envmap.pfm")?;
    let camera_fov = 20.0;
    let transform_cam_lcl2glbl = del_geo_core::mat4_col_major::from_translate(&[0., 0., -5.]);
    let img_shape = (pfm.w, pfm.h);
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
            // img[ih * img_shape.0 + iw] = image::Rgb([0f32; 3]);
            img[ih * img_shape.0 + iw] = image::Rgb(pfm.rgb(iw, ih).unwrap());
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
