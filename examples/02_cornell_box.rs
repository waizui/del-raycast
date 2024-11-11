use std::io::{BufRead, Seek, SeekFrom};

struct TriangleMesh {
    // vtx2uv: Vec<f32>,
    // vtx2nrm: Vec<f32>,
    vtx2xyz: Vec<f32>,
    tri2vtx: Vec<usize>,
}

fn parse_pbrt_file(
    file_path: &str,
) -> anyhow::Result<(Vec<TriangleMesh>, f32, [f32; 16], (usize, usize))> {
    let scene = pbrt4::Scene::from_file(file_path)?;
    let (camera_fov, transform_cam_glbl2lcl, img_shape) = del_raycast::parse_pbrt::hoge(&scene);
    let mut shapes: Vec<TriangleMesh> = vec![];
    for shape_entity in scene.shapes {
        let (tri2vtx, vtx2xyz) =
            del_raycast::parse_pbrt::trimesh3_from_shape_entity(&shape_entity, file_path).unwrap();
        let ts = TriangleMesh { tri2vtx, vtx2xyz };
        shapes.push(ts);
    }
    for material in scene.materials {
        match material {
            pbrt4::types::Material {ty} => {dbg!(ty);},
            _ => {}
        }
    }
    Ok((shapes, camera_fov, transform_cam_glbl2lcl, img_shape))
}

fn main() -> anyhow::Result<()> {
    let pbrt_file_path = "examples/asset/cornell-box/scene-v4.pbrt";
    let (trimeshs, camera_fov, transform_cam_glbl2lcl, img_shape) =
        parse_pbrt_file(pbrt_file_path)?;
    {
        let mut tri2vtx: Vec<usize> = vec![];
        let mut vtx2xyz: Vec<f32> = vec![];
        for trimesh in trimeshs.iter() {
            let trimesh_vtx2xyz =
                del_msh_core::vtx2xyz::transform(&trimesh.vtx2xyz, &transform_cam_glbl2lcl);
            del_msh_core::uniform_mesh::merge(
                &mut tri2vtx,
                &mut vtx2xyz,
                &trimesh.tri2vtx,
                &trimesh_vtx2xyz,
                3,
            );
        }
        del_msh_core::io_obj::save_tri2vtx_vtx2xyz(
            "target/2_cornell_box.obj",
            &tri2vtx,
            &vtx2xyz,
            3,
        )?;
    }
    let transform_cam_lcl2glbl =
        del_geo_core::mat4_col_major::try_inverse(&transform_cam_glbl2lcl).unwrap();

    let mut img = Vec::<image::Rgb<f32>>::new();
    img.resize(img_shape.0 * img_shape.1, image::Rgb([0_f32; 3]));
    for iw in 0..img_shape.0 {
        for ih in 0..img_shape.1 {
            let (ray_org, ray_dir) = del_raycast::cam_pbrt::cast_ray(
                iw,
                ih,
                img_shape,
                camera_fov,
                transform_cam_lcl2glbl,
            );
            // compute intersection below
            let mut t_min = f32::INFINITY;
            for trimesh in trimeshs.iter() {
                let Some((t, _i_tri)) =
                    del_msh_core::trimesh3_search_bruteforce::first_intersection_ray(
                        &ray_org,
                        &ray_dir,
                        &trimesh.tri2vtx,
                        &trimesh.vtx2xyz,
                    )
                else {
                    continue;
                };
                if t < t_min {
                    t_min = t;
                }
            }
            let v = t_min * 0.05;
            img[ih * img_shape.0 + iw] = image::Rgb([v; 3]);
            // dbg!(t_min);
        }
    }
    {
        use image::codecs::hdr::HdrEncoder;
        let file = std::fs::File::create("target/02_cornell_box.hdr").unwrap();
        let enc = HdrEncoder::new(file);
        let _ = enc.encode(&img, img_shape.0, img_shape.1);
    }
    Ok(())
}
