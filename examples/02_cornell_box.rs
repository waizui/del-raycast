use pbrt4::types::Light;

#[derive(Debug)]
struct TriangleMesh {
    // vtx2uv: Vec<f32>,
    // vtx2nrm: Vec<f32>,
    vtx2xyz: Vec<f32>,
    tri2vtx: Vec<usize>,
    normal: Vec<f32>,
    reflectance: [f32; 3],
}

impl TriangleMesh {
    fn raw_mesh(_vtx2xyz: Vec<f32>, _tri2vtx: Vec<usize>) -> Self {
        Self {
            vtx2xyz: _vtx2xyz,
            tri2vtx: _tri2vtx,
            normal: vec![],
            reflectance: [0.0, 0.0, 0.0],
        }
    }
    fn new(
        _vtx2xyz: Vec<f32>,
        _tri2vtx: Vec<usize>,
        _normal: Vec<f32>,
        _reflectance: [f32; 3],
    ) -> Self {
        Self {
            vtx2xyz: _vtx2xyz,
            tri2vtx: _tri2vtx,
            normal: _normal,
            reflectance: _reflectance,
        }
    }
}

fn parse_pbrt_file(
    file_path: &str,
) -> anyhow::Result<(Vec<TriangleMesh>, f32, [f32; 16], (usize, usize))> {
    let scene = pbrt4::Scene::from_file(file_path)?;
    // println!("{:?}", scene.materials[0]);
    // println!("{:?}", scene.shapes[0]);
    let (camera_fov, transform_cam_glbl2lcl, img_shape) = del_raycast::parse_pbrt::hoge(&scene);
    let mut shapes: Vec<TriangleMesh> = vec![];
    // println!("{:?}", shapes);
    for material in scene.materials {
        match material {
            pbrt4::types::Material {
                name,
                attributes,
                reflectance,
            } => {
                let ts = TriangleMesh::new(vec![], vec![], vec![], reflectance.get_rgb());
                shapes.push(ts);
            }
            _ => {}
        }
    }
    for shape_entity in scene.shapes {
        let (idx, tri2vtx, vtx2xyz, normal) =
            del_raycast::parse_pbrt::trimesh3_from_shape_entity(&shape_entity, file_path).unwrap();
        shapes[idx].vtx2xyz = vtx2xyz;
        shapes[idx].tri2vtx = tri2vtx;
        shapes[idx].normal = normal;
        // let ts = TriangleMesh::raw_mesh(vtx2xyz, tri2vtx);
        // shapes.push(ts);
    }
    /*
    for shape in &shapes {
        println!("{:?}", shape);
    }
    */
    Ok((shapes, camera_fov, transform_cam_glbl2lcl, img_shape))
}

/// For Cornell Box, we assume that no emittance from cube.
fn trace_ray(
    scene: &Vec<TriangleMesh>,
    camera_fov: f32,
    transform_cam_glbl2lcl: [f32; 16],
    img_shape: (usize, usize),
) -> Vec<image::Rgb<f32>> {
    let mut img = Vec::<image::Rgb<f32>>::new();
    img.resize(img_shape.0 * img_shape.1, image::Rgb([0_f32; 3]));
    for iw in 0..img_shape.0 {
        for ih in 0..img_shape.1 {
            let (ray_org, ray_dir) = del_raycast::cam_pbrt::cast_ray(
                iw,
                ih,
                img_shape,
                camera_fov,
                transform_cam_glbl2lcl,
            );
            // compute intersection below
            let mut t_min = f32::INFINITY;
            let mut Lo = [0.0, 0.0, 0.0];
            let mut target_mesh: &TriangleMesh = &scene[0];
            let mut hit_pos = [0.0, 0.0, 0.0];
            for trimesh in scene.iter() {
                let Some((t, hit_pos)) =
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
                    target_mesh = trimesh;
                }
            }

            // Ray Tracing
            let wo = nalgebra::Vector3::new(-ray_dir[0], -ray_dir[1], -ray_dir[2]);
            let light_sources: Vec<Light> = vec![];
            Lo = shade(wo, &hit_pos, target_mesh, &light_sources);

            // store Lo to img
            img[ih * img_shape.0 + iw] = image::Rgb(Lo);
        }
    }
    img
}

fn shade(wo: nalgebra::Vector3<f32>, hit_pos: &[f32; 3], target_msh: &TriangleMesh, light_sources: &Vec<Light>) -> [f32; 3] {
    // normal at hit_pos
    let n = interpolate(
        &target_msh.normal,
        &target_msh.tri2vtx,
        &target_msh.vtx2xyz,
        &hit_pos,
    );
    let mut Lo = [0.0, 0.0, 0.0];
    for i in 1..16 {
        let wi = del_raycast::sampling::hemisphere_cos_weighted(&n, &rand::random::<[f32; 2]>());
        let Some((t, next_hit)) = del_msh_core::trimesh3_search_bruteforce::first_intersection_ray(
            &hit_pos,
            &wi.into(),
            &target_msh.tri2vtx,
            &target_msh.vtx2xyz,
        ) else {
            todo!();
        };
    }
    Lo
}

fn interpolate(
    params: &Vec<f32>,
    tri2vtx: &Vec<usize>,
    vtx2xyz: &Vec<f32>,
    hit_pos: &[f32; 3],
) -> nalgebra::Vector3<f32> {
    todo!()
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
            "target/02_cornell_box.obj",
            &tri2vtx,
            &vtx2xyz,
            3,
        )?;
    }
    let transform_cam_lcl2glbl =
        del_geo_core::mat4_col_major::try_inverse(&transform_cam_glbl2lcl).unwrap();

    let mut img = Vec::<image::Rgb<f32>>::new();
    img.resize(img_shape.0 * img_shape.1, image::Rgb([0_f32; 3]));
    let mut img1 = Vec::<image::Rgb<f32>>::new();
    img1.resize(img_shape.0 * img_shape.1, image::Rgb([0_f32; 3]));
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
            let mut color_buf = [0.0, 0.0, 0.0];
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
                    color_buf = trimesh.reflectance;
                }
            }
            let v = t_min * 0.05;
            img[ih * img_shape.0 + iw] = image::Rgb([v; 3]);
            img1[ih * img_shape.0 + iw] = image::Rgb(color_buf);
            // dbg!(t_min);
        }
    }
    {
        use image::codecs::hdr::HdrEncoder;
        let file = std::fs::File::create("target/02_cornell_box.hdr").unwrap();
        let enc = HdrEncoder::new(file);
        let _ = enc.encode(&img, img_shape.0, img_shape.1);
        let file1 = std::fs::File::create("target/02_cornell_box_color.hdr").unwrap();
        let enc = HdrEncoder::new(file1);
        let _ = enc.encode(&img1, img_shape.0, img_shape.1);
    }
    Ok(())
}
