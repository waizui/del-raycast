use nalgebra::coordinates::X;
use rand::Rng;

#[derive(Debug, PartialEq)]
struct TriangleMesh {
    // vtx2uv: Vec<f32>,
    // vtx2nrm: Vec<f32>,
    vtx2xyz: Vec<f32>,
    tri2vtx: Vec<usize>,
    normal: Vec<f32>,
    reflectance: [f32; 3],
}

#[derive(Debug)]
struct LightSource {
    vtx2xyz: Vec<f32>,
    tri2vtx: Vec<usize>,
    normal: Vec<f32>,
    spectrum: [f32; 3],
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
) -> anyhow::Result<(
    Vec<TriangleMesh>,
    Vec<LightSource>,
    f32,
    [f32; 16],
    (usize, usize),
)> {
    let scene = pbrt4::Scene::from_file(file_path)?;
    let (camera_fov, transform_cam_glbl2lcl, img_shape) = del_raycast::parse_pbrt::hoge(&scene);
    let mut shapes: Vec<TriangleMesh> = vec![];
    let mut lights: Vec<LightSource> = vec![];
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
    for area_light in scene.area_lights {
        match area_light.clone() {
            pbrt4::types::AreaLight::Diffuse {
                filename,
                two_sided,
                spectrum,
                scale,
            } => {
                let light_src = LightSource {
                    vtx2xyz: vec![],
                    tri2vtx: vec![],
                    normal: vec![],
                    spectrum: del_raycast::parse_pbrt::spectrum_from_light_entity(&area_light)
                        .unwrap(),
                };
                lights.push(light_src);
            }
            _ => {}
        }
    }
    for shape_entity in scene.shapes {
        let (shape_idx, light_idx, tri2vtx, vtx2xyz, normal) =
            del_raycast::parse_pbrt::trimesh3_from_shape_entity(&shape_entity, file_path).unwrap();
        shapes[shape_idx].vtx2xyz = vtx2xyz.clone();
        shapes[shape_idx].tri2vtx = tri2vtx.clone();
        shapes[shape_idx].normal = normal.clone();
        match light_idx {
            Some(idx) => {
                lights[idx].vtx2xyz = vtx2xyz;
                lights[idx].tri2vtx = tri2vtx;
                lights[idx].normal = normal;
            }
            None => {}
        }
        // let ts = TriangleMesh::raw_mesh(vtx2xyz, tri2vtx);
        // shapes.push(ts);
    }
    Ok((
        shapes,
        lights,
        camera_fov,
        transform_cam_glbl2lcl,
        img_shape,
    ))
}

/// For Cornell Box, we assume that no emittance from cube.
fn trace_ray(
    trimeshes: &[TriangleMesh],
    light_msh: &TriangleMesh,
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
            let mut target_mesh: &TriangleMesh = &trimeshes[0];
            let mut hit_pos = [0.0, 0.0, 0.0];
            let mut hit_idx = [0, 0, 0];
            for trimesh in trimeshes {
                // dbg!(&trimesh.vtx2xyz);
                let Some((t, pos, idx)) =
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
                    hit_pos = pos;
                    hit_idx = idx;
                    target_mesh = trimesh;
                }
            }
            // Ray Tracing
            let wo = nalgebra::Vector3::new(-ray_dir[0], -ray_dir[1], -ray_dir[2]);
            Lo = shade(wo, &hit_pos, &hit_idx, target_mesh, &light_msh, trimeshes);

            // store Lo to img
            img[ih * img_shape.0 + iw] = image::Rgb(Lo);
        }
    }
    img
}

fn shade(
    wo: nalgebra::Vector3<f32>,
    hit_pos: &[f32; 3],
    hit_idx: &[usize; 3],
    target_msh: &TriangleMesh,
    light_sources: &TriangleMesh,
    meshes: &[TriangleMesh],
) -> [f32; 3] {
    // normal at triangle vertices
    let p0 = nalgebra::Vector3::new(
        target_msh.normal[hit_idx[0] * 3],
        target_msh.normal[hit_idx[0] * 3 + 1],
        target_msh.normal[hit_idx[0] * 3 + 2],
    );
    let p1 = nalgebra::Vector3::new(
        target_msh.normal[hit_idx[1] * 3],
        target_msh.normal[hit_idx[1] * 3 + 1],
        target_msh.normal[hit_idx[1] * 3 + 2],
    );
    let p2 = nalgebra::Vector3::new(
        target_msh.normal[hit_idx[2] * 3],
        target_msh.normal[hit_idx[2] * 3 + 1],
        target_msh.normal[hit_idx[2] * 3 + 2],
    );

    // coord at triangle vertices
    let v0 = [
        target_msh.vtx2xyz[hit_idx[0] * 3],
        target_msh.vtx2xyz[hit_idx[0] * 3 + 1],
        target_msh.vtx2xyz[hit_idx[0] * 3 + 2],
    ];
    let v1 = [
        target_msh.vtx2xyz[hit_idx[1] * 3],
        target_msh.vtx2xyz[hit_idx[1] * 3 + 1],
        target_msh.vtx2xyz[hit_idx[1] * 3 + 2],
    ];
    let v2 = [
        target_msh.vtx2xyz[hit_idx[2] * 3],
        target_msh.vtx2xyz[hit_idx[2] * 3 + 1],
        target_msh.vtx2xyz[hit_idx[2] * 3 + 2],
    ];

    // normal at hit_pos
    let n = interpolate(&p0, &p1, &p2, &v0, &v1, &v2, hit_pos);

    let mut L_o = [0.0, 0.0, 0.0];

    // direct lighting from light sources, importance sampling on rectangle area light.
    let mut L_dir = [0.0, 0.0, 0.0];
    let mut sampled_des = nalgebra::Vector3::new(0.0, 0.0, 0.0);
    // Hacking sampling method for rectangle area light. Uniform sampling for meshes will be implemented in the future.
    let mut rng = rand::thread_rng();
    let x_min = -0.24;
    let x_max = 0.23;
    let z_min = -0.22;
    let z_max = 0.16;
    let x = rng.gen_range(x_min..x_max);
    let z = rng.gen_range(z_min..z_max);
    sampled_des = nalgebra::Vector3::new(x, 1.98, z);

    let ref_ray_org = hit_pos;
    let ref_ray_dir = (sampled_des - nalgebra::Vector3::new(hit_pos[0], hit_pos[1], hit_pos[2]));
    let t_lightsrc = 1.0;
    if !is_blocked(
        ref_ray_org,
        &ref_ray_dir.clone().into(),
        &meshes,
        t_lightsrc,
    ) {
        let L_i = [17.0, 12.0, 4.0];
        let cos_theta = n.dot(&ref_ray_dir.into()).max(0.0);
        let A = (x_max - x_min) * (z_max - z_min);
        let pdf = 1.0 / A;
        L_dir = [
            L_i[0] * cos_theta / (t_lightsrc * t_lightsrc) * pdf,
            L_i[1] * cos_theta / (t_lightsrc * t_lightsrc) * pdf,
            L_i[2] * cos_theta / (t_lightsrc * t_lightsrc) * pdf,
        ];
    }
    L_o = [
        L_o[0] + L_dir[0],
        L_o[1] + L_dir[1],
        L_o[2] + L_dir[2],
    ];

    L_o
}

fn interpolate<T>(
    p0: &T,
    p1: &T,
    p2: &T,
    v0: &[f32; 3],
    v1: &[f32; 3],
    v2: &[f32; 3],
    target: &[f32; 3],
) -> T
where
    T: Clone + std::ops::Add<Output = T> + std::ops::Mul<f32, Output = T>,
{
    let v0v1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let v0v2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
    let v0target = [target[0] - v0[0], target[1] - v0[1], target[2] - v0[2]];

    let dot01 = v0v1[0] * v0target[0] + v0v1[1] * v0target[1] + v0v1[2] * v0target[2];
    let dot02 = v0v2[0] * v0target[0] + v0v2[1] * v0target[1] + v0v2[2] * v0target[2];
    let dot12 = v0v1[0] * v0v2[0] + v0v1[1] * v0v2[1] + v0v1[2] * v0v2[2];

    let inv_denom = 1.0 / (dot12 * dot12 - dot01 * dot02);
    let u = (dot12 * dot02 - dot01 * dot12) * inv_denom;
    let v = (dot01 * dot12 - dot02 * dot01) * inv_denom;
    let w = 1.0 - u - v;

    p0.clone() * u + p1.clone() * v + p2.clone() * w
}

fn is_blocked(
    ray_org: &[f32; 3],
    ray_dir: &[f32; 3],
    trimeshs: &[TriangleMesh],
    t_lightsrc: f32,
) -> bool {
    let mut t_min = t_lightsrc;
    for trimesh in trimeshs {
        let Some((t, _, _)) = del_msh_core::trimesh3_search_bruteforce::first_intersection_ray(
            ray_org,
            ray_dir,
            &trimesh.tri2vtx,
            &trimesh.vtx2xyz,
        ) else {
            continue;
        };
        if t < t_min {
            t_min = t;
        }
    }
    (t_min - t_lightsrc).abs() < 1e-6
}

fn main() -> anyhow::Result<()> {
    let pbrt_file_path = "examples/asset/cornell-box/scene-v4.pbrt";
    let (trimeshs, lightsrcs, camera_fov, transform_cam_glbl2lcl, img_shape) =
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
                let Some((t, _, _)) =
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

    let img2 = trace_ray(
        &trimeshs,
        &trimeshs[7],
        camera_fov,
        transform_cam_lcl2glbl,
        img_shape,
    );

    {
        use image::codecs::hdr::HdrEncoder;
        let file = std::fs::File::create("target/02_cornell_box.hdr").unwrap();
        let enc = HdrEncoder::new(file);
        let _ = enc.encode(&img, img_shape.0, img_shape.1);
        let file1 = std::fs::File::create("target/02_cornell_box_color.hdr").unwrap();
        let enc = HdrEncoder::new(file1);
        let _ = enc.encode(&img1, img_shape.0, img_shape.1);

        let file2 = std::fs::File::create("target/02_cornell_box_trace.hdr").unwrap();
        let enc = HdrEncoder::new(file2);
        let _ = enc.encode(&img2, img_shape.0, img_shape.1);
    }
    Ok(())
}
