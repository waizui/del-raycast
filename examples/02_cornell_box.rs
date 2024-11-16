use rand::Rng;

#[derive(Debug, PartialEq, Clone, Default)]
struct TriangleMesh {
    // vtx2uv: Vec<f32>,
    // vtx2nrm: Vec<f32>,
    vtx2xyz: Vec<f32>,
    tri2vtx: Vec<usize>,
    vtx2nrm: Vec<f32>,
    reflectance: [f32; 3],
    spectrum: Option<([f32; 3], bool)>,
}

impl TriangleMesh {
    fn normal_at(&self, pos: &[f32; 3], i_tri: usize) -> [f32; 3] {
        assert!(i_tri < self.tri2vtx.len() / 3);
        let iv0 = self.tri2vtx[i_tri * 3];
        let iv1 = self.tri2vtx[i_tri * 3 + 1];
        let iv2 = self.tri2vtx[i_tri * 3 + 2];
        let p0 = arrayref::array_ref![self.vtx2xyz, iv0 * 3, 3];
        let p1 = arrayref::array_ref![self.vtx2xyz, iv1 * 3, 3];
        let p2 = arrayref::array_ref![self.vtx2xyz, iv2 * 3, 3];
        let bc = del_geo_core::tri3::barycentric_coords(p0, p1, p2, pos);
        let n0 = arrayref::array_ref![self.vtx2nrm, iv0 * 3, 3];
        let n1 = arrayref::array_ref![self.vtx2nrm, iv1 * 3, 3];
        let n2 = arrayref::array_ref![self.vtx2nrm, iv2 * 3, 3];
        let n = [
            bc[0] * n0[0] + bc[1] * n1[0] + bc[2] * n2[0],
            bc[0] * n0[1] + bc[1] * n1[1] + bc[2] * n2[1],
            bc[0] * n0[2] + bc[1] * n1[2] + bc[2] * n2[2],
        ];
        del_geo_core::vec3::normalized(&n)
    }

    fn raw_mesh(_vtx2xyz: Vec<f32>, _tri2vtx: Vec<usize>) -> Self {
        Self {
            vtx2xyz: _vtx2xyz,
            tri2vtx: _tri2vtx,
            vtx2nrm: vec![],
            reflectance: [0.0, 0.0, 0.0],
            spectrum: None,
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
            vtx2nrm: _normal,
            reflectance: _reflectance,
            spectrum: None,
        }
    }
}

fn parse_pbrt_file(
    file_path: &str,
) -> anyhow::Result<(Vec<TriangleMesh>, f32, [f32; 16], (usize, usize))> {
    let scene = pbrt4::Scene::from_file(file_path)?;
    let (camera_fov, transform_cam_glbl2lcl, img_shape) = del_raycast::parse_pbrt::hoge(&scene);
    let mut materials: Vec<[f32; 3]> = vec![];
    for material in scene.materials {
        match material {
            pbrt4::types::Material {
                name,
                attributes,
                reflectance,
            } => {
                materials.push(reflectance.get_rgb());
            }
            _ => {}
        }
    }
    let mut lights: Vec<([f32; 3], bool)> = vec![];
    for area_light in scene.area_lights {
        match area_light.clone() {
            pbrt4::types::AreaLight::Diffuse {
                filename,
                two_sided,
                spectrum,
                scale,
            } => {
                let spectrum =
                    del_raycast::parse_pbrt::spectrum_from_light_entity(&area_light).unwrap();
                lights.push((spectrum, two_sided));
            }
            _ => {}
        }
    }
    let mut shapes: Vec<TriangleMesh> = vec![Default::default(); scene.shapes.len()];
    for (i_shape, shape_entity) in scene.shapes.iter().enumerate() {
        let (material_idx, light_idx, tri2vtx, vtx2xyz, normal) =
            del_raycast::parse_pbrt::trimesh3_from_shape_entity(&shape_entity, file_path).unwrap();
        shapes[i_shape].vtx2xyz = vtx2xyz.clone();
        shapes[i_shape].tri2vtx = tri2vtx.clone();
        shapes[i_shape].vtx2nrm = normal.clone();
        if let Some(light_idx) = light_idx {
            // dbg!(i_shape, light_idx);
            shapes[i_shape].spectrum = Some(lights[light_idx]);
        }
        shapes[i_shape].reflectance = materials[material_idx];
    }
    Ok((shapes, camera_fov, transform_cam_glbl2lcl, img_shape))
}

fn shade(
    wo: nalgebra::Vector3<f32>,
    hit_pos: &[f32; 3],
    hit_trimesh_tri_idx: (usize, usize),
    light_sources: &TriangleMesh,
    meshes: &[TriangleMesh],
) -> [f32; 3] {
    let hit_trimesh_idx = hit_trimesh_tri_idx.0;
    let hit_tri_idx = hit_trimesh_tri_idx.1;
    let target_msh = &meshes[hit_trimesh_idx];
    // normal at triangle vertices
    let i0 = target_msh.tri2vtx[hit_tri_idx * 3];
    let n0 = nalgebra::Vector3::new(
        target_msh.vtx2nrm[i0 * 3],
        target_msh.vtx2nrm[i0 * 3 + 1],
        target_msh.vtx2nrm[i0 * 3 + 2],
    );
    let i1 = target_msh.tri2vtx[hit_tri_idx * 3 + 1];
    let n1 = nalgebra::Vector3::new(
        target_msh.vtx2nrm[i1 * 3],
        target_msh.vtx2nrm[i1 * 3 + 1],
        target_msh.vtx2nrm[i1 * 3 + 2],
    );
    let i2 = target_msh.tri2vtx[hit_tri_idx * 3 + 2];
    let n2 = nalgebra::Vector3::new(
        target_msh.vtx2nrm[i2 * 3],
        target_msh.vtx2nrm[i2 * 3 + 1],
        target_msh.vtx2nrm[i2 * 3 + 2],
    );

    // coord at triangle vertices
    let v0 = [
        target_msh.vtx2xyz[i0 * 3],
        target_msh.vtx2xyz[i0 * 3 + 1],
        target_msh.vtx2xyz[i0 * 3 + 2],
    ];
    let v1 = [
        target_msh.vtx2xyz[i1 * 3],
        target_msh.vtx2xyz[i1 * 3 + 1],
        target_msh.vtx2xyz[i1 * 3 + 2],
    ];
    let v2 = [
        target_msh.vtx2xyz[i2 * 3],
        target_msh.vtx2xyz[i2 * 3 + 1],
        target_msh.vtx2xyz[i2 * 3 + 2],
    ];

    // normal at hit_pos
    let n = interpolate(&n0, &n1, &n2, &v0, &v1, &v2, hit_pos);

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
    let ref_ray_dir = sampled_des - nalgebra::Vector3::new(hit_pos[0], hit_pos[1], hit_pos[2]);
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
    L_o = [L_o[0] + L_dir[0], L_o[1] + L_dir[1], L_o[2] + L_dir[2]];

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
        let Some((t, _i_tri)) = del_msh_core::trimesh3_search_bruteforce::first_intersection_ray(
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

fn intersection_ray_trimeshs(
    ray_org: &[f32; 3],
    ray_dir: &[f32; 3],
    trimeshs: &[TriangleMesh],
) -> Option<(f32, usize, usize)> {
    let mut t_trimsh_tri = Option::<(f32, usize, usize)>::None;
    for (i_trimesh, trimesh) in trimeshs.iter().enumerate() {
        let Some((t, i_tri)) = del_msh_core::trimesh3_search_bruteforce::first_intersection_ray(
            &ray_org,
            &ray_dir,
            &trimesh.tri2vtx,
            &trimesh.vtx2xyz,
        ) else {
            continue;
        };
        match t_trimsh_tri {
            None => {
                if t > 0. {
                    t_trimsh_tri = Some((t, i_trimesh, i_tri));
                }
            }
            Some((t_min, _, _)) => {
                if t < t_min && t > 0. {
                    t_trimsh_tri = Some((t, i_trimesh, i_tri));
                }
            }
        }
    }
    t_trimsh_tri
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

    {
        let mut img = vec![image::Rgb([0f32; 3]); img_shape.0 * img_shape.1];
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
                    let Some((t, i_tri)) =
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
            }
        }
        use image::codecs::hdr::HdrEncoder;
        let file = std::fs::File::create("target/02_cornell_box_depth.hdr").unwrap();
        let enc = HdrEncoder::new(file);
        let _ = enc.encode(&img, img_shape.0, img_shape.1);
    }
    {
        let mut img = vec![image::Rgb([0f32; 3]); img_shape.0 * img_shape.1];
        for iw in 0..img_shape.0 {
            for ih in 0..img_shape.1 {
                let (ray_org, ray_dir) = del_raycast::cam_pbrt::cast_ray(
                    iw,
                    ih,
                    img_shape,
                    camera_fov,
                    transform_cam_lcl2glbl,
                );
                let Some((_t, i_trimsh, _i_tri)) =
                    intersection_ray_trimeshs(&ray_org, &ray_dir, &trimeshs)
                else {
                    continue;
                };
                img[ih * img_shape.0 + iw] = image::Rgb(trimeshs[i_trimsh].reflectance);
            }
        }
        let file1 = std::fs::File::create("target/02_cornell_box_color.hdr").unwrap();
        use image::codecs::hdr::HdrEncoder;
        let enc = HdrEncoder::new(file1);
        let _ = enc.encode(&img, img_shape.0, img_shape.1);
    }

    {
        // path tracing
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);
        let mut img = vec![image::Rgb([0f32; 3]); img_shape.0 * img_shape.1];
        img.resize(img_shape.0 * img_shape.1, image::Rgb([0_f32; 3]));
        for iw in 0..img_shape.0 {
            for ih in 0..img_shape.1 {
                let (ray0_org, ray0_dir) = del_raycast::cam_pbrt::cast_ray(
                    iw,
                    ih,
                    img_shape,
                    camera_fov,
                    transform_cam_lcl2glbl,
                );
                let Some((t1, i_trimsh1, i_tri1)) =
                    intersection_ray_trimeshs(&ray0_org, &ray0_dir, &trimeshs)
                else {
                    continue;
                };
                let pos1 = del_geo_core::vec3::axpy(t1, &ray0_dir, &ray0_org);
                let nrm1 = trimeshs[i_trimsh1].normal_at(&pos1, i_tri1);
                if let Some((spectrum1, two_sided1)) = trimeshs[i_trimsh1].spectrum {
                    if two_sided1 || (del_geo_core::vec3::dot(&nrm1, &ray0_dir) < 0.) {
                        img[ih * img_shape.0 + iw].0[0] += spectrum1[0];
                        img[ih * img_shape.0 + iw].0[1] += spectrum1[1];
                        img[ih * img_shape.0 + iw].0[2] += spectrum1[2];
                        continue;
                    }
                }
                let nrm1 = if del_geo_core::vec3::dot(&nrm1, &ray0_dir) > 0. {
                    [-nrm1[0], -nrm1[1], -nrm1[2]]
                } else {
                    nrm1
                };
                let ray1_dir: [f32; 3] = del_raycast::sampling::hemisphere_cos_weighted(
                    &nalgebra::Vector3::<f32>::new(nrm1[0], nrm1[1], nrm1[2]),
                    &[rng.gen::<f32>(), rng.gen::<f32>()],
                )
                .into();
                let ray1_org = del_geo_core::vec3::axpy(1.0e-3, &nrm1, &pos1);
                let Some((t2, i_trimsh2, i_tri2)) =
                    intersection_ray_trimeshs(&ray1_org, &ray1_dir, &trimeshs)
                else {
                    continue;
                };
                let Some((spectrum2, two_sided2)) = trimeshs[i_trimsh2].spectrum else {
                    continue;
                };
                let pos2 = del_geo_core::vec3::axpy(t2, &ray1_dir, &ray1_org);
                let nrm2 = trimeshs[i_trimsh2].normal_at(&pos2, i_tri2);
                if !two_sided2 && (del_geo_core::vec3::dot(&nrm2, &ray1_dir) > 0.) {
                    continue;
                }
                img[ih * img_shape.0 + iw].0[0] +=
                    spectrum2[0] * trimeshs[i_trimsh1].reflectance[0];
                img[ih * img_shape.0 + iw].0[1] +=
                    spectrum2[1] * trimeshs[i_trimsh1].reflectance[1];
                img[ih * img_shape.0 + iw].0[2] +=
                    spectrum2[2] * trimeshs[i_trimsh1].reflectance[2];
                /*
                                // compute intersection below
                                let mut t_min = f32::INFINITY;
                                let mut Lo = [0.0, 0.0, 0.0];
                                let mut hit_pos = [0.0, 0.0, 0.0];
                                let mut hit_trimesh_tri_idx = Option::<(usize, usize)>::None;
                                for (i_trimesh, trimesh) in trimeshs.iter().enumerate() {
                                    let Some((t, i_tri)) =
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
                                        hit_trimesh_tri_idx = Some((i_trimesh, i_tri));
                                    }
                                }
                                let Some(hit_trimesh_tri_idx) = hit_trimesh_tri_idx else {
                                    continue;
                                };
                                // Ray Tracing
                                let wo = nalgebra::Vector3::new(-ray_dir[0], -ray_dir[1], -ray_dir[2]);
                                Lo = shade(wo, &hit_pos, hit_trimesh_tri_idx, &trimeshs[7], &trimeshs);
                */
                // store Lo to img
                /*
                img[ih * img_shape.0 + iw] =
                    image::Rgb([nrm[0] * 0.5 + 0.5, nrm[1] * 0.5 + 0.5, nrm[2] * 0.5 + 0.5]);
                 */
            }
        }
        let file2 = std::fs::File::create("target/02_cornell_box_trace.hdr").unwrap();
        use image::codecs::hdr::HdrEncoder;
        let enc = HdrEncoder::new(file2);
        let _ = enc.encode(&img, img_shape.0, img_shape.1);
    }
    Ok(())
}
