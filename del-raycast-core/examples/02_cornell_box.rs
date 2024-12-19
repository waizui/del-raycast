#[derive(Debug, Clone, Default)]
struct TriangleMesh {
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
        let bc = del_geo_core::tri3::to_barycentric_coords(p0, p1, p2, pos);
        let n0 = arrayref::array_ref![self.vtx2nrm, iv0 * 3, 3];
        let n1 = arrayref::array_ref![self.vtx2nrm, iv1 * 3, 3];
        let n2 = arrayref::array_ref![self.vtx2nrm, iv2 * 3, 3];
        let n = [
            bc[0] * n0[0] + bc[1] * n1[0] + bc[2] * n2[0],
            bc[0] * n0[1] + bc[1] * n1[1] + bc[2] * n2[1],
            bc[0] * n0[2] + bc[1] * n1[2] + bc[2] * n2[2],
        ];
        del_geo_core::vec3::normalize(&n)
    }
}

fn parse_pbrt_file(
    file_path: &str,
) -> anyhow::Result<(Vec<TriangleMesh>, f32, [f32; 16], (usize, usize))> {
    let scene = pbrt4::Scene::from_file(file_path)?;
    let (camera_fov, transform_cam_glbl2lcl, img_shape) =
        del_raycast_core::parse_pbrt::hoge(&scene);
    let mut materials: Vec<[f32; 3]> = vec![];
    for material in scene.materials {
        match material {
            pbrt4::types::Material {
                name,
                attributes,
                reflectance,
                ..
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
                    del_raycast_core::parse_pbrt::spectrum_from_light_entity(&area_light).unwrap();
                lights.push((spectrum, two_sided));
            }
            _ => {}
        }
    }
    let mut shapes: Vec<TriangleMesh> = vec![Default::default(); scene.shapes.len()];
    for (i_shape, shape_entity) in scene.shapes.iter().enumerate() {
        let (material_idx, light_idx, tri2vtx, vtx2xyz, normal) =
            del_raycast_core::parse_pbrt::trimesh3_from_shape_entity(&shape_entity, file_path)
                .unwrap();
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

fn intersection_ray_against_trimeshs(
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

fn hit_position_normal_emission_at_ray_intersection(
    ray_org: &[f32; 3],
    ray_dir: &[f32; 3],
    trimeshs: &[TriangleMesh],
) -> Option<([f32; 3], [f32; 3], [f32; 3], usize)> {
    let Some((ray_t, i_trimsh_hit, i_tri_hit)) =
        intersection_ray_against_trimeshs(&ray_org, &ray_dir, &trimeshs)
    else {
        return None;
    };
    let hit_pos = vec3::axpy(ray_t, &ray_dir, &ray_org);
    let hit_nrm = trimeshs[i_trimsh_hit].normal_at(&hit_pos, i_tri_hit);
    use del_geo_core::vec3;
    let mut hit_emission = [0f32; 3];
    if let Some((spectrum, is_both_side)) = trimeshs[i_trimsh_hit].spectrum {
        // hit light
        if is_both_side || del_geo_core::vec3::dot(&hit_nrm, ray_dir) < 0.0 {
            hit_emission = spectrum;
        }
    }
    let hit_nrm = if vec3::dot(&hit_nrm, &ray_dir) > 0. {
        [-hit_nrm[0], -hit_nrm[1], -hit_nrm[2]]
    } else {
        hit_nrm
    };
    Some((hit_pos, hit_nrm, hit_emission, i_trimsh_hit))
}

fn sampling_light(
    tri2cumsumarea: &[f32],
    r2: [f32; 2],
    triangle_mesh: &TriangleMesh,
) -> ([f32; 3], [f32; 3]) {
    let (i_tri_light, r1, r2) =
        del_msh_core::sampling::sample_uniformly_trimesh(&tri2cumsumarea, r2[0], r2[1]);
    let (p0, p1, p2) = del_msh_core::trimesh3::to_corner_points(
        &triangle_mesh.tri2vtx,
        &triangle_mesh.vtx2xyz,
        i_tri_light,
    );
    let light_pos = del_geo_core::tri3::position_from_barycentric_coords(
        &p0,
        &p1,
        &p2,
        &[1. - r1 - r2, r1, r2],
    );
    let light_nrm = triangle_mesh.normal_at(&light_pos, i_tri_light);
    let light_pos = del_geo_core::vec3::axpy(1.0e-3, &light_nrm, &light_pos);
    (light_pos, light_nrm)
}

fn radiance_from_light<RNG>(
    hit_pos: &[f32; 3],
    trimeshs: &[TriangleMesh],
    rng: &mut RNG,
    tri2cumsumarea: &[f32],
    i_trimesh_light: usize,
) -> Option<([f32; 3], f32, [f32; 3])>
where
    RNG: rand::Rng,
{
    use del_geo_core::vec3;
    let &area_light = tri2cumsumarea.last().unwrap();
    let (light_pos, light_nrm) = sampling_light(
        &tri2cumsumarea,
        [rng.gen(), rng.gen()],
        &trimeshs[i_trimesh_light],
    );
    //
    let uvec_hit2light = vec3::normalize(&vec3::sub(&light_pos, &hit_pos));
    let cos_theta_light = -vec3::dot(&light_nrm, &uvec_hit2light);
    if cos_theta_light < 0. {
        return None;
    } // backside of light
    if let Some((_t, i_trimsh, _i_tri)) =
        intersection_ray_against_trimeshs(&hit_pos, &uvec_hit2light, &trimeshs)
    {
        if i_trimsh != i_trimesh_light {
            return None;
        }
    } else {
        return None;
    };
    let l_i = trimeshs[i_trimesh_light].spectrum.unwrap().0;
    let pdf = 1.0 / area_light;
    let r2 = del_geo_core::edge3::squared_length(&light_pos, &hit_pos);
    let geo_term = cos_theta_light / r2;
    // dbg!(light_pos.sub(&hit_pos));
    use del_geo_core::vec3::Vec3;
    Some((l_i, pdf / geo_term, uvec_hit2light))
}

struct MyScene<'a> {
    trimesh: &'a [TriangleMesh],
    tri2cumsumarea: &'a [f32],
    i_trimesh_light: usize,
}

impl del_raycast_core::monte_carlo_integrator::Scene for MyScene<'_> {
    fn hit_position_normal_emission_at_ray_intersection(
        &self,
        ray_org: &[f32; 3],
        ray_dir: &[f32; 3],
    ) -> Option<([f32; 3], [f32; 3], [f32; 3], usize)> {
        hit_position_normal_emission_at_ray_intersection(ray_org, ray_dir, self.trimesh)
    }
    fn pdf_light(
        &self,
        hit_pos: &[f32; 3],
        hit_pos_light: &[f32; 3],
        hit_nrm_light: &[f32; 3],
    ) -> f32 {
        use del_geo_core::vec3;
        let vec_obj2light = vec3::sub(hit_pos_light, hit_pos);
        let uvec_obj2light = vec3::normalize(&vec_obj2light);
        let distance = del_geo_core::edge3::length(&hit_pos, &hit_pos_light);
        let geo_term = -del_geo_core::vec3::dot(&uvec_obj2light, &hit_nrm_light) / distance.powi(2);
        1.0 / self.tri2cumsumarea.last().unwrap() / geo_term
    }

    fn sample_brdf<RNG>(
        &self,
        hit_nrm: [f32; 3],
        itrimsh: usize,
        rng: &mut RNG,
    ) -> ([f32; 3], [f32; 3], f32)
    where
        RNG: rand::Rng,
    {
        let ray_dir_next: [f32; 3] = del_raycast_core::sampling::hemisphere_cos_weighted(
            &nalgebra::Vector3::<f32>::from(hit_nrm),
            &[rng.gen::<f32>(), rng.gen::<f32>()],
        )
        .into();
        use del_geo_core::vec3::Vec3;
        let brdf = self.trimesh[itrimsh]
            .reflectance
            .scale(std::f32::consts::FRAC_1_PI);
        let cos_hit = ray_dir_next.dot(&hit_nrm).clamp(f32::EPSILON, 1f32);
        let pdf = cos_hit * std::f32::consts::FRAC_1_PI;
        (ray_dir_next, brdf, pdf)
    }
    fn brdf(&self, itrimsh: usize) -> [f32; 3] {
        use del_geo_core::vec3::Vec3;
        let reflectance1 = self.trimesh[itrimsh].reflectance;
        reflectance1.scale(1. / std::f32::consts::PI)
    }

    fn radiance_from_light<Rng: rand::Rng>(
        &self,
        hit_pos: &[f32; 3],
        rng: &mut Rng,
    ) -> Option<([f32; 3], f32, [f32; 3])> {
        radiance_from_light(
            hit_pos,
            self.trimesh,
            rng,
            self.tri2cumsumarea,
            self.i_trimesh_light,
        )
    }
}

fn main() -> anyhow::Result<()> {
    let pbrt_file_path = "asset/cornell-box/scene-v4.pbrt";
    let (trimeshs, camera_fov, transform_cam_glbl2lcl, img_shape) =
        parse_pbrt_file(pbrt_file_path)?;
    {
        // make obj file
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
    use itertools::Itertools;
    // Get the area light source
    let i_trimesh_light = trimeshs
        .iter()
        .enumerate()
        .find_or_first(|(_i_trimesh, trimsh)| trimsh.spectrum.is_some())
        .unwrap()
        .0;
    let tri2cumsumarea = del_msh_core::sampling::cumulative_area_sum(
        &trimeshs[i_trimesh_light].tri2vtx,
        &trimeshs[i_trimesh_light].vtx2xyz,
        3,
    );
    let area_light = tri2cumsumarea.last().unwrap().clone();
    dbg!(area_light);
    let img_gt = image::open("asset/cornell-box/TungstenRender.exr")
        .unwrap()
        .to_rgb32f();
    assert!(img_gt.dimensions() == (img_shape.0 as u32, img_shape.1 as u32));
    let img_gt = img_gt.to_vec();
    {
        // computing depth image
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = arrayref::array_mut_ref![pix, 0, 3];
            let (ray_org, ray_dir) = del_raycast_core::cam_pbrt::cast_ray(
                (i_pix % img_shape.0, i_pix / img_shape.0),
                (0., 0.),
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
            *pix = [v; 3];
        };
        let mut img_out = vec![0f32; img_shape.0 * img_shape.1 * 3];
        use rayon::prelude::*;
        img_out
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        del_canvas::write_hdr_file("target/02_cornell_box_depth.hdr", img_shape, &img_out)?;
    }
    {
        // computing reflectance image
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = arrayref::array_mut_ref![pix, 0, 3];
            let (ray_org, ray_dir) = del_raycast_core::cam_pbrt::cast_ray(
                (i_pix % img_shape.0, i_pix / img_shape.0),
                (0., 0.),
                img_shape,
                camera_fov,
                transform_cam_lcl2glbl,
            );
            let Some((_t, i_trimsh, _i_tri)) =
                intersection_ray_against_trimeshs(&ray_org, &ray_dir, &trimeshs)
            else {
                return;
            };
            *pix = trimeshs[i_trimsh].reflectance;
        };
        let mut img_out = vec![0f32; img_shape.0 * img_shape.1 * 3];
        use rayon::prelude::*;
        img_out
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        del_canvas::write_hdr_file("target/02_cornell_box_color.hdr", img_shape, &img_out)?;
    }
    println!("---------------------path tracer---------------------");
    for i in 1..4 {
        let scene = MyScene {
            trimesh: &trimeshs,
            tri2cumsumarea: &tri2cumsumarea,
            i_trimesh_light,
        };
        // path tracing sampling material
        let num_sample = 8 * i;
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = arrayref::array_mut_ref![pix, 0, 3];
            use rand::Rng;
            use rand::SeedableRng;
            let mut rng = rand_chacha::ChaChaRng::seed_from_u64(i_pix as u64);
            let mut l_o = [0., 0., 0.];
            for _i_sample in 0..num_sample {
                let (ray0_org, ray0_dir) = del_raycast_core::cam_pbrt::cast_ray(
                    (i_pix % img_shape.0, i_pix / img_shape.0),
                    (
                        del_raycast_core::sampling::tent(rng.gen::<f32>()),
                        del_raycast_core::sampling::tent(rng.gen::<f32>()),
                    ),
                    img_shape,
                    camera_fov,
                    transform_cam_lcl2glbl,
                );
                let rad = del_raycast_core::monte_carlo_integrator::radiance_pt(
                    &ray0_org, &ray0_dir, &scene, 65, &mut rng,
                );
                l_o = del_geo_core::vec3::add(&l_o, &rad);
            }
            *pix = del_geo_core::vec3::scale(&l_o, 1. / num_sample as f32);
        };
        let img_out = {
            let mut img_out = vec![0f32; img_shape.0 * img_shape.1 * 3];
            use rayon::prelude::*;
            img_out
                .par_chunks_mut(3)
                .enumerate()
                .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
            img_out
        };
        del_canvas::write_hdr_file(
            format!("target/02_cornell_box_pt_{}.hdr", num_sample),
            img_shape,
            &img_out,
        )?;
        let path_error_map = format!("target/02_cornell_box_pt_{}_error_map.hdr", num_sample);
        del_canvas::write_hdr_file_mse_rgb_error_map(path_error_map, img_shape, &img_gt, &img_out);
        let err = del_canvas::rmse_error(&img_gt, &img_out);
        println!("num_sample: {}, mse: {}", num_sample, err);
    }
    println!("---------------------MIS sampling---------------------");
    for i in 1..4 {
        let scene = MyScene {
            trimesh: &trimeshs,
            tri2cumsumarea: &tri2cumsumarea,
            i_trimesh_light,
        };
        let num_sample = 8 * i;
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = arrayref::array_mut_ref!(pix, 0, 3);
            use rand::Rng;
            use rand::SeedableRng;
            let mut rng = rand_chacha::ChaChaRng::seed_from_u64(i_pix as u64);
            let mut l_o = [0., 0., 0.];
            for _i_sample in 0..num_sample {
                let (ray0_org, ray0_dir) = del_raycast_core::cam_pbrt::cast_ray(
                    (i_pix % img_shape.0, i_pix / img_shape.0),
                    (
                        del_raycast_core::sampling::tent(rng.gen::<f32>()),
                        del_raycast_core::sampling::tent(rng.gen::<f32>()),
                    ),
                    img_shape,
                    camera_fov,
                    transform_cam_lcl2glbl,
                );
                let rad = del_raycast_core::monte_carlo_integrator::radiance_mis(
                    &ray0_org, &ray0_dir, &scene, &mut rng,
                );
                l_o = del_geo_core::vec3::add(&l_o, &rad);
            }
            *pix = del_geo_core::vec3::scale(&l_o, 1. / num_sample as f32);
        };
        let img_out = {
            let mut img_out = vec![0f32; img_shape.0 * img_shape.1 * 3];
            use rayon::prelude::*;
            img_out
                .par_chunks_mut(3)
                .enumerate()
                .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
            img_out
        };
        del_canvas::write_hdr_file(
            format!("target/02_cornell_box_mis_{}.hdr", num_sample),
            img_shape,
            &img_out,
        )?;
        let path_error_map = format!("target/02_cornell_box_mis_{}_error_map.hdr", num_sample);
        del_canvas::write_hdr_file_mse_rgb_error_map(path_error_map, img_shape, &img_gt, &img_out);
        let err = del_canvas::rmse_error(&img_gt, &img_out);
        println!("num_sample: {}, mse: {}", num_sample, err);
    }
    println!("---------------------NEE tracer---------------------");
    for i in 1..4 {
        let scene = MyScene {
            trimesh: &trimeshs,
            tri2cumsumarea: &tri2cumsumarea,
            i_trimesh_light,
        };
        // path tracing next event estimation
        let num_sample = 8 * i;
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = arrayref::array_mut_ref!(pix, 0, 3);
            use rand::Rng;
            use rand::SeedableRng;
            let mut rng = rand_chacha::ChaChaRng::seed_from_u64(i_pix as u64);
            let mut l_o = [0., 0., 0.];
            for _i_sample in 0..num_sample {
                let (ray0_org, ray0_dir) = del_raycast_core::cam_pbrt::cast_ray(
                    (i_pix % img_shape.0, i_pix / img_shape.0),
                    (
                        del_raycast_core::sampling::tent(rng.gen::<f32>()),
                        del_raycast_core::sampling::tent(rng.gen::<f32>()),
                    ),
                    img_shape,
                    camera_fov,
                    transform_cam_lcl2glbl,
                );
                let rad = del_raycast_core::monte_carlo_integrator::radiance_nee(
                    &ray0_org, &ray0_dir, &scene, &mut rng,
                );
                l_o = del_geo_core::vec3::add(&l_o, &rad);
            }
            *pix = del_geo_core::vec3::scale(&l_o, 1. / num_sample as f32);
        };
        let img_out = {
            let mut img_out = vec![0f32; img_shape.0 * img_shape.1 * 3];
            use rayon::prelude::*;
            img_out
                .par_chunks_mut(3)
                .enumerate()
                .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
            img_out
        };
        del_canvas::write_hdr_file(
            format!("target/02_cornell_box_nee_{}.hdr", num_sample),
            img_shape,
            &img_out,
        )?;
        let path_error_map = format!("target/02_cornell_box_nee_{}_error_map.hdr", num_sample);
        del_canvas::write_hdr_file_mse_rgb_error_map(path_error_map, img_shape, &img_gt, &img_out);
        let err = del_canvas::rmse_error(&img_gt, &img_out);
        println!("num_sample: {}, mse: {}", num_sample, err);
    }
    println!("---------------------light sampling---------------------");
    for i in 1..4 {
        use del_geo_core::vec3::Vec3;
        // light sampling
        let num_sample = 8 * i;
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = arrayref::array_mut_ref![pix, 0, 3];
            use rand::Rng;
            use rand::SeedableRng;
            let mut rng = rand_chacha::ChaChaRng::seed_from_u64(i_pix as u64);
            let mut l_o = [0., 0., 0.];
            for _i_sample in 0..num_sample {
                let (ray_org, ray_dir) = del_raycast_core::cam_pbrt::cast_ray(
                    (i_pix % img_shape.0, i_pix / img_shape.0),
                    (
                        del_raycast_core::sampling::tent(rng.gen::<f32>()),
                        del_raycast_core::sampling::tent(rng.gen::<f32>()),
                    ),
                    img_shape,
                    camera_fov,
                    transform_cam_lcl2glbl,
                );
                let Some((hit_pos, hit_nrm, hit_emission, hit_itrimsh)) =
                    hit_position_normal_emission_at_ray_intersection(&ray_org, &ray_dir, &trimeshs)
                else {
                    continue;
                };
                l_o = vec3::add(&l_o, &hit_emission);
                use del_geo_core::vec3;
                let hit_pos = vec3::axpy(1.0e-3, &hit_nrm, &hit_pos);
                if let Some((li, pdf_light, uvec_hit2light)) = radiance_from_light(
                    &hit_pos,
                    &trimeshs,
                    &mut rng,
                    &tri2cumsumarea,
                    i_trimesh_light,
                ) {
                    let brdf_hit = trimeshs[hit_itrimsh]
                        .reflectance
                        .scale(1. / std::f32::consts::PI);
                    let cos_hit = vec3::dot(&uvec_hit2light, &hit_nrm);
                    let li_r = vec3::element_wise_mult(&li, &brdf_hit.scale(cos_hit / pdf_light));
                    l_o = vec3::add(&l_o, &li_r);
                }
            }
            *pix = l_o.scale(1.0 / num_sample as f32);
        };
        let img_out = {
            let mut img_out = vec![0f32; img_shape.0 * img_shape.1 * 3];
            use rayon::prelude::*;
            img_out
                .par_chunks_mut(3)
                .enumerate()
                .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
            img_out
        };
        del_canvas::write_hdr_file(
            format!("target/02_cornell_box_light_sampling_{}.hdr", num_sample),
            img_shape,
            &img_out,
        )?;
        let path_error_map = format!(
            "target/02_cornell_box_light_sampling_{}_error_map.hdr",
            num_sample
        );
        del_canvas::write_hdr_file_mse_rgb_error_map(path_error_map, img_shape, &img_gt, &img_out);
        let err = del_canvas::rmse_error(&img_gt, &img_out);
        println!("num_sample: {}, mse: {}", num_sample, err);
    }
    println!("---------------------material sampling---------------------");
    for i in 1..4 {
        // material sampling
        let num_sample = 8 * i;
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = arrayref::array_mut_ref![pix, 0, 3];
            use del_geo_core::vec3;
            use rand::Rng;
            use rand::SeedableRng;
            let mut rng = rand_chacha::ChaChaRng::seed_from_u64(i_pix as u64);
            let mut l_o = [0., 0., 0.];
            for _i_sample in 0..num_sample {
                let (ray0_org, ray0_dir) = del_raycast_core::cam_pbrt::cast_ray(
                    (i_pix % img_shape.0, i_pix / img_shape.0),
                    (
                        del_raycast_core::sampling::tent(rng.gen::<f32>()),
                        del_raycast_core::sampling::tent(rng.gen::<f32>()),
                    ),
                    img_shape,
                    camera_fov,
                    transform_cam_lcl2glbl,
                );
                let Some((hit_pos, hit_nrm, hit_emission, hit_itrimsh)) =
                    hit_position_normal_emission_at_ray_intersection(
                        &ray0_org, &ray0_dir, &trimeshs,
                    )
                else {
                    continue;
                };
                l_o = vec3::add(&l_o, &hit_emission);
                let ray_dir_next: [f32; 3] = del_raycast_core::sampling::hemisphere_cos_weighted(
                    &nalgebra::Vector3::<f32>::from(hit_nrm),
                    &[rng.gen::<f32>(), rng.gen::<f32>()],
                )
                .into();
                let hit_pos_w_offset = vec3::axpy(1.0e-3, &hit_nrm, &hit_pos);
                let Some((hit2_pos, hit2_nrm, hit2_emission, hit2_itrimsh)) =
                    hit_position_normal_emission_at_ray_intersection(
                        &hit_pos_w_offset,
                        &ray_dir_next,
                        &trimeshs,
                    )
                else {
                    continue;
                };
                use del_geo_core::vec3::Vec3;
                let brdf = trimeshs[hit_itrimsh]
                    .reflectance
                    .scale(std::f32::consts::FRAC_1_PI);
                let cos_hit = ray_dir_next.dot(&hit_nrm).clamp(f32::EPSILON, 1f32);
                let pdf = cos_hit * std::f32::consts::FRAC_1_PI;
                l_o = vec3::add(
                    &l_o,
                    &vec3::element_wise_mult(&hit2_emission, &brdf.scale(cos_hit / pdf)),
                );
                /*
                l_o = vec3::add(
                    &l_o,
                    &vec3::element_wise_mult(&hit2_emission, &trimeshs[hit_itrimsh].reflectance));
                 */
            }
            *pix = vec3::scale(&l_o, 1.0 / num_sample as f32);
        };
        let img_out = {
            let mut img_out = vec![0f32; img_shape.0 * img_shape.1 * 3];
            use rayon::prelude::*;
            img_out
                .par_chunks_mut(3)
                .enumerate()
                .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
            img_out
        };
        del_canvas::write_hdr_file(
            format!("target/02_cornell_box_material_sampling_{}.hdr", num_sample),
            img_shape,
            &img_out,
        )?;
        let path_error_map = format!(
            "target/02_cornell_box_material_sampling_{}_error_map.hdr",
            num_sample
        );
        del_canvas::write_hdr_file_mse_rgb_error_map(path_error_map, img_shape, &img_gt, &img_out);
        let err = del_canvas::rmse_error(&img_gt, &img_out);
        println!("num_sample: {}, mse: {}", num_sample, err);
    }
    Ok(())
}
