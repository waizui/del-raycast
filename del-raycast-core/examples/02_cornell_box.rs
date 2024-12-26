use del_geo_core::vec3;
use del_raycast_core::area_light::AreaLight;
use del_raycast_core::shape::ShapeEntity;

struct MyScene {
    shape_entities: Vec<ShapeEntity>,
    materials: Vec<del_raycast_core::material::Material>,
    area_lights: Vec<AreaLight>,
    i_shape_entity_light: usize,
}

fn parse_pbrt_file(
    file_path: &str,
) -> anyhow::Result<(MyScene, del_raycast_core::parse_pbrt::Camera)> {
    let scene = pbrt4::Scene::from_file(file_path)?;
    let camera = del_raycast_core::parse_pbrt::camera(&scene);
    let materials = del_raycast_core::parse_pbrt::parse_material(&scene);
    let area_lights = del_raycast_core::parse_pbrt::parse_area_light(&scene);
    let shape_entities = del_raycast_core::parse_pbrt::parse_shapes(&scene);
    //
    use itertools::Itertools;
    // Get the area light source
    let i_trimesh_light = shape_entities
        .iter()
        .enumerate()
        .find_or_first(|(_i_trimesh, trimsh)| trimsh.area_light_index.is_some())
        .unwrap()
        .0;
    let tri2cumsumarea = {
        match &shape_entities[i_trimesh_light].shape {
            del_raycast_core::shape::ShapeType::TriangleMesh {
                tri2vtx, vtx2xyz, ..
            } => del_msh_core::sampling::cumulative_area_sum(tri2vtx, vtx2xyz, 3),
            _ => {
                todo!();
            }
        }
    };
    let scene = MyScene {
        shape_entities,
        area_lights,
        materials,
        i_shape_entity_light: i_trimesh_light,
    };
    Ok((scene, camera))
}

impl del_raycast_core::monte_carlo_integrator::Scene for MyScene {
    /// * Return
    /// position, normal, emission, i_shape_entity
    fn hit_position_normal_emission_at_ray_intersection(
        &self,
        ray_org: &[f32; 3],
        ray_dir: &[f32; 3],
    ) -> Option<([f32; 3], [f32; 3], [f32; 3], usize)> {
        let Some((t, i_shape_entity, i_elem)) =
            del_raycast_core::shape::intersection_ray_against_shape_entities(
                ray_org,
                ray_dir,
                &self.shape_entities,
            )
        else {
            return None;
        };
        let hit_pos = vec3::axpy(t, &ray_dir, &ray_org);
        let hit_nrm = del_raycast_core::shape::normal_at(
            &self.shape_entities[i_shape_entity],
            &hit_pos,
            i_elem,
        );
        use del_geo_core::vec3;
        let mut hit_emission = [0f32; 3];
        if let Some(i_area_light) = self.shape_entities[i_shape_entity].area_light_index {
            let al = &self.area_lights[i_area_light];
            if let Some(spectrum) = al.spectrum_rgb {
                // hit light
                if al.two_sided || del_geo_core::vec3::dot(&hit_nrm, ray_dir) < 0.0 {
                    hit_emission = spectrum;
                }
            }
        }
        let hit_nrm = if vec3::dot(&hit_nrm, &ray_dir) > 0. {
            [-hit_nrm[0], -hit_nrm[1], -hit_nrm[2]]
        } else {
            hit_nrm
        };
        Some((hit_pos, hit_nrm, hit_emission, i_shape_entity))
    }
    fn pdf_light(
        &self,
        hit_pos: &[f32; 3],
        hit_pos_light: &[f32; 3],
        hit_nrm_light: &[f32; 3],
        i_shape_entity: usize,
    ) -> f32 {
        use del_geo_core::vec3;
        let vec_obj2light = vec3::sub(hit_pos_light, hit_pos);
        let uvec_obj2light = vec3::normalize(&vec_obj2light);
        let distance = del_geo_core::edge3::length(&hit_pos, &hit_pos_light);
        let geo_term = -del_geo_core::vec3::dot(&uvec_obj2light, &hit_nrm_light) / distance.powi(2);
        let se = &self.shape_entities[self.i_shape_entity_light];
        let area = match &se.shape {
            del_raycast_core::shape::ShapeType::TriangleMesh { tri2cumsumarea, .. } => {
                tri2cumsumarea.as_ref().unwrap().last().unwrap()
            }
            _ => {
                todo!()
            }
        };
        1.0 / area / geo_term
    }

    fn sample_brdf<RNG>(
        &self,
        obj_nrm: &[f32; 3],
        uvec_ray_in_outward: &[f32; 3],
        i_shape_entity: usize,
        rng: &mut RNG,
    ) -> Option<([f32; 3], [f32; 3], f32)>
    where
        RNG: rand::Rng,
    {
        let se = &self.shape_entities[i_shape_entity];
        let i_material = se.material_index.unwrap();
        assert!(i_material < self.materials.len());
        del_raycast_core::material::sample_brdf(
            &self.materials[i_material],
            obj_nrm,
            uvec_ray_in_outward,
            rng,
        )
    }

    fn eval_brdf(
        &self,
        i_shape_entity: usize,
        obj_nrm: &[f32; 3],
        ray_in_outward_normalized: &[f32; 3],
        ray_out_normalized: &[f32; 3],
    ) -> [f32; 3] {
        assert!(
            (vec3::norm(ray_in_outward_normalized) - 1f32).abs() < 1.0e-5,
            "{}",
            vec3::norm(ray_in_outward_normalized)
        );
        let i_material = self.shape_entities[i_shape_entity].material_index.unwrap();
        del_raycast_core::material::eval_brdf(
            &self.materials[i_material],
            obj_nrm,
            ray_in_outward_normalized,
            ray_out_normalized,
        )
    }

    /// # Return
    /// - `Some(radiance: [f32;3], pdf: f32, uvec_hit2light:[f32;3])`
    ///    - `pdf: f32` the pdf is computed on the unit hemisphere (pdf of light / geometric term)
    /// - `None`
    fn radiance_from_light<Rng: rand::Rng>(
        &self,
        hit_pos: &[f32; 3],
        rng: &mut Rng,
    ) -> Option<([f32; 3], f32, [f32; 3])> {
        use del_geo_core::vec3;
        let (light_pos, light_nrm, pdf) =
            self.shape_entities[self.i_shape_entity_light].sample_uniform(&[rng.gen(), rng.gen()]);
        let uvec_hit2light = vec3::normalize(&vec3::sub(&light_pos, &hit_pos));
        let cos_theta_light = -vec3::dot(&light_nrm, &uvec_hit2light);
        if cos_theta_light < 0. {
            return None;
        } // backside of light
        if let Some((t, i_shape_entity, _i_tri)) =
            del_raycast_core::shape::intersection_ray_against_shape_entities(
                &hit_pos,
                &uvec_hit2light,
                &self.shape_entities,
            )
        {
            if i_shape_entity != self.i_shape_entity_light {
                return None;
            }
            let light_pos2 = vec3::axpy(t, &uvec_hit2light, hit_pos);
            if del_geo_core::edge3::length(&light_pos, &light_pos2) > 1.0e-3 {
                return None;
            }
        } else {
            return None;
        };
        let i_area_light = self.shape_entities[self.i_shape_entity_light]
            .area_light_index
            .unwrap();
        let l_i = self.area_lights[i_area_light].spectrum_rgb.unwrap();
        let r2 = del_geo_core::edge3::squared_length(&light_pos, &hit_pos);
        let geo_term = cos_theta_light / r2;
        Some((l_i, pdf / geo_term, uvec_hit2light))
    }
}

fn main() -> anyhow::Result<()> {
    let pbrt_file_path = "asset/cornell-box/scene-v4.pbrt";
    let (scene, camera) = parse_pbrt_file(pbrt_file_path)?;
    del_raycast_core::shape::write_wavefront_obj_file_from_camera_view(
        "target/02_cornell_box.obj",
        &scene.shape_entities,
        &camera.transform_world2camlcl,
    )?;
    // let area_light = scene.tri2cumsumarea.last().unwrap().clone();
    // dbg!(area_light);
    let img_gt = image::open("asset/cornell-box/TungstenRender.exr")
        .unwrap()
        .to_rgb32f();
    let img_shape = camera.img_shape;
    assert!(img_gt.dimensions() == (img_shape.0 as u32, img_shape.1 as u32));
    let img_gt = img_gt.to_vec();
    {
        // computing depth image
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = arrayref::array_mut_ref![pix, 0, 3];
            let (ray_org, ray_dir) = camera.ray(i_pix, [0f32; 2]);
            let Some((t, _i_shape_entity, _i_elem)) =
                del_raycast_core::shape::intersection_ray_against_shape_entities(
                    &ray_org,
                    &ray_dir,
                    &scene.shape_entities,
                )
            else {
                return;
            };
            let v = t * 0.05;
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
            let (ray_org, ray_dir) = camera.ray(i_pix, [0f32; 2]);
            let Some((_t, i_shape_entity, _i_elem)) =
                del_raycast_core::shape::intersection_ray_against_shape_entities(
                    &ray_org,
                    &ray_dir,
                    &scene.shape_entities,
                )
            else {
                return;
            };
            let i_material = scene.shape_entities[i_shape_entity].material_index.unwrap();
            assert!(
                i_material < scene.materials.len(),
                "{} {}",
                i_material,
                scene.materials.len()
            );
            let reflectance = match &scene.materials[i_material] {
                del_raycast_core::material::Material::Diff(mat) => mat.reflectance,
                _ => {
                    todo!()
                }
            };
            *pix = reflectance;
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
        // path tracing sampling material
        let num_sample = 8 * i;
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = arrayref::array_mut_ref![pix, 0, 3];
            use rand::Rng;
            use rand::SeedableRng;
            let mut rng = rand_chacha::ChaChaRng::seed_from_u64(i_pix as u64);
            let mut l_o = [0., 0., 0.];
            for _i_sample in 0..num_sample {
                let (ray0_org, ray0_dir) = camera.ray(
                    i_pix,
                    [
                        del_raycast_core::sampling::tent(rng.gen::<f32>()),
                        del_raycast_core::sampling::tent(rng.gen::<f32>()),
                    ],
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
        let num_sample = 8 * i;
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = arrayref::array_mut_ref!(pix, 0, 3);
            use rand::Rng;
            use rand::SeedableRng;
            let mut rng = rand_chacha::ChaChaRng::seed_from_u64(i_pix as u64);
            let mut l_o = [0., 0., 0.];
            for _i_sample in 0..num_sample {
                let (ray0_org, ray0_dir) = camera.ray(
                    i_pix,
                    [
                        del_raycast_core::sampling::tent(rng.gen::<f32>()),
                        del_raycast_core::sampling::tent(rng.gen::<f32>()),
                    ],
                );
                let rad = del_raycast_core::monte_carlo_integrator::radiance_mis(
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
        // path tracing next event estimation
        let num_sample = 8 * i;
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = arrayref::array_mut_ref!(pix, 0, 3);
            use rand::Rng;
            use rand::SeedableRng;
            let mut rng = rand_chacha::ChaChaRng::seed_from_u64(i_pix as u64);
            let mut l_o = [0., 0., 0.];
            for _i_sample in 0..num_sample {
                let (ray0_org, ray0_dir) = camera.ray(
                    i_pix,
                    [
                        del_raycast_core::sampling::tent(rng.gen::<f32>()),
                        del_raycast_core::sampling::tent(rng.gen::<f32>()),
                    ],
                );
                let rad = del_raycast_core::monte_carlo_integrator::radiance_nee(
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
                let (ray_org, ray_dir) = camera.ray(
                    i_pix,
                    [
                        del_raycast_core::sampling::tent(rng.gen::<f32>()),
                        del_raycast_core::sampling::tent(rng.gen::<f32>()),
                    ],
                );
                use del_raycast_core::monte_carlo_integrator::Scene;
                let Some((hit_pos, hit_nrm, hit_emission, hit_itrimsh)) =
                    scene.hit_position_normal_emission_at_ray_intersection(&ray_org, &ray_dir)
                else {
                    continue;
                };
                l_o = vec3::add(&l_o, &hit_emission);
                use del_geo_core::vec3;
                let hit_pos = vec3::axpy(1.0e-3, &hit_nrm, &hit_pos);
                if let Some((li, pdf_light, uvec_hit2light)) =
                    scene.radiance_from_light(&hit_pos, &mut rng)
                {
                    let brdf_hit = scene.eval_brdf(
                        hit_itrimsh,
                        &hit_nrm,
                        &ray_dir.scale(-1.).normalize(),
                        &uvec_hit2light,
                    );
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
                let (ray0_org, ray0_dir) = camera.ray(
                    i_pix,
                    [
                        del_raycast_core::sampling::tent(rng.gen::<f32>()),
                        del_raycast_core::sampling::tent(rng.gen::<f32>()),
                    ],
                );
                let Some((hit_pos, hit_nrm, hit_emission, hit_itrimsh)) =
                    scene.hit_position_normal_emission_at_ray_intersection(&ray0_org, &ray0_dir)
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
                let Some((_hit2_pos, _hit2_nrm, hit2_emission, _hit2_itrimsh)) = scene
                    .hit_position_normal_emission_at_ray_intersection(
                        &hit_pos_w_offset,
                        &ray_dir_next,
                    )
                else {
                    continue;
                };
                use del_geo_core::vec3::Vec3;
                use del_raycast_core::monte_carlo_integrator::Scene;
                let brdf = scene.eval_brdf(
                    hit_itrimsh,
                    &hit_nrm,
                    &ray0_dir.scale(-1.).normalize(),
                    &ray_dir_next,
                );
                let cos_hit = ray_dir_next.dot(&hit_nrm).clamp(f32::EPSILON, 1f32);
                let pdf = cos_hit * std::f32::consts::FRAC_1_PI;
                l_o = vec3::add(
                    &l_o,
                    &vec3::element_wise_mult(&hit2_emission, &brdf.scale(cos_hit / pdf)),
                );
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
