#[derive(Default, Clone)]
struct AreaLightGeometry {
    pub i_shape_entity: usize,
    pub area: f32,
    pub cog: [f32; 3],
}

struct MyScene {
    shape_entities: Vec<del_raycast_core::shape::ShapeEntity>,
    area_lights: Vec<del_raycast_core::area_light::AreaLight>,
    materials: Vec<del_raycast_core::material::Material>,
    area_light_geometries: Vec<AreaLightGeometry>,
}

fn parse_pbrt_file(
    file_path: &str,
) -> anyhow::Result<(MyScene, del_raycast_core::parse_pbrt::Camera)> {
    let scene = pbrt4::Scene::from_file(file_path)?;
    let camera = del_raycast_core::parse_pbrt::camera(&scene);
    let materials = del_raycast_core::parse_pbrt::parse_material(&scene);
    let area_lights = del_raycast_core::parse_pbrt::parse_area_light(&scene);
    let shape_entities = del_raycast_core::parse_pbrt::parse_shapes(&scene);
    let area_light_geometries = {
        let mut area_light_geometries = vec![AreaLightGeometry::default(); area_lights.len()];
        for (i_shape_entity, shape_entity) in shape_entities.iter().enumerate() {
            let Some(i_area_light) = shape_entity.area_light_index else {
                continue;
            };
            assert!(i_area_light < area_light_geometries.len());
            area_light_geometries[i_area_light].i_shape_entity = i_shape_entity;
            let (cog, area) = shape_entity.cog_and_area();
            area_light_geometries[i_area_light].area = area;
            area_light_geometries[i_area_light].cog = cog;
        }
        area_light_geometries
    };
    let my_scene = MyScene {
        shape_entities,
        area_lights,
        materials,
        area_light_geometries,
    };
    Ok((my_scene, camera))
}

impl MyScene {
    fn hoge(&self, hit_pos: &[f32; 3]) -> Vec<f32> {
        let mut al2mag = vec![0f32; self.area_light_geometries.len() + 1];
        for ial in 0..self.area_light_geometries.len() {
            let area = self.area_light_geometries[ial].area;
            let cog = self.area_light_geometries[ial].cog;
            let dist_sq = del_geo_core::edge3::squared_length(&cog, hit_pos);
            let emission = self.area_lights[ial].spectrum_rgb.unwrap();
            let emission = emission.iter().fold(f32::NAN, |a, b| a.max(*b));
            let mag = area * emission / dist_sq;
            al2mag[ial + 1] = al2mag[ial] + mag;
        }
        al2mag
    }
}

impl del_raycast_core::monte_carlo_integrator::Scene for MyScene {
    fn eval_brdf(
        &self,
        i_shape_entity: usize,
        obj_nrm: &[f32; 3],
        ray_in_outward_normalized: &[f32; 3],
        ray_out_normalized: &[f32; 3],
    ) -> [f32; 3] {
        use del_geo_core::vec3::Vec3;
        assert!((ray_in_outward_normalized.norm() - 1.0).abs() < 1.0e-5);
        let i_material = self.shape_entities[i_shape_entity].material_index.unwrap();
        del_raycast_core::material::eval_brdf(
            &self.materials[i_material],
            obj_nrm,
            ray_in_outward_normalized,
            ray_out_normalized,
        )
    }

    fn pdf_light(
        &self,
        hit_pos: &[f32; 3],
        hit_pos_light: &[f32; 3],
        hit_nrm_light: &[f32; 3],
        i_shape_element: usize,
    ) -> f32 {
        let ial = self.shape_entities[i_shape_element]
            .area_light_index
            .unwrap();
        let al2mag = self.hoge(hit_pos);
        let pdf0 = (al2mag[ial + 1] - al2mag[ial]) / al2mag.last().unwrap();
        let pdf1 = 1.0 / self.area_light_geometries[ial].area;
        let r2 = del_geo_core::edge3::squared_length(&hit_pos_light, &hit_pos);
        use del_geo_core::vec3;
        use del_geo_core::vec3::Vec3;
        let uvec_hit2light = hit_pos_light.sub(&hit_pos).normalize();
        let cos_theta_light = -vec3::dot(&hit_nrm_light, &uvec_hit2light);
        if cos_theta_light <= 0. {
            return f32::EPSILON;
        } // backside of light
        let geo_term = cos_theta_light / r2;
        pdf0 * pdf1 / geo_term
    }
    fn sample_brdf<RNG>(
        &self,
        obj_nrm: &[f32; 3],
        uvec_ray_in: &[f32; 3],
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
            uvec_ray_in,
            rng,
        )
    }
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
        let hit_pos_world = del_geo_core::vec3::axpy(t, &ray_dir, &ray_org);
        let se = &self.shape_entities[i_shape_entity];
        let hit_nrm_world = del_raycast_core::shape::normal_at(se, &hit_pos_world, i_elem);
        let hit_emission = if let Some(ial) = se.area_light_index {
            self.area_lights[ial]
                .spectrum_rgb
                .unwrap_or_else(|| [0f32; 3])
        } else {
            [0f32; 3]
        };
        Some((hit_pos_world, hit_nrm_world, hit_emission, i_shape_entity))
    }

    /// # Return
    /// - `Some(radiance: [f32;3], pdf: f32, uvec_hit2light:[f32;3])`
    ///    - `pdf: f32` the pdf is computed on the unit hemisphere (pdf of light / geometric term)
    /// - `None`
    fn radiance_from_light<RNG: rand::Rng>(
        &self,
        hit_pos: &[f32; 3],
        rng: &mut RNG,
    ) -> Option<([f32; 3], f32, [f32; 3])> {
        use del_geo_core::vec3;
        let al2mag = self.hoge(hit_pos);
        let (ial, _rand1, pdf0) = del_msh_core::cumsum::sample(&al2mag, rng.gen::<f32>());
        let ise = self.area_light_geometries[ial].i_shape_entity;
        let (light_pos, light_nrm, pdf1) =
            &self.shape_entities[ise].sample_uniform(&[rng.gen::<f32>(), rng.gen::<f32>()]);
        let uvec_hit2light = vec3::normalize(&vec3::sub(&light_pos, &hit_pos));
        let cos_theta_light = -vec3::dot(&light_nrm, &uvec_hit2light);
        if cos_theta_light <= f32::EPSILON {
            return None;
        } // backside of light
        if let Some((t, i_shape_entity, _i_tri)) =
            del_raycast_core::shape::intersection_ray_against_shape_entities(
                &hit_pos,
                &uvec_hit2light,
                &self.shape_entities,
            )
        {
            if i_shape_entity != ise {
                return None;
            }
            let light_pos2 = vec3::axpy(t, &uvec_hit2light, hit_pos);
            if del_geo_core::edge3::length(&light_pos, &light_pos2) > 1.0e-3 {
                return None;
            }
        } else {
            return None;
        };
        let r2 = del_geo_core::edge3::squared_length(&light_pos, &hit_pos);
        let geo_term = cos_theta_light / r2;
        let l_i = self.area_lights[ial].spectrum_rgb.unwrap();
        let pdf = pdf0 * pdf1 / geo_term;
        Some((l_i, pdf, uvec_hit2light))
    }
}

fn hoge_nee(
    scene: &MyScene,
    camera: &del_raycast_core::parse_pbrt::Camera,
    num_sample: usize,
    img_gt: &[f32],
) -> anyhow::Result<()> {
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
                &ray0_org, &ray0_dir, scene, 4, &mut rng,
            );
            l_o = del_geo_core::vec3::add(&l_o, &rad);
        }
        *pix = del_geo_core::vec3::scale(&l_o, 1. / num_sample as f32);
    };
    let img_out = {
        let mut img_out = vec![0f32; camera.img_shape.0 * camera.img_shape.1 * 3];
        use rayon::prelude::*;
        img_out
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        img_out
    };
    del_canvas::write_hdr_file(
        format!("target/07_cornell_box_nee_{}.hdr", num_sample),
        camera.img_shape,
        &img_out,
    )?;
    let path_error_map = format!("target/02_cornell_box_nee_{}_error_map.hdr", num_sample);
    del_canvas::write_hdr_file_mse_rgb_error_map(
        path_error_map,
        camera.img_shape,
        &img_gt,
        &img_out,
    );
    let err = del_canvas::rmse_error(&img_gt, &img_out);
    println!("num_sample: {}, mse: {}", num_sample, err);
    Ok(())
}

fn hoge_pt(
    scene: &MyScene,
    camera: &del_raycast_core::parse_pbrt::Camera,
    num_sample: usize,
    img_gt: &[f32],
) -> anyhow::Result<()> {
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
            let rad = del_raycast_core::monte_carlo_integrator::radiance_pt(
                &ray_org, &ray_dir, scene, 3, &mut rng,
            );
            l_o = del_geo_core::vec3::add(&l_o, &rad);
        }
        *pix = del_geo_core::vec3::scale(&l_o, 1. / num_sample as f32);
    };
    let img_out = {
        let mut img_out = vec![0f32; camera.img_shape.0 * camera.img_shape.1 * 3];
        use rayon::prelude::*;
        img_out
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        img_out
    };
    del_canvas::write_hdr_file(
        format!("target/08_veach_mis_pt_{}.hdr", num_sample),
        camera.img_shape,
        &img_out,
    )?;
    let path_error_map = format!("target/08_veach_mis_pt_{}_error_map.hdr", num_sample);
    del_canvas::write_hdr_file_mse_rgb_error_map(
        path_error_map,
        camera.img_shape,
        &img_gt,
        &img_out,
    );
    let err = del_canvas::rmse_error(&img_gt, &img_out);
    println!("num_sample: {}, mse: {}", num_sample, err);
    Ok(())
}

fn hoge_mis(
    scene: &MyScene,
    camera: &del_raycast_core::parse_pbrt::Camera,
    num_sample: usize,
    img_gt: &[f32],
) -> anyhow::Result<()> {
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
                &ray0_org, &ray0_dir, scene, 65, &mut rng,
            );
            l_o = del_geo_core::vec3::add(&l_o, &rad);
        }
        *pix = del_geo_core::vec3::scale(&l_o, 1. / num_sample as f32);
    };
    let img_out = {
        let mut img_out = vec![0f32; camera.img_shape.0 * camera.img_shape.1 * 3];
        use rayon::prelude::*;
        img_out
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        img_out
    };
    del_canvas::write_hdr_file(
        format!("target/02_cornell_box_mis_{}.hdr", num_sample),
        camera.img_shape,
        &img_out,
    )?;
    let path_error_map = format!("target/02_cornell_box_mis_{}_error_map.hdr", num_sample);
    del_canvas::write_hdr_file_mse_rgb_error_map(
        path_error_map,
        camera.img_shape,
        &img_gt,
        &img_out,
    );
    let err = del_canvas::rmse_error(&img_gt, &img_out);
    println!("num_sample: {}, mse: {}", num_sample, err);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let pbrt_file_path = "asset/veach-mis/scene-v4.pbrt";
    let (scene, camera) = parse_pbrt_file(pbrt_file_path)?;
    del_raycast_core::shape::write_wavefront_obj_file_from_camera_view(
        "target/08_veach_mis.obj",
        &scene.shape_entities,
        &camera.transform_world2camlcl,
    )?;
    {
        // computing depth image
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = arrayref::array_mut_ref![pix, 0, 3];
            let (ray_org, ray_dir) = camera.ray(i_pix, [0.; 2]);
            use del_raycast_core::shape::intersection_ray_against_shape_entities;
            let t = match intersection_ray_against_shape_entities(
                &ray_org,
                &ray_dir,
                &scene.shape_entities,
            ) {
                Some((t, _ise, _ie)) => t,
                None => f32::INFINITY,
            };
            let v = t * 0.05;
            *pix = [v; 3];
        };
        let mut img_out = vec![0f32; camera.img_shape.0 * camera.img_shape.1 * 3];
        use rayon::prelude::*;
        img_out
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        del_canvas::write_hdr_file("target/08_veach_mis_depth.hdr", camera.img_shape, &img_out)?;
    }
    let img_gt = image::open("asset/veach-mis/TungstenRender.exr")
        .unwrap()
        .to_rgb32f();
    assert_eq!(
        img_gt.dimensions(),
        (camera.img_shape.0 as u32, camera.img_shape.1 as u32)
    );
    //
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
                let Some((hit_pos, hit_nrm, hit_emission, hit_i_shape_entity)) =
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
                        hit_i_shape_entity,
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
            let mut img_out = vec![0f32; camera.img_shape.0 * camera.img_shape.1 * 3];
            use rayon::prelude::*;
            img_out
                .par_chunks_mut(3)
                .enumerate()
                .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
            img_out
        };
        del_canvas::write_hdr_file(
            format!("target/08_veach_mis_light_sampling_{}.hdr", num_sample),
            camera.img_shape,
            &img_out,
        )?;
        let path_error_map = format!(
            "target/08_veach_mis_light_sampling_{}_error_map.hdr",
            num_sample
        );
        del_canvas::write_hdr_file_mse_rgb_error_map(
            path_error_map,
            camera.img_shape,
            &img_gt,
            &img_out,
        );
        let err = del_canvas::rmse_error(&img_gt, &img_out);
        println!("num_sample: {}, mse: {}", num_sample, err);
    }
    println!("---------------------MIS sampling---------------------");
    for i in 1..4 {
        let num_sample = 8 * i;
        hoge_mis(&scene, &camera, num_sample, &img_gt)?;
    }
    println!("---------------------path tracer---------------------");
    for i in 1..4 {
        // path tracing sampling material
        let num_sample = 8 * i;
        hoge_pt(&scene, &camera, num_sample, &img_gt)?;
    }
    println!("---------------------NEE tracer---------------------");
    for i in 1..4 {
        // path tracing next event estimation
        let num_sample = 8 * i;
        hoge_nee(&scene, &camera, num_sample, &img_gt)?;
    }
    Ok(())
}
