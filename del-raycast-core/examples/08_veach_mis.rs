struct MyScene {
    shape_entities: Vec<del_raycast_core::shape::ShapeEntity>,
    area_lights: Vec<del_raycast_core::area_light::AreaLight>,
    materials: Vec<del_raycast_core::material::Material>,
}

fn parse_pbrt_file(
    file_path: &str,
) -> anyhow::Result<(MyScene, del_raycast_core::parse_pbrt::Camera)> {
    let scene = pbrt4::Scene::from_file(file_path)?;
    let camera = del_raycast_core::parse_pbrt::camera(&scene);
    let materials = del_raycast_core::parse_pbrt::parse_material(&scene);
    let area_lights = del_raycast_core::parse_pbrt::parse_area_light(&scene);
    let shape_entities = del_raycast_core::parse_pbrt::parse_shapes(&scene);
    let my_scene = MyScene {
        shape_entities,
        area_lights,
        materials,
    };
    Ok((my_scene, camera))
}

impl del_raycast_core::monte_carlo_integrator::Scene for MyScene {
    fn brdf(&self, _itrimsh: usize) -> [f32; 3] {
        todo!()
    }

    fn pdf_light(
        &self,
        _hit_pos: &[f32; 3],
        _hit_pos_light: &[f32; 3],
        _hit_nrm_light: &[f32; 3],
    ) -> f32 {
        todo!()
    }

    fn sample_brdf<Rng: rand::Rng>(
        &self,
        obj_nrm: &[f32; 3],
        uvec_ray_in: &[f32; 3], // direction same as nrm
        i_shape_entity: usize,
        rng: &mut Rng,
    ) -> Option<([f32; 3], [f32; 3], f32)> {
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
    fn radiance_from_light<RNG: rand::Rng>(
        &self,
        _hit_pos_w_offset: &[f32; 3],
        _rng: &mut RNG,
    ) -> Option<([f32; 3], f32, [f32; 3])> {
        todo!()
    }
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
    assert!(img_gt.dimensions() == (camera.img_shape.0 as u32, camera.img_shape.1 as u32));
    println!("---------------------path tracer---------------------");
    for i in 1..2 {
        // path tracing sampling material
        let num_sample = 1024 * i;
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
                    &ray_org, &ray_dir, &scene, 3, &mut rng,
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
        let path_error_map = format!("target/07_veach_mis_pt_{}_error_map.hdr", num_sample);
        del_canvas::write_hdr_file_mse_rgb_error_map(
            path_error_map,
            camera.img_shape,
            &img_gt,
            &img_out,
        );
        let err = del_canvas::rmse_error(&img_gt, &img_out);
        println!("num_sample: {}, mse: {}", num_sample, err);
    }
    Ok(())
}
