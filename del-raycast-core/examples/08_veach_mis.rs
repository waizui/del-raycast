use del_raycast_core::shape::ShapeEntity;
use itertools::Itertools;
use rand::Rng;

fn parse_pbrt_file(
    file_path: &str,
) -> anyhow::Result<(
    Vec<del_raycast_core::shape::ShapeEntity>,
    Vec<del_raycast_core::area_light::AreaLight>,
    del_raycast_core::parse_pbrt::Camera,
)> {
    let scene = pbrt4::Scene::from_file(file_path)?;
    let camera = del_raycast_core::parse_pbrt::camera(&scene);
    let materials = del_raycast_core::parse_pbrt::parse_material(&scene);
    let area_lights = del_raycast_core::parse_pbrt::parse_area_light(&scene);
    let shape_entities = del_raycast_core::parse_pbrt::parse_shapes(&scene);
    Ok((shape_entities, area_lights, camera))
}

struct MyScene {
    shape_entities: Vec<del_raycast_core::shape::ShapeEntity>,
    area_lights: Vec<del_raycast_core::area_light::AreaLight>,
}

impl del_raycast_core::monte_carlo_integrator::Scene for MyScene {
    fn brdf(&self, itrimsh: usize) -> [f32; 3] {
        todo!()
    }

    fn pdf_light(
        &self,
        hit_pos: &[f32; 3],
        hit_pos_light: &[f32; 3],
        hit_nrm_light: &[f32; 3],
    ) -> f32 {
        todo!()
    }

    fn sample_brdf<Rng: rand::Rng>(
        &self,
        hit_nrm: [f32; 3],
        i_shape_entity: usize,
        rng: &mut Rng,
    ) -> ([f32; 3], [f32; 3], f32) {
        let ray_dir_next: [f32; 3] = del_raycast_core::sampling::hemisphere_cos_weighted(
            &nalgebra::Vector3::<f32>::from(hit_nrm),
            &[rng.gen::<f32>(), rng.gen::<f32>()],
        )
        .into();
        use del_geo_core::vec3::Vec3;
        let se = &self.shape_entities[i_shape_entity];
        let brdf = match se.material_index {
            Some(i_material) => [1f32; 3].scale(std::f32::consts::FRAC_1_PI),
            None => [0f32; 3],
        };
        let cos_hit = ray_dir_next.dot(&hit_nrm).clamp(f32::EPSILON, 1f32);
        let pdf = cos_hit * std::f32::consts::FRAC_1_PI;
        (ray_dir_next, brdf, pdf)
    }
    fn hit_position_normal_emission_at_ray_intersection(
        &self,
        ray_org: &[f32; 3],
        ray_dir: &[f32; 3],
    ) -> Option<([f32; 3], [f32; 3], [f32; 3], usize)> {
        use del_raycast_core::shape::intersection_ray_against_shape_entities;
        let Some((t, i_shape_entity, i_elem)) =
            intersection_ray_against_shape_entities(ray_org, ray_dir, &self.shape_entities)
        else {
            return None;
        };
        let hit_pos_world = del_geo_core::vec3::axpy(t, &ray_dir, &ray_org);
        let se = &self.shape_entities[i_shape_entity];
        let (hit_nrm_world) = del_raycast_core::shape::normal_at(se, &hit_pos_world, i_elem);
        let hit_emission = if let Some(ial) = se.area_light_index {
            match self.area_lights[ial].spectrum_rgb {
                Some(rgb) => rgb,
                None => [0f32; 3],
            }
        } else {
            [0f32; 3]
        };
        Some((hit_pos_world, hit_nrm_world, hit_emission, i_shape_entity))
    }
    fn radiance_from_light<Rng: rand::Rng>(
        &self,
        hit_pos_w_offset: &[f32; 3],
        rng: &mut Rng,
    ) -> Option<([f32; 3], f32, [f32; 3])> {
        todo!()
    }
}

fn main() -> anyhow::Result<()> {
    let pbrt_file_path = "asset/veach-mis/scene-v4.pbrt";
    let (scene, camera) = {
        let (shape_entities, area_lights, camera) = parse_pbrt_file(pbrt_file_path)?;
        let scene = MyScene {
            shape_entities,
            area_lights,
        };
        (scene, camera)
    };
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
                Some((t, ise, ie)) => t,
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
                let (ray_org, ray_dir) = camera.ray(
                    i_pix,
                    [
                        del_raycast_core::sampling::tent(rng.gen::<f32>()),
                        del_raycast_core::sampling::tent(rng.gen::<f32>()),
                    ],
                );
                let rad = del_raycast_core::monte_carlo_integrator::radiance_pt(
                    &ray_org, &ray_dir, &scene, 65, &mut rng,
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
        /*
        let path_error_map = format!("target/02_cornell_box_pt_{}_error_map.hdr", num_sample);
        del_canvas::write_hdr_file_mse_rgb_error_map(path_error_map, img_shape, &img_gt, &img_out);
        let err = del_canvas::rmse_error(&img_gt, &img_out);
        println!("num_sample: {}, mse: {}", num_sample, err);
         */
    }
    Ok(())
}
