pub trait Scene {
    #[allow(clippy::type_complexity)]
    fn hit_position_normal_emission_at_ray_intersection(
        &self,
        ray_org: &[f32; 3],
        ray_dir: &[f32; 3],
    ) -> Option<([f32; 3], [f32; 3], [f32; 3], usize)>;

    fn brdf(&self, itrimsh: usize) -> [f32; 3];
    fn sample_brdf<Rng: rand::Rng>(
        &self,
        hit_nrm: [f32; 3],
        itrimsh: usize,
        rng: &mut Rng,
    ) -> ([f32; 3], [f32; 3], f32);
    fn radiance_from_light<Rng: rand::Rng>(
        &self,
        hit_pos_w_offset: &[f32; 3],
        rng: &mut Rng,
    ) -> Option<([f32; 3], f32, [f32; 3])>;
    fn pdf_light(
        &self,
        hit_pos: &[f32; 3],
        hit_pos_light: &[f32; 3],
        hit_nrm_light: &[f32; 3],
    ) -> f32;
}

pub fn radiance_pt<RNG, SCENE>(
    ray_org_ini: &[f32; 3],
    ray_dir_ini: &[f32; 3],
    scene: &SCENE,
    max_depth: usize,
    rng: &mut RNG,
) -> [f32; 3]
where
    RNG: rand::Rng,
    SCENE: Scene,
{
    use del_geo_core::vec3::Vec3;
    let mut rad_out = [0f32; 3];
    let mut throughput = [1f32; 3];
    let mut ray_org: [f32; 3] = ray_org_ini.to_owned();
    let mut ray_dir: [f32; 3] = ray_dir_ini.to_owned();
    for _i_depth in 0..max_depth {
        let Some((hit_pos, hit_nrm, hit_emission, hit_itrimsh)) =
            scene.hit_position_normal_emission_at_ray_intersection(&ray_org, &ray_dir)
        else {
            break;
        };
        rad_out = rad_out.add(&hit_emission.element_wise_mult(&throughput));
        //
        let (ray_dir_next, brdf, pdf) = scene.sample_brdf(hit_nrm, hit_itrimsh, rng);
        let cos_hit = ray_dir_next.dot(&hit_nrm);
        throughput = throughput.element_wise_mult(&brdf.scale(cos_hit / pdf));
        {
            let &russian_roulette_prob = throughput
                .iter()
                .max_by(|&a, &b| a.partial_cmp(b).unwrap())
                .unwrap();
            if rng.gen::<f32>() < russian_roulette_prob {
                throughput = del_geo_core::vec3::scale(&throughput, 1.0 / russian_roulette_prob);
            } else {
                break; // terminate ray
            }
        }
        let hit_pos_w_offset = del_geo_core::vec3::axpy(1.0e-3, &hit_nrm, &hit_pos);
        ray_org = hit_pos_w_offset;
        ray_dir = ray_dir_next;
    }
    rad_out
}

pub fn radiance_nee<RNG, SCENE>(
    ray_org_ini: &[f32; 3],
    ray_dir_ini: &[f32; 3],
    scene: &SCENE,
    max_depth: usize,
    rng: &mut RNG,
) -> [f32; 3]
where
    RNG: rand::Rng,
    SCENE: Scene,
{
    use del_geo_core::vec3::Vec3;
    let mut rad_out = [0f32; 3];
    let mut throughput = [1f32; 3];
    let mut ray_org: [f32; 3] = ray_org_ini.to_owned();
    let mut ray_dir: [f32; 3] = ray_dir_ini.to_owned();
    for i_depth in 0..max_depth {
        use del_geo_core::vec3;
        let Some((hit_pos, hit_nrm, hit_emission, hit_itrimsh)) =
            scene.hit_position_normal_emission_at_ray_intersection(&ray_org, &ray_dir)
        else {
            break;
        };
        // ------------
        if i_depth == 0 {
            rad_out = rad_out.add(&hit_emission.element_wise_mult(&throughput));
        };
        let hit_pos_w_offset = vec3::axpy(1.0e-3, &hit_nrm, &hit_pos);
        if hit_emission == [0f32; 3] {
            // sample light
            if let Some((li_light, pdf_light, uvec_hit2light)) =
                scene.radiance_from_light(&hit_pos_w_offset, rng)
            {
                let brdf_hit = scene.brdf(hit_itrimsh);
                let cos_hit = vec3::dot(&uvec_hit2light, &hit_nrm);
                let lo_light =
                    vec3::element_wise_mult(&brdf_hit, &li_light.scale(cos_hit / pdf_light));
                rad_out = rad_out.add(&lo_light.element_wise_mult(&throughput));
            }
        }
        let ray_dir_next = {
            let (ray_dir_next, brdf, pdf) = scene.sample_brdf(hit_nrm, hit_itrimsh, rng);
            let cos_hit = ray_dir_next.dot(&hit_nrm).clamp(f32::EPSILON, 1f32);
            throughput = throughput.element_wise_mult(&brdf.scale(cos_hit / pdf));
            ray_dir_next
        };
        {
            // russian roulette
            let &russian_roulette_prob = throughput
                .iter()
                .max_by(|&a, &b| a.partial_cmp(b).unwrap())
                .unwrap();
            if rng.gen::<f32>() < russian_roulette_prob {
                throughput = vec3::scale(&throughput, 1.0 / russian_roulette_prob);
            } else {
                break; // terminate ray
            }
        }
        ray_org = hit_pos_w_offset;
        ray_dir = ray_dir_next;
    }
    rad_out
}

pub fn radiance_mis<RNG, SCENE>(
    ray_org_ini: &[f32; 3],
    ray_dir_ini: &[f32; 3],
    scene: &SCENE,
    max_depth: usize,
    rng: &mut RNG,
) -> [f32; 3]
where
    RNG: rand::Rng,
    SCENE: Scene,
{
    use del_geo_core::vec3::Vec3;
    let mut rad_out = [0f32; 3];
    let mut throughput = [1f32; 3];
    let mut ray_org: [f32; 3] = ray_org_ini.to_owned();
    let mut ray_dir: [f32; 3] = ray_dir_ini.to_owned();
    for i_depth in 0..max_depth {
        use del_geo_core::vec3;
        let Some((hit_pos, hit_nrm, hit_emission, hit_itrimsh)) =
            scene.hit_position_normal_emission_at_ray_intersection(&ray_org, &ray_dir)
        else {
            break;
        };
        // ------------
        if i_depth == 0 {
            rad_out = rad_out.add(&hit_emission.element_wise_mult(&throughput));
        };
        let hit_pos_w_offset = vec3::axpy(1.0e-3, &hit_nrm, &hit_pos);
        if hit_emission == [0f32; 3] {
            if let Some((li_light, pdf_light, uvec_hit2light)) =
                scene.radiance_from_light(&hit_pos_w_offset, rng)
            {
                let brdf_hit = scene.brdf(hit_itrimsh);
                let cos_hit = vec3::dot(&uvec_hit2light, &hit_nrm);
                let pdf_brdf = cos_hit * std::f32::consts::FRAC_1_PI;
                let mis_weight_light = pdf_light / (pdf_light + pdf_brdf);
                let lo_light = vec3::element_wise_mult(
                    &brdf_hit,
                    &li_light.scale(cos_hit / pdf_light * mis_weight_light),
                );
                rad_out = rad_out.add(&lo_light.element_wise_mult(&throughput));
            }
        }
        if hit_emission == [0f32; 3] {
            let (ray_dir_brdf, brdf, pdf_brdf) = scene.sample_brdf(hit_nrm, hit_itrimsh, rng);
            if let Some((hit_pos_light, hit_nrm_light, hit_emission, _hit_itrimsh_light)) = scene
                .hit_position_normal_emission_at_ray_intersection(&hit_pos_w_offset, &ray_dir_brdf)
            {
                if hit_emission != [0f32; 3] {
                    let cos_hit = ray_dir_brdf.dot(&hit_nrm).clamp(f32::EPSILON, 1f32);
                    let pdf_light = scene.pdf_light(&hit_pos, &hit_pos_light, &hit_nrm_light);
                    let mis_weight_brdf = pdf_brdf / (pdf_brdf + pdf_light);
                    let lo_brdf = hit_emission
                        .element_wise_mult(&brdf.scale(cos_hit / pdf_brdf * mis_weight_brdf));
                    rad_out = rad_out.add(&lo_brdf.element_wise_mult(&throughput));
                }
            }
        }
        let ray_dir_next = {
            // update throughput
            let (ray_dir_next, brdf, pdf_brdf) = scene.sample_brdf(hit_nrm, hit_itrimsh, rng);
            let cosine = ray_dir_next.dot(&hit_nrm).clamp(f32::EPSILON, 1f32);
            throughput = throughput.element_wise_mult(&brdf.scale(cosine / pdf_brdf));
            ray_dir_next
        };
        {
            let &russian_roulette_prob = throughput
                .iter()
                .max_by(|&a, &b| a.partial_cmp(b).unwrap())
                .unwrap();
            if rng.gen::<f32>() < russian_roulette_prob {
                throughput = vec3::scale(&throughput, 1.0 / russian_roulette_prob);
            } else {
                break; // terminate ray
            }
        }
        ray_org = hit_pos_w_offset;
        ray_dir = ray_dir_next;
    }
    rad_out
}
