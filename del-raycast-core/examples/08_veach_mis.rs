use itertools::Itertools;
use rand::Rng;

struct ShapeEntity {
    pub transform_objlcl2world: [f32; 16],
    pub transform_world2objlcl: [f32; 16],
    pub shape: ShapeType,
    pub material_index: Option<usize>,
    pub area_light_index: Option<usize>,
}

enum ShapeType {
    TriangleMesh {
        tri2vtx: Vec<usize>,
        vtx2xyz: Vec<f32>,
        vtx2nrm: Vec<f32>,
    },
    Sphere {
        radius: f32,
    },
}

struct AreaLight {
    pub spectrum_rgb: Option<[f32; 3]>,
    pub two_sided: bool,
}

fn intersection_ray_against_shape_entities(
    ray_org: &[f32; 3],
    ray_dir: &[f32; 3],
    shape_entities: &Vec<ShapeEntity>,
) -> Option<(f32, usize, usize)> {
    use del_geo_core::mat4_col_major;
    let mut t_min = f32::INFINITY;
    let mut i_shape_entity_min = 0usize;
    let mut i_elem_min = 0usize;
    for (i_shape_entity, shape_entity) in shape_entities.iter().enumerate() {
        let transform_world2objlcl = shape_entity.transform_world2objlcl;
        let ray_org_objlcl =
            mat4_col_major::transform_homogeneous(&transform_world2objlcl, &ray_org).unwrap();
        let ray_dir_objlcl = mat4_col_major::transform_direction(&transform_world2objlcl, &ray_dir);
        match &shape_entity.shape {
            ShapeType::TriangleMesh {
                tri2vtx, vtx2xyz, ..
            } => {
                if let Some((t, i_tri)) =
                    del_msh_core::trimesh3_search_bruteforce::first_intersection_ray(
                        &ray_org_objlcl,
                        &ray_dir_objlcl,
                        &tri2vtx,
                        &vtx2xyz,
                    )
                {
                    if t < t_min {
                        t_min = t;
                        i_shape_entity_min = i_shape_entity;
                        i_elem_min = i_tri;
                    }
                }
            }
            ShapeType::Sphere { radius } => {
                if let Some(t) = del_geo_core::sphere::intersection_ray::<f32>(
                    *radius,
                    &[0f32; 3],
                    &ray_org_objlcl,
                    &ray_dir_objlcl,
                ) {
                    if t < t_min {
                        t_min = t;
                        i_shape_entity_min = i_shape_entity;
                        i_elem_min = 0usize;
                    }
                }
            }
        };
    }
    if t_min == f32::INFINITY {
        None
    } else {
        Some((t_min, i_shape_entity_min, i_elem_min))
    }
}

fn parse_pbrt_file(
    file_path: &str,
) -> anyhow::Result<(
    Vec<ShapeEntity>,
    Vec<AreaLight>,
    del_raycast_core::parse_pbrt::Camera,
)> {
    let scene = pbrt4::Scene::from_file(file_path)?;
    let camera = del_raycast_core::parse_pbrt::camera(&scene);
    for material in scene.materials {
        dbg!(material.name);
        dbg!(material.reflectance);
        dbg!(material.params);
    }
    let mut area_lights = Vec::<AreaLight>::new();
    for area_light in scene.area_lights {
        match area_light {
            pbrt4::types::AreaLight::Diffuse {
                filename,
                two_sided,
                spectrum,
                scale,
            } => {
                dbg!(filename, two_sided, spectrum, scale);
                let spectrum_rgb = if let Some(spectrum) = spectrum {
                    match spectrum {
                        pbrt4::param::Spectrum::Rgb(rgb) => Some(rgb),
                        pbrt4::param::Spectrum::Blackbody(i) => {
                            todo!()
                        }
                    }
                } else {
                    None
                };
                let al = AreaLight {
                    spectrum_rgb,
                    two_sided,
                };
                area_lights.push(al);
            }
        }
    }
    let mut shape_entities = Vec::<ShapeEntity>::new();
    for (i_shape, shape_entity) in scene.shapes.iter().enumerate() {
        dbg!(
            i_shape,
            shape_entity.material_index,
            shape_entity.area_light_index
        );
        let shape = match &shape_entity.params {
            pbrt4::types::Shape::TriangleMesh {
                indices,
                positions,
                normals,
                ..
            } => ShapeType::TriangleMesh {
                tri2vtx: indices.iter().map(|&v| v as usize).collect(),
                vtx2xyz: positions.clone(),
                vtx2nrm: normals.clone(),
            },
            pbrt4::types::Shape::Sphere { radius, .. } => ShapeType::Sphere { radius: *radius },
            _ => {
                panic!()
            }
        };
        let transform_objlcl2world = shape_entity.transform.as_ref().to_owned();
        let transform_world2objlcl =
            del_geo_core::mat4_col_major::try_inverse(&transform_objlcl2world).unwrap();
        shape_entities.push(ShapeEntity {
            shape,
            transform_objlcl2world,
            transform_world2objlcl,
            material_index: shape_entity.material_index,
            area_light_index: shape_entity.area_light_index,
        });
    }
    Ok((shape_entities, area_lights, camera))
}

struct MyScene {
    shape_entities: Vec<ShapeEntity>,
    area_lights: Vec<AreaLight>,
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
        let Some((t, i_shape_entity, i_elem)) =
            intersection_ray_against_shape_entities(ray_org, ray_dir, &self.shape_entities)
        else {
            return None;
        };
        use del_geo_core::mat4_col_major;
        let hit_pos_world = del_geo_core::vec3::axpy(t, &ray_dir, &ray_org);
        let se = &self.shape_entities[i_shape_entity];
        let (hit_nrm_world) = {
            let hit_pos_objlcl =
                mat4_col_major::transform_homogeneous(&se.transform_world2objlcl, &hit_pos_world)
                    .unwrap();
            let hit_nrm_objlcl = match &se.shape {
                ShapeType::TriangleMesh {
                    tri2vtx, vtx2xyz, ..
                } => del_msh_core::trimesh3::to_tri3(tri2vtx, vtx2xyz, i_elem).normal(),
                ShapeType::Sphere { radius } => hit_pos_objlcl.to_owned(),
            };
            let hit_nrm_world =
                mat4_col_major::transform_direction(&se.transform_objlcl2world, &hit_nrm_objlcl);
            let hit_nrm_world = del_geo_core::vec3::normalize(&hit_nrm_world);
            hit_nrm_world
        };
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
    {
        // make obj file
        let mut tri2vtx_out: Vec<usize> = vec![];
        let mut vtx2xyz_out: Vec<f32> = vec![];
        for shape_entity in scene.shape_entities.iter() {
            let (tri2vtx, vtx2xyz_camlcl) = {
                let (tri2vtx, vtx2xyz_objlcl) = match &shape_entity.shape {
                    ShapeType::TriangleMesh {
                        tri2vtx, vtx2xyz, ..
                    } => (tri2vtx.to_owned(), vtx2xyz.to_owned()),
                    ShapeType::Sphere { radius } => {
                        let (tri2vtx, vtx2xyz) = del_msh_core::trimesh3_primitive::sphere_yup::<
                            usize,
                            f32,
                        >(*radius, 32, 32);
                        (tri2vtx, vtx2xyz)
                    }
                };
                let vtx2xyz_world = del_msh_core::vtx2xyz::transform(
                    &vtx2xyz_objlcl,
                    &shape_entity.transform_objlcl2world,
                );
                let vtx2xyz_camlcl = del_msh_core::vtx2xyz::transform(
                    &vtx2xyz_world,
                    &camera.transform_world2camlcl,
                );
                (tri2vtx, vtx2xyz_camlcl)
            };
            del_msh_core::uniform_mesh::merge(
                &mut tri2vtx_out,
                &mut vtx2xyz_out,
                &tri2vtx,
                &vtx2xyz_camlcl,
                3,
            );
        }
        del_msh_core::io_obj::save_tri2vtx_vtx2xyz(
            "target/07_veach_mis.obj",
            &tri2vtx_out,
            &vtx2xyz_out,
            3,
        )?;
    }
    {
        // computing depth image
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = arrayref::array_mut_ref![pix, 0, 3];
            let (ray_org, ray_dir) = camera.ray(i_pix, [0.; 2]);
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
        del_canvas::write_hdr_file("target/07_veach_mis_depth.hdr", camera.img_shape, &img_out)?;
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
