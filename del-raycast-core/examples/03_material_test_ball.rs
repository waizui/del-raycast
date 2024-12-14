use del_msh_core::search_bvh3::TriMeshWithBvh;

struct Shape {
    vtx2xyz: Vec<f32>,
    tri2vtx: Vec<usize>,
    transform: [f32; 16],
    bvhnodes: Vec<usize>,
    bvhnode2aabb: Vec<f32>,
    material: Material,
}

#[derive(Debug)]
enum Material {
    None,
    Diff(DiffuseMaterial),
    Cond(ConductorMaterial),
}

#[derive(Debug)]
struct DiffuseMaterial {
    pub refl: [f32; 3],
}

#[derive(Debug)]
struct ConductorMaterial {
    pub uroughness: f32,
    pub vroughness: f32,
}

fn parse_material(scene: &pbrt4::Scene, shape: &pbrt4::ShapeEntity) -> Material {
    let mat_i = {
        match shape.material_index {
            None => return Material::None,
            Some(i) => i,
        }
    };

    let mat = &scene.materials[mat_i];
    match mat.name.as_str() {
        "Stand" => {
            let mut refl = [0.; 3];
            if let Some((_, _, val)) = mat.params.get("reflectance") {
                let rgb: Vec<f32> = val
                    .split(" ")
                    .map(|v| v.parse::<f32>().unwrap_or_default())
                    .collect();

                if rgb.len() >= 3 {
                    refl[0] = rgb[0];
                    refl[1] = rgb[1];
                    refl[2] = rgb[2];
                }
            }

            Material::Diff(DiffuseMaterial { refl })
        }
        "RoughMetal" => {
            let mut uroughness = 0.;
            let mut vroughness = 0.;
            if let Some((_, _, val1)) = mat.params.get("uroughness") {
                if let Some((_, _, val2)) = mat.params.get("vroughness") {
                    uroughness = val1.parse::<f32>().unwrap_or_default();
                    vroughness = val2.parse::<f32>().unwrap_or_default();
                }
            }

            Material::Cond(ConductorMaterial {
                uroughness,
                vroughness,
            })
        }
        _ => Material::None,
    }
}

fn parse() -> anyhow::Result<(Vec<Shape>, f32, [f32; 16], (usize, usize))> {
    let path_file = "asset/material-testball/scene-v4.pbrt";
    let scene = pbrt4::Scene::from_file(path_file)?;
    // dbg!(scene.shapes.len());
    let mut shapes: Vec<Shape> = vec![];
    let (camera_fov, transform_cam_glbl2lcl, img_shape) =
        del_raycast_core::parse_pbrt::hoge(&scene);
    for shape_entity in scene.shapes.iter() {
        let transform = shape_entity.transform.to_cols_array();
        let (_, _, tri2vtx, vtx2xyz, _) =
            del_raycast_core::parse_pbrt::trimesh3_from_shape_entity(&shape_entity, path_file)
                .unwrap();
        let bvhnodes = del_msh_core::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
        let bvhnode2aabb = del_msh_core::bvhnode2aabb3::from_uniform_mesh_with_bvh(
            0,
            &bvhnodes,
            Some((&tri2vtx, 3)),
            &vtx2xyz,
            None,
        );
        let mat = parse_material(&scene, &shape_entity);
        let shape = Shape {
            tri2vtx,
            vtx2xyz,
            transform,
            bvhnodes,
            bvhnode2aabb,
            material: mat,
        };

        shapes.push(shape);
    }
    Ok((shapes, camera_fov, transform_cam_glbl2lcl, img_shape))
}

fn main() -> anyhow::Result<()> {
    let (shapes, camera_fov, transform_cam_glbl2lcl, img_shape) = parse()?;
    {
        let mut tri2vtx: Vec<usize> = vec![];
        let mut vtx2xyz: Vec<f32> = vec![];
        for trimesh in shapes.iter() {
            let t =
                del_geo_core::mat4_col_major::mult_mat(&transform_cam_glbl2lcl, &trimesh.transform);
            let trimesh_vtx2xyz = del_msh_core::vtx2xyz::transform(&trimesh.vtx2xyz, &t);
            del_msh_core::uniform_mesh::merge(
                &mut tri2vtx,
                &mut vtx2xyz,
                &trimesh.tri2vtx,
                &trimesh_vtx2xyz,
                3,
            );
        }
        del_msh_core::io_obj::save_tri2vtx_vtx2xyz(
            "target/03_material_test_ball.obj",
            &tri2vtx,
            &vtx2xyz,
            3,
        )?;
    }

    let transform_cam_lcl2glbl =
        del_geo_core::mat4_col_major::try_inverse(&transform_cam_glbl2lcl).unwrap();
    let img_shape = (540, 360);
    let mut img = Vec::<image::Rgb<f32>>::new();
    img.resize(img_shape.0 * img_shape.1, image::Rgb([0_f32; 3]));
    for ih in 0..img_shape.1 {
        for iw in 0..img_shape.0 {
            let (ray_org, ray_dir) = del_raycast_core::cam_pbrt::cast_ray(
                (iw, ih),
                (0., 0.),
                img_shape,
                camera_fov,
                transform_cam_lcl2glbl,
            );
            // compute intersection below
            let mut t_min = f32::INFINITY;
            let mut shape_i: usize = usize::MAX;
            for (s_i, trimesh) in shapes.iter().enumerate() {
                let ti = del_geo_core::mat4_col_major::try_inverse(&trimesh.transform).unwrap();
                let ray_org =
                    del_geo_core::mat4_col_major::transform_homogeneous(&ti, &ray_org).unwrap();
                let ray_dir = del_geo_core::mat4_col_major::transform_vector(&ti, &ray_dir);
                let Some((t, _i_tri)) =
                /*790
                    del_msh_core::trimesh3_search_bruteforce::first_intersection_ray(
                        &ray_org,
                        &ray_dir,
                        &trimesh.tri2vtx,
                        &trimesh.vtx2xyz,
                    )
                 */
                del_msh_core::search_bvh3::first_intersection_ray(&ray_org, &ray_dir, &TriMeshWithBvh {
                    tri2vtx: &trimesh.tri2vtx,
                    vtx2xyz: &trimesh.vtx2xyz,
                    bvhnodes: &trimesh.bvhnodes,
                    bvhnode2aabb: &trimesh.bvhnode2aabb,
                }, 0, f32::INFINITY)
                else {
                    continue;
                };
                if t < t_min {
                    shape_i = s_i;
                    t_min = t;
                }
            }

            // TODO:delete test code
            let v = (t_min - 1.5) * 0.8;
            let c = match &shapes[shape_i].material {
                Material::Diff(_) => [0., 0., v],
                Material::Cond(_) => [0., v, 0.],
                _ => [v; 3],
            };

            img[ih * img_shape.0 + iw] = image::Rgb(c);
        }
    }
    {
        use image::codecs::hdr::HdrEncoder;
        let file = std::fs::File::create("target/03_material_test_ball.hdr").unwrap();
        let enc = HdrEncoder::new(file);
        let _ = enc.encode(&img, img_shape.0, img_shape.1);
    }
    Ok(())
}
