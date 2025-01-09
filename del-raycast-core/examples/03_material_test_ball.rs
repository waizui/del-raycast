use del_msh_core::search_bvh3::TriMeshWithBvh;

struct Shape {
    vtx2xyz: Vec<f32>,
    tri2vtx: Vec<usize>,
    transform: [f32; 16],
    bvhnodes: Vec<usize>,
    bvhnode2aabb: Vec<f32>,
    material_index: Option<usize>,
}

// return scale factor t and shape index of intersection
fn intersect_bvh(ray_org: &[f32; 3], ray_dir: &[f32; 3], shapes: &[Shape]) -> Option<(f32, usize)> {
    // compute intersection
    let mut t_min = f32::INFINITY;
    let mut shape_i: usize = usize::MAX;
    for (s_i, trimesh) in shapes.iter().enumerate() {
        let ti = del_geo_core::mat4_col_major::try_inverse(&trimesh.transform).unwrap();
        let ray_org = del_geo_core::mat4_col_major::transform_homogeneous(&ti, ray_org).unwrap();
        let ray_dir = del_geo_core::mat4_col_major::transform_direction(&ti, ray_dir);

        let Some((t, _i_tri)) = del_msh_core::search_bvh3::first_intersection_ray(
            &ray_org,
            &ray_dir,
            &TriMeshWithBvh {
                tri2vtx: &trimesh.tri2vtx,
                vtx2xyz: &trimesh.vtx2xyz,
                bvhnodes: &trimesh.bvhnodes,
                bvhnode2aabb: &trimesh.bvhnode2aabb,
            },
            0,
            f32::INFINITY,
        ) else {
            continue;
        };
        if t < t_min {
            shape_i = s_i;
            t_min = t;
        }
    }

    if t_min == f32::INFINITY {
        None
    } else {
        Some((t_min, shape_i))
    }
}

fn parse() -> anyhow::Result<(
    Vec<Shape>,
    del_raycast_core::parse_pbrt::Camera,
    pbrt4::Scene,
)> {
    let path_file = "asset/material-testball/scene-v4.pbrt";
    let scene = pbrt4::Scene::from_file(path_file)?;
    let mut shapes: Vec<Shape> = vec![];
    let camera = del_raycast_core::parse_pbrt::camera(&scene);
    for shape_entity in scene.shapes.iter() {
        let transform = shape_entity.transform.to_cols_array();
        let (_, _, tri2vtx, vtx2xyz, _) =
            del_raycast_core::parse_pbrt::trimesh3_from_shape_entity(shape_entity, path_file)
                .unwrap();
        let bvhnodes = del_msh_core::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
        let bvhnode2aabb = del_msh_core::bvhnode2aabb3::from_uniform_mesh_with_bvh(
            0,
            &bvhnodes,
            Some((&tri2vtx, 3)),
            &vtx2xyz,
            None,
        );
        let shape = Shape {
            tri2vtx,
            vtx2xyz,
            transform,
            bvhnodes,
            bvhnode2aabb,
            material_index: shape_entity.material_index,
        };
        shapes.push(shape);
    }
    Ok((shapes, camera, scene))
}

fn main() -> anyhow::Result<()> {
    let (shapes, camera, scene) = parse()?;
    {
        let mut tri2vtx: Vec<usize> = vec![];
        let mut vtx2xyz: Vec<f32> = vec![];
        for trimesh in shapes.iter() {
            let t = del_geo_core::mat4_col_major::mult_mat(
                &camera.transform_world2camlcl,
                &trimesh.transform,
            );
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
        del_geo_core::mat4_col_major::try_inverse(&camera.transform_world2camlcl).unwrap();
    let img_shape = (540, 360);

    {
        // computing depth image
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = arrayref::array_mut_ref![pix, 0, 3];
            let iw = i_pix % img_shape.0;
            let ih = i_pix / img_shape.0;

            let (ray_org, ray_dir) = del_raycast_core::cam_pbrt::cast_ray_plus_z(
                (iw, ih),
                (0., 0.),
                img_shape,
                camera.camera_fov,
                transform_cam_lcl2glbl,
            );

            let Some((t, _i_shape_entity)) = intersect_bvh(&ray_org, &ray_dir, &shapes) else {
                return;
            };
            let v = (t - 1.5) * 0.8;
            *pix = [v; 3];
        };

        let mut img_out = vec![0f32; img_shape.0 * img_shape.1 * 3];
        use rayon::prelude::*;
        img_out
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        del_canvas::write_hdr_file("target/03_material_test_ball.hdr", img_shape, &img_out)?;
    }

    {
        // computing reflectance image
        let materials = del_raycast_core::parse_pbrt::parse_material(&scene);
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = arrayref::array_mut_ref![pix, 0, 3];
            let iw = i_pix % img_shape.0;
            let ih = i_pix / img_shape.0;

            let (ray_org, ray_dir) = del_raycast_core::cam_pbrt::cast_ray_plus_z(
                (iw, ih),
                (0., 0.),
                img_shape,
                camera.camera_fov,
                transform_cam_lcl2glbl,
            );

            let Some((_t, i_shape_entity)) = intersect_bvh(&ray_org, &ray_dir, &shapes) else {
                return;
            };

            let i_material = shapes[i_shape_entity].material_index.unwrap();

            assert!(
                i_material < materials.len(),
                "{} {}",
                i_material,
                materials.len()
            );

            let reflectance = match &materials[i_material] {
                del_raycast_core::material::Material::Diff(mat) => mat.reflectance,
                del_raycast_core::material::Material::Cond(mat) => mat.reflectance,
                del_raycast_core::material::Material::CoaDiff(mat) => mat.reflectance,
                _ => {
                    panic!("No reflectance of Material");
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
        del_canvas::write_hdr_file(
            "target/03_material_test_ball_color.hdr",
            img_shape,
            &img_out,
        )?;
    }

    Ok(())
}
