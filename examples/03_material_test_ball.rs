use del_msh_core::vtx2xyz::transform;
use pbrt4::types::Camera;
use ply_rs::ply::PropertyAccess;

struct Shape {
    vtx2xyz: Vec<f32>,
    tri2vtx: Vec<usize>,
    transform: [f32;16]
}

fn parse() -> anyhow::Result<(Vec<Shape>, f32, [f32;16], (usize, usize))> {
    let path_file = "examples/asset/material-testball/scene-v4.pbrt";
    let scene = pbrt4::Scene::from_file(path_file)?;
    dbg!(scene.shapes.len());
    let mut shapes:  Vec<Shape> = vec!();
    let (camera_fov,transform_cam_glbl2lcl) = {
        let camera = scene.camera.unwrap();
        let transform = camera.transform.to_cols_array();
        dbg!(&camera.params);
        let fov = match camera.params {
            Camera::Perspective {fov, ..} => {
                fov
            },
            _ => {todo!()}
        };
        (fov, transform)
    };
    let img_shape = {
        let film = scene.film.unwrap();
        (film.xresolution as usize, film.yresolution as usize)
    };
    for shape_entity in scene.shapes {
        let transform = shape_entity.transform.to_cols_array();
        let (tri2vtx, vtx2xyz) = match shape_entity.params {
            pbrt4::types::Shape::TriangleMesh { positions, indices, .. } => {
                let tri2vtx = indices.iter().map(|&v| v as usize ).collect::<Vec<usize>>();
                (tri2vtx, positions)
            },
            pbrt4::types::Shape::PlyMesh { filename } => {
                let filename = filename.strip_suffix("\"").unwrap().to_string();
                let filename = filename.strip_prefix("\"").unwrap().to_string();
                let mut path = std::path::Path::new(path_file);
                let path = path.parent().unwrap();
                let path = path.to_str().unwrap().to_string() + "/" + &filename;
                let mut f = std::fs::File::open(path)?;
                let p = ply_rs::parser::Parser::<ply_rs::ply::DefaultElement>::new();
                let ply = p.read_ply(&mut f)?;
                let mut vtx2xyz: Vec<f32> = vec!();
                let mut tri2vtx: Vec<usize> = vec!();
                for (str,vals) in ply.payload {
                    if str == "vertex" {
                        vtx2xyz.resize(vals.len() * 3, 0f32);
                        for (i_vtx, val) in vals.iter().enumerate() {
                            vtx2xyz[i_vtx*3+0] = val.get_float(&"x".to_string()).unwrap();
                            vtx2xyz[i_vtx*3+1] = val.get_float(&"y".to_string()).unwrap();
                            vtx2xyz[i_vtx*3+2] = val.get_float(&"z".to_string()).unwrap();
                            /*
                            let nx = val.get_float(&"nx".to_string()).unwrap();
                            let ny = val.get_float(&"ny".to_string()).unwrap();
                            let nz = val.get_float(&"nz".to_string()).unwrap();
                            let u = val.get_float(&"u".to_string()).unwrap();
                            let v = val.get_float(&"v".to_string()).unwrap();
                             */
                        }
                    }
                    else if str == "face" {
                        tri2vtx.resize(vals.len() * 3, 0usize);
                        for (i_vtx, val) in vals.iter().enumerate() {
                            let a = val.get_list_int(&"vertex_indices".to_string()).unwrap();
                            tri2vtx[i_vtx*3] = a[0] as usize;
                            tri2vtx[i_vtx*3+1] = a[1] as usize;
                            tri2vtx[i_vtx*3+2] = a[2] as usize;
                        }
                    }
                }
                (tri2vtx, vtx2xyz)
            },
            _ => { panic!(); }
        };
        let shape = Shape {
            tri2vtx,
            vtx2xyz,
            transform
        };
        shapes.push(shape);
    }
    Ok((shapes, camera_fov, transform_cam_glbl2lcl, img_shape))
}

fn main() -> anyhow::Result<()>{
    let (shape, camera_fov, transform_cam_glbl2lcl, img_shape) = parse()?;
    {
        let mut tri2vtx: Vec<usize> = vec![];
        let mut vtx2xyz: Vec<f32> = vec![];
        for trimesh in shape.iter() {
            let t = del_geo_core::mat4_col_major::mult_mat(&transform_cam_glbl2lcl, &trimesh.transform);
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
            "target/3_material_test_ball.obj",
            &tri2vtx,
            &vtx2xyz,
            3,
        )?;
    }

    let transform_cam_lcl2glbl = del_geo_core::mat4_col_major::try_inverse(&transform_cam_glbl2lcl).unwrap();
    let img_shape = (200,150);
    let mut img = Vec::<image::Rgb<f32>>::new();
    img.resize(img_shape.0*img_shape.1, image::Rgb([0_f32;3]));
    for ih in 0..img_shape.1 {
        dbg!(ih);
        for iw in 0..img_shape.0 {
            let (ray_org, ray_dir)
                = del_raycast::cam_pbrt::cast_ray(iw, ih, img_shape, camera_fov, transform_cam_lcl2glbl);
            // compute intersection below
            let mut t_min = f32::INFINITY;
            for trimesh in shape.iter() {
                let ti = del_geo_core::mat4_col_major::try_inverse(&trimesh.transform).unwrap();
                let ray_org = del_geo_core::mat4_col_major::transform_homogeneous(&ti, &ray_org).unwrap();
                let ray_dir = del_geo_core::mat4_col_major::transform_vector(&ti, &ray_dir);
                let Some((t, _i_tri)) = del_msh_core::trimesh3_search_bruteforce::first_intersection_ray(
                    &ray_org, &ray_dir, &trimesh.tri2vtx, &trimesh.vtx2xyz) else { continue; };
                if t < t_min { t_min = t; }
            }
            let v = (t_min-1.5) * 0.8;
            img[ih*img_shape.0+iw] = image::Rgb([v;3]);
            // dbg!(v);
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