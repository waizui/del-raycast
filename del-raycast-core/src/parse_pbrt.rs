use ply_rs::ply::PropertyAccess;
pub struct Camera {
    pub camera_fov: f32,
    /// this transformation actually flip the scene in x direction.
    /// the camera is looking from -z to +z direction
    pub transform_world2camlcl: [f32; 16],
    /// inverse of `transform_world2camlcl'
    pub transform_camlcl2world: [f32; 16],
    pub img_shape: (usize, usize),
}

pub fn camera(scene: &pbrt4::Scene) -> Camera {
    let camera = scene.camera.as_ref().unwrap();
    let transform = camera.transform.to_cols_array();
    // dbg!(&camera.params);
    let fov = match camera.params {
        pbrt4::types::Camera::Perspective { fov, .. } => fov,
        _ => {
            todo!()
        }
    };
    let film = scene.film.as_ref().unwrap();
    let img_shape = (film.xresolution as usize, film.yresolution as usize);
    Camera {
        camera_fov: fov,
        transform_world2camlcl: transform,
        transform_camlcl2world: del_geo_core::mat4_col_major::try_inverse(&transform).unwrap(),
        img_shape,
    }
}

impl Camera {
    /// # Return
    /// `(ray_org: [f32;3], ray_dir [f32;3])`
    /// ray_org: camera focus point
    /// ray_dir: ray direction (not normalized)
    pub fn ray(&self, i_pix: usize, offset: [f32; 2]) -> ([f32; 3], [f32; 3]) {
        crate::cam_pbrt::cast_ray_plus_z(
            (i_pix % self.img_shape.0, i_pix / self.img_shape.0),
            offset.into(),
            self.img_shape,
            self.camera_fov,
            self.transform_camlcl2world,
        )
    }
}

#[allow(clippy::type_complexity)]
pub fn trimesh3_from_shape_entity(
    shape_entity: &pbrt4::ShapeEntity,
    path_file: &str,
) -> Option<(usize, Option<usize>, Vec<usize>, Vec<f32>, Vec<f32>)> {
    match &shape_entity.params {
        pbrt4::types::Shape::TriangleMesh {
            positions,
            indices,
            normals,
            ..
        } => {
            let tri2vtx = indices.iter().map(|&v| v as usize).collect::<Vec<usize>>();
            Some((
                shape_entity.material_index.unwrap(),
                shape_entity.area_light_index,
                tri2vtx,
                positions.to_vec(),
                normals.to_vec(),
            ))
        }
        pbrt4::types::Shape::PlyMesh { filename } => {
            let filename = filename.strip_suffix("\"").unwrap().to_string();
            let filename = filename.strip_prefix("\"").unwrap().to_string();
            let path = std::path::Path::new(path_file);
            let path = path.parent().unwrap();
            let path = path.to_str().unwrap().to_string() + "/" + &filename;
            let mut f = std::fs::File::open(path).unwrap();
            let p = ply_rs::parser::Parser::<ply_rs::ply::DefaultElement>::new();
            let ply = p.read_ply(&mut f).unwrap();
            let mut vtx2xyz: Vec<f32> = vec![];
            let mut tri2vtx: Vec<usize> = vec![];
            for (str, vals) in ply.payload {
                if str == "vertex" {
                    vtx2xyz.resize(vals.len() * 3, 0f32);
                    for (i_vtx, val) in vals.iter().enumerate() {
                        vtx2xyz[i_vtx * 3] = val.get_float(&"x".to_string()).unwrap();
                        vtx2xyz[i_vtx * 3 + 1] = val.get_float(&"y".to_string()).unwrap();
                        vtx2xyz[i_vtx * 3 + 2] = val.get_float(&"z".to_string()).unwrap();
                        /*
                        let nx = val.get_float(&"nx".to_string()).unwrap();
                        let ny = val.get_float(&"ny".to_string()).unwrap();
                        let nz = val.get_float(&"nz".to_string()).unwrap();
                        let u = val.get_float(&"u".to_string()).unwrap();
                        let v = val.get_float(&"v".to_string()).unwrap();
                         */
                    }
                } else if str == "face" {
                    tri2vtx.resize(vals.len() * 3, 0usize);
                    for (i_vtx, val) in vals.iter().enumerate() {
                        let a = val.get_list_int(&"vertex_indices".to_string()).unwrap();
                        tri2vtx[i_vtx * 3] = a[0] as usize;
                        tri2vtx[i_vtx * 3 + 1] = a[1] as usize;
                        tri2vtx[i_vtx * 3 + 2] = a[2] as usize;
                    }
                }
            }
            // TODO: parse normals for .ply
            Some((
                shape_entity.material_index.unwrap(),
                shape_entity.area_light_index,
                tri2vtx,
                vtx2xyz,
                vec![],
            ))
        }
        _ => {
            panic!();
        }
    }
}

pub fn spectrum_from_light_entity(area_light_entity: &pbrt4::types::AreaLight) -> Option<[f32; 3]> {
    match area_light_entity {
        pbrt4::types::AreaLight::Diffuse {
            filename: _filename,
            two_sided: _two_sided,
            spectrum,
            scale: _scale,
        } => match spectrum.unwrap().to_owned() {
            pbrt4::param::Spectrum::Rgb(rgb) => Some(rgb),
            _ => {
                panic!();
            }
        },
    }
}
