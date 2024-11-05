use std::io::{BufRead, Seek, SeekFrom};
use del_msh_core::uniform_mesh::vtx2vtx;
use del_msh_core::vtx2xyz::transform;

struct TriangleMesh {
    vtx2uv: Vec<f32>,
    vtx2nrm: Vec<f32>,
    vtx2xyz: Vec<f32>,
    tri2vtx: Vec<usize>
}

impl TriangleMesh {
    fn new() -> Self {
        TriangleMesh {
            vtx2uv: vec!(),
            vtx2nrm: vec!(),
            vtx2xyz: vec!(),
            tri2vtx: vec!(),
        }
    }
    fn add(&mut self, tag: &String, vals: &Vec<String>) {
        match tag.as_str() {
            "point2 uv" => self.vtx2uv = vals.iter().map(|v| v.parse().unwrap() ).collect(),
            "normal N" => self.vtx2nrm = vals.iter().map(|v| v.parse().unwrap() ).collect(),
            "point3 P" => self.vtx2xyz = vals.iter().map(|v| v.parse().unwrap() ).collect(),
            "integer indices" => self.tri2vtx = vals.iter().map(|v| v.parse().unwrap() ).collect(),
            _ => panic!("{}", tag),
        };
    }
}

fn parse(reader: &mut std::io::BufReader<std::fs::File>) -> anyhow::Result<(String, Vec<String>)>{
    let mut a: String = "".to_string();
    let ipos = reader.seek(SeekFrom::Current(0)).unwrap();
    let tag = {
        reader.read_line(&mut a)?;
        a = a.trim_start().to_string();
        if a.is_empty() || a.chars().nth(0).unwrap() != '"' { // if start with \"
            reader.seek(SeekFrom::Start(ipos)).unwrap(); // re-wind
            return Ok(("".to_string(),vec!()));
        }
        a = a.trim_end().to_string();
        let re = regex::Regex::new(r##""([^"]*)""##).unwrap(); // extract double quoted word
        let tags: Vec<String> = re.captures_iter(&a)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .collect();
        if tags.is_empty() {
            return Err(anyhow::Error::msg("hoge"));
        }
        tags[0].clone()
    };
    a = a[tag.len()+2..].to_string();
    a = a.trim_start().to_string();
    assert!(a.starts_with("["));
    loop {
        if a.ends_with("]") {
            break;
        }
        reader.read_line(&mut a)?;
        a = a.trim_end().to_string();
    }
    let a: Vec<_> = a.split_whitespace().collect();
    let a: Vec<_> = a[1..a.len()-1].iter().map(|v|v.to_string()).collect();
    Ok((tag, a))
}

fn parse_pbrt_file(file_path: &str) -> anyhow::Result<(Vec<TriangleMesh>, f32, [f32;16], (usize, usize))> {
    let path = std::path::Path::new(file_path);
    let file = std::fs::File::open(&path)?;
    use std::io::BufRead;
    let mut reader = std::io::BufReader::new(file);
    let mut a: String = "".to_string(); // buffer
    let mut trimesh3s = Vec::<TriangleMesh>::new();
    let mut camera_fov = f32::NAN;
    let mut transform_world2ndc = [0f32;16];
    let mut img_shape = (usize::MAX, usize::MAX);
    while reader.read_line(&mut a)? > 0  {
        if a.starts_with("Camera") {
            let (tag, strs) = parse(&mut reader)?;
            dbg!(&tag, &strs);
            assert_eq!(tag, "float fov");
            camera_fov = strs[0].parse()?;
        }
        else if a.starts_with("Transform") {
            dbg!(&a);
            let a = a["Transform".len()..].to_string();
            let a = a.trim().to_string();
            let mut vals: Vec<_> = a.split_whitespace().collect();
            dbg!(&vals);
            assert_eq!(vals.len(),18);
            assert_eq!(vals[0],"[");
            assert_eq!(vals[vals.len()-1],"]");
            transform_world2ndc = std::array::from_fn(|i| vals[i+1].parse().unwrap());
        }
        else if a.starts_with("Shape") {
            let vals: Vec<_> = a.split(" ").collect();
            if vals[1].trim() ==  "\"trianglemesh\"" {
                let mut trimesh3 = TriangleMesh::new();
                loop {
                    let (tag, strs) = parse(&mut reader)?;
                    if tag == "" {
                        break;
                    }
                    trimesh3.add(&tag, &strs);
                }
                trimesh3s.push(trimesh3);
            }
            else {
                todo!();
            }
        }
        else if a.starts_with("Film") {
            let vals: Vec<_> = a.split(" ").collect();
            if vals[1].trim() ==  "\"rgb\"" {
                loop {
                    let (tag, strs) = parse(&mut reader)?;
                    dbg!(&tag, &strs);
                    if tag == "integer xresolution" {
                        img_shape.0 = strs[0].parse()?;
                        dbg!("fdafdsa",img_shape);
                    }
                    if tag == "integer yresolution" {
                        img_shape.1 = strs[0].parse()?;
                    }
                    if tag == "" {
                        break;
                    }
                }
            }
            else {
                todo!();
            }
        }
        a.clear();
    }
    dbg!(&img_shape);
    Ok((trimesh3s, camera_fov, transform_world2ndc, img_shape))
}

fn main() -> anyhow::Result<()>{
    let pbrt_file_path = "examples/asset/cornell-box/scene-v4.pbrt";
    let (trimeshs, camera_fov, transform, img_shape)
        = parse_pbrt_file(pbrt_file_path)?;
    {
        let mut tri2vtx: Vec<usize> = vec!();
        let mut vtx2xyz: Vec<f32> = vec!();
        for trimesh in trimeshs.iter() {
            del_msh_core::uniform_mesh::merge(
                &mut tri2vtx, &mut vtx2xyz,
                &trimesh.tri2vtx, &trimesh.vtx2xyz, 3);
        }
        del_msh_core::io_obj::save_tri2vtx_vtx2xyz(
            "target/2_cornell_box.obj", &tri2vtx, &vtx2xyz, 3)?;
    }

    for iw in 0..img_shape.0 {
        for ih in 0..img_shape.1 {
            // let (ray_org, ray_dir) = compute_ray(iw, ih, img_shape, camera_fov, transform);
            // compute intersection below
            // del_msh_core::trimesh3_search_bruteforce::first_intersection_ray(ray_org, ray_dir, vtx2xyz, vtx2vtx)
        }
    }
    Ok(())
}