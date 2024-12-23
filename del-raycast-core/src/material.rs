#[derive(Debug)]
pub enum Material {
    None,
    Diff(DiffuseMaterial),
    Cond(ConductorMaterial),
}

#[derive(Debug)]
pub struct DiffuseMaterial {
    pub reflectance: [f32; 3],
}

#[derive(Debug)]
pub struct ConductorMaterial {
    pub uroughness: f32,
    pub vroughness: f32,
    pub k: [f32; 3],
    pub eta: [f32; 3],
}
