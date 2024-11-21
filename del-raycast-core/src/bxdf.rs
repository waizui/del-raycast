use nalgebra::{Vector2, Vector3};

type Vec3 = Vector3<f32>;
type Vec2 = Vector2<f32>;

pub enum BxDFFlags {
    Unset = 0,
    Reflection = 1,
    Transmission = 1 << 2,
    Diffuse = 1 << 3,
    Glossy = 1 << 4,
    Specular = 1 << 5,
}

pub trait BxDF {
    fn get_type(&self) -> BxDFFlags;

    fn f(&self, wo: Vec3, wi: Vec3) -> Vec3;

    fn pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        0.0
    }

    fn sample_f(&self, wo: Vec3, sample: Vec2) -> Option<(Vec3, Vec3, f32)> {
        None
    }
}
