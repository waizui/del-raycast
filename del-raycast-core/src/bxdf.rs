use std::ops::{BitAnd, BitOr};

use nalgebra::{Vector2, Vector3};

type Vec3f = Vector3<f32>;
type Vec2f = Vector2<f32>;

#[derive(Debug, Clone, Copy)]
pub struct BxDFFlags(u32);

impl BxDFFlags {
    pub const UNSET: Self = Self(0);
    pub const REFLECTION: Self = Self(1);
    pub const TRANSMISSION: Self = Self(1 << 2);
    pub const DIFFUSE: Self = Self(1 << 3);
    pub const GLOSSY: Self = Self(1 << 4);
    pub const SPECULAR: Self = Self(1 << 5);
}

#[derive(Debug, Clone, Copy)]
pub struct BxDFReflTransFlags(u32);

impl BxDFReflTransFlags {
    pub const UNSET: Self = Self(0);
    pub const REFLECTION: Self = Self(1);
    pub const TRANSMISSION: Self = Self(1 << 2);
}

impl BitAnd for BxDFFlags {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        BxDFFlags(self.0 & rhs.0)
    }
}

impl BitOr for BxDFFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        BxDFFlags(self.0 | rhs.0)
    }
}

pub struct BSDFSample {
    pub flag: BxDFFlags,
    pub wi: Vec3f,
    pub pdf: f32,
}

impl BSDFSample {
    pub fn has_flag(&self, flag: BxDFFlags) -> bool {
        self.flag.0 & flag.0 != 0
    }
}

#[derive(Debug, Clone, Copy)]
pub enum TransportMode {
    Radiance,
    Importance,
}

/// for extentable sample arguments
#[derive(Debug, Clone, Copy, Default)]
pub struct BxDFExtArgs {
    pub mode: Option<TransportMode>,
    pub sample_flags: Option<BxDFReflTransFlags>,
}

/// interface of BxDFs
pub trait BxDF {
    fn get_type(&self) -> BxDFFlags;

    fn f(&self, wo: Vec3f, wi: Vec3f, args: BxDFExtArgs) -> Vec3f;

    fn pdf(&self, wo: Vec3f, wi: Vec3f, args: BxDFExtArgs) -> f32 {
        0.0
    }

    fn sample_f(&self, wo: Vec3f, uc: f32, u: Vec2f, args: BxDFExtArgs) -> Option<BSDFSample> {
        None
    }
}
