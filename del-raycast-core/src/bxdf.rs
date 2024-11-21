use std::ops::{BitAnd, BitOr, Mul};

use nalgebra::{Vector2, Vector3};

type Real = f32;

type Vec3f = Vector3<Real>;
type Vec2f = Vector2<Real>;

const N_SPECTRUM_SAMPLES: usize = 4;
const INV_PI: Real = 1. / std::f32::consts::PI;

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

#[derive(Debug, Clone, Copy)]
pub struct SampledSpectrum {
    pub arr: [Real; N_SPECTRUM_SAMPLES],
}

impl Mul<Real> for SampledSpectrum {
    type Output = Self;
    fn mul(self, rhs: Real) -> Self::Output {
        let res = self.arr.map(|x| x * rhs);
        SampledSpectrum { arr: res }
    }
}

pub struct BSDFSample {
    pub flag: BxDFFlags,
    pub wi: Vec3f,
    pub pdf: Real,
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
// #[derive(Debug, Clone, Copy, Default)]
// pub struct BxDFExtArgs {
//     pub mode: Option<TransportMode>,
//     pub sample_flags: Option<BxDFReflTransFlags>,
// }

// /// interface of BxDFs
// pub trait BxDF {
//     fn get_type(&self) -> BxDFFlags;
//
//     fn f(&self, wo: Vec3f, wi: Vec3f, args: BxDFExtArgs) -> SampledSpectrum;
//
//     fn pdf(&self, wo: Vec3f, wi: Vec3f, args: BxDFExtArgs) -> f32 {
//         0.0
//     }
//
//     fn sample_f(&self, wo: Vec3f, uc: f32, u: Vec2f, args: BxDFExtArgs) -> Option<BSDFSample> {
//         None
//     }
// }

pub struct DiffuseBxDF {
    r: SampledSpectrum,
}

impl DiffuseBxDF {
    pub fn new(r: SampledSpectrum) -> Self {
        DiffuseBxDF { r }
    }

    pub fn f(&self, wo: Vec3f, wi: Vec3f) -> SampledSpectrum {
        if wo.z * wi.z < 0. {
            return SampledSpectrum {
                arr: [0.; N_SPECTRUM_SAMPLES],
            };
        }
        self.r * INV_PI
    }
    pub fn pdf(&self, wo: Vec3f, wi: Vec3f, sf: BxDFReflTransFlags) -> Real {
        todo!()
    }

    pub fn sample_f(
        &self,
        wo: Vec3f,
        uc: Real,
        u: Vec2f,
        sf: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        todo!()
    }

    pub fn get_type(&self) -> BxDFFlags {
        BxDFFlags::DIFFUSE
    }
}
