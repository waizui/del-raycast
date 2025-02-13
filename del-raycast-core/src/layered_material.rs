use crate::material::CoatedDiffuse;

pub fn sample_brdf_coated_diffuse<RNG>(
    wo: &[f32; 3],
    nsamples: i32,
    mdepth: i32,
    thickness: f32,
    mat: &CoatedDiffuse,
    rng: &mut RNG,
) -> Option<([f32; 3], [f32; 3], f32)>
where
    RNG: rand::Rng,
{
    let mut m_pdf = 0.;
    for isample in 0..nsamples {
        for idepth in 0..mdepth {
            todo!();
        }
    }

    // refl, brdf, pdf
    todo!()
}

pub fn eval_brdf_coated_diffuse<RNG>(
    wi: &[f32; 3],
    wo: &[f32; 3],
    nsamples: i32,
    mdepth: i32,
    thickness: f32,
    mat: &CoatedDiffuse,
    rng: &mut RNG,
) -> [f32; 3]
where
    RNG: rand::Rng,
{
    todo!()
}
