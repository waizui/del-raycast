use candle_core::{CpuStorage, Layout, Shape, Tensor};
use std::ops::Deref;
// use std::time::Instant;

pub struct Layer {
    pub tri2vtx: Tensor,
    pub pix2tri: Tensor,
    pub img_shape: (usize, usize),      // (width, height)
    pub transform_nbc2world: [f32; 16], // transform column major
}

impl candle_core::CustomOp1 for Layer {
    fn name(&self) -> &'static str {
        "render"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let (_num_vtx, three) = layout.shape().dims2()?;
        assert_eq!(three, 3);
        let tri2vtx = self.tri2vtx.storage_and_layout().0;
        let tri2vtx = match tri2vtx.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<u32>()?,
            _ => panic!(),
        };
        let vtx2xyz = storage.as_slice::<f32>()?;
        let img2tri = self.pix2tri.storage_and_layout().0;
        let img2tri = match img2tri.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<u32>()?,
            _ => panic!(),
        };
        let mut pix2depth = vec![0f32; self.img_shape.0 * self.img_shape.1];
        for i_h in 0..self.img_shape.1 {
            for i_w in 0..self.img_shape.0 {
                let i_tri = img2tri[i_h * self.img_shape.0 + i_w];
                if i_tri == u32::MAX {
                    continue;
                }
                let (ray_org, ray_dir) = del_raycast_core::cam3::ray3_homogeneous(
                    (i_w, i_h),
                    &self.img_shape,
                    &self.transform_nbc2world,
                );
                let (p0, p1, p2) =
                    del_msh_core::trimesh3::to_corner_points(tri2vtx, vtx2xyz, i_tri as usize);
                let coeff = del_geo_core::tri3::intersection_against_line(
                    &p0, &p1, &p2, &ray_org, &ray_dir,
                )
                .unwrap();
                // let q = del_geo::vec3::axpy_(coeff, &ray_dir, &ray_org);
                pix2depth[i_h * self.img_shape.0 + i_w] =
                    (1. - coeff) * del_geo_core::vec3::norm(&ray_dir);
            }
        }
        let shape = candle_core::Shape::from((self.img_shape.1, self.img_shape.0));
        let storage = candle_core::WithDType::to_cpu_storage_owned(pix2depth);
        Ok((storage, shape))
    }

    /// This function takes as argument the argument `arg` used in the forward pass, the result
    /// produced by the forward operation `res` and the gradient of the result `grad_res`.
    /// The function should return the gradient of the argument.
    #[allow(clippy::identity_op)]
    fn bwd(
        &self,
        vtx2xyz: &Tensor,
        pix2depth: &Tensor,
        dw_pix2depth: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let (num_vtx, three) = vtx2xyz.shape().dims2()?;
        assert_eq!(three, 3);
        assert_eq!(pix2depth.shape(), dw_pix2depth.shape());
        let (height, width) = pix2depth.shape().dims2()?;
        let tri2vtx = self.tri2vtx.storage_and_layout().0;
        let tri2vtx = match tri2vtx.deref() {
            candle_core::Storage::Cpu(cpu_tri2vtx) => cpu_tri2vtx.as_slice::<u32>()?,
            _ => panic!(),
        };
        let vtx2xyz = vtx2xyz.storage_and_layout().0;
        let vtx2xyz = match vtx2xyz.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
            _ => panic!(),
        };
        let pix2tri = self.pix2tri.storage_and_layout().0;
        let pix2tri = match pix2tri.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<u32>()?,
            _ => panic!(),
        };
        let dw_pix2depth = dw_pix2depth.storage_and_layout().0;
        let dw_pix2depth = match dw_pix2depth.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
            _ => panic!(),
        };
        //
        let mut dw_vtx2xyz = vec![0f32; num_vtx * 3];
        for i_h in 0..height {
            for i_w in 0..width {
                let i_tri = pix2tri[i_h * self.img_shape.0 + i_w];
                if i_tri == u32::MAX {
                    continue;
                }
                let (ray_org, ray_dir) = del_raycast_core::cam3::ray3_homogeneous(
                    (i_w, i_h),
                    &self.img_shape,
                    &self.transform_nbc2world,
                );
                let i_tri = i_tri as usize;
                let (p0, p1, p2) =
                    del_msh_core::trimesh3::to_corner_points(tri2vtx, vtx2xyz, i_tri);
                let Some((_t, _u, _v, data)) = del_geo_nalgebra::tri3::ray_triangle_intersection(
                    &ray_org.into(),
                    &ray_dir.into(),
                    &p0.into(),
                    &p1.into(),
                    &p2.into(),
                ) else {
                    continue;
                };
                let dw_depth = dw_pix2depth[i_h * self.img_shape.0 + i_w];
                let (dw_p0, dw_p1, dw_p2) =
                    del_geo_nalgebra::tri3::dw_ray_triangle_intersection_(-dw_depth, 0., 0., &data);
                let scale = data.dir.norm();
                let dw_p0 = dw_p0 * scale;
                let dw_p1 = dw_p1 * scale;
                let dw_p2 = dw_p2 * scale;
                /*
                if dw_depth != 0.0 {
                    println!("{} {} {} {}", i_tri, _t, dw_p0+dw_p1+dw_p2, data.dir);
                }
                 */
                let iv0 = tri2vtx[i_tri * 3 + 0] as usize;
                let iv1 = tri2vtx[i_tri * 3 + 1] as usize;
                let iv2 = tri2vtx[i_tri * 3 + 2] as usize;
                dw_vtx2xyz[iv0 * 3..iv0 * 3 + 3]
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, v)| *v += dw_p0[i]);
                dw_vtx2xyz[iv1 * 3..iv1 * 3 + 3]
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, v)| *v += dw_p1[i]);
                dw_vtx2xyz[iv2 * 3..iv2 * 3 + 3]
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, v)| *v += dw_p2[i]);
            }
        }
        let dw_vtx2xyz = Tensor::from_vec(
            dw_vtx2xyz,
            candle_core::Shape::from((num_vtx, 3)),
            &candle_core::Device::Cpu,
        )?;
        Ok(Some(dw_vtx2xyz))
    }
}

#[test]
fn test_optimize_depth() -> anyhow::Result<()> {
    let (tri2vtx, vtx2xyz) = del_msh_core::trimesh3_primitive::sphere_yup::<u32, f32>(0.8, 32, 32);
    let vtx2xyz = {
        let mut vtx2xyz_new = vtx2xyz.clone();
        del_msh_core::vtx2xyz::translate_then_scale(
            &mut vtx2xyz_new,
            &vtx2xyz,
            &[0.2, 0.0, 0.0],
            1.0,
        );
        vtx2xyz_new
    };
    let num_tri = tri2vtx.len() / 3;
    let tri2vtx = Tensor::from_vec(tri2vtx, (num_tri, 3), &candle_core::Device::Cpu)?;
    let num_vtx = vtx2xyz.len() / 3;
    let vtx2xyz = candle_core::Var::from_vec(vtx2xyz, (num_vtx, 3), &candle_core::Device::Cpu)?;
    let img_shape = (200, 200);
    //
    let transform_ndc2world = del_geo_core::mat4_col_major::from_identity::<f32>();
    let (pix2depth_trg, pix2mask) = {
        let mut img2depth_trg = vec![0f32; img_shape.0 * img_shape.1];
        let mut img2mask = vec![0f32; img_shape.0 * img_shape.1];
        for i_h in 0..img_shape.1 {
            for i_w in 0..img_shape.0 {
                let (ray_org, _ray_dir) = del_raycast_core::cam3::ray3_homogeneous(
                    (i_w, i_h),
                    &img_shape,
                    &transform_ndc2world,
                );
                let x = ray_org[0];
                let y = ray_org[1];
                let r = (x * x + y * y).sqrt();
                if r > 0.5 {
                    continue;
                }
                img2depth_trg[i_h * img_shape.0 + i_w] = 0.6;
                img2mask[i_h * img_shape.0 + i_w] = 1.0;
            }
        }
        let img2depth_trg = Tensor::from_vec(img2depth_trg, img_shape, &candle_core::Device::Cpu)?;
        let img2mask = Tensor::from_vec(img2mask, img_shape, &candle_core::Device::Cpu)?;
        (img2depth_trg, img2mask)
    };
    {
        let pix2depth_trg = pix2depth_trg.flatten_all()?.to_vec1::<f32>()?;
        del_canvas::write_png_from_float_image_grayscale(
            "../target/pix2depth_trg.png",
            &img_shape,
            &pix2depth_trg,
        )?;
        //
        let pix2mask = pix2mask.flatten_all()?.to_vec1::<f32>()?;
        del_canvas::write_png_from_float_image_grayscale(
            "../target/pix2mask.png",
            &img_shape,
            &pix2mask,
        )?;
    }

    let mut optimizer = crate::gd_with_laplacian_reparam::Optimizer::new(
        vtx2xyz.clone(),
        0.001,
        tri2vtx.clone(),
        vtx2xyz.dims2()?.0,
        0.8,
    )?;

    // let mut optimizer = candle_nn::AdamW::new_lr(vec!(vtx2xyz.clone()), 0.01)?;

    for itr in 0..100 {
        let (bvhnodes, aabbs) = del_msh_candle::bvh::from_trimesh3(&tri2vtx, &vtx2xyz)?;
        let pix2tri = crate::raycast_trimesh::raycast3(
            &tri2vtx,
            &vtx2xyz,
            &bvhnodes,
            &aabbs,
            &img_shape,
            &transform_ndc2world,
        )?;
        let render = Layer {
            tri2vtx: tri2vtx.clone(),
            pix2tri: pix2tri.clone(),
            img_shape,
            transform_nbc2world: transform_ndc2world.clone(),
        };
        let pix2depth = vtx2xyz.apply_op1(render)?;
        dbg!(pix2depth.shape());
        let pix2diff = pix2depth.sub(&pix2depth_trg)?.mul(&pix2mask)?;
        {
            let pix2depth = pix2depth.flatten_all()?.to_vec1::<f32>()?;
            del_canvas::write_png_from_float_image_grayscale(
                "../target/pix2depth.png",
                &img_shape,
                &pix2depth,
            )?;
            let pix2diff = (pix2diff.clone() * 10.0)?
                .abs()?
                .flatten_all()?
                .to_vec1::<f32>()?;
            del_canvas::write_png_from_float_image_grayscale(
                "../target/pix2diff.png",
                &img_shape,
                &pix2diff,
            )?;
        }
        let loss = pix2diff.sqr()?.sum_all()?;
        println!("loss: {}", loss.to_vec0::<f32>()?);
        optimizer.step(&loss.backward()?)?;
        {
            let vtx2xyz = vtx2xyz.flatten_all()?.to_vec1::<f32>()?;
            let tri2vtx = tri2vtx.flatten_all()?.to_vec1::<u32>()?;
            del_msh_core::io_obj::save_tri2vtx_vtx2xyz(
                format!("../target/hoge_{}.obj", itr),
                &tri2vtx,
                &vtx2xyz,
                3,
            )?;
        }
    }
    Ok(())
}

pub fn render(
    tri2vtx: &Tensor,
    vtx2xyz: &Tensor,
    img_shape: &(usize, usize),
    transform_ndc2world: &[f32; 16],
) -> candle_core::Result<Tensor> {
    // let time0 = Instant::now();
    let (bvhnodes, aabbs) = del_msh_candle::bvh::from_trimesh3(tri2vtx, vtx2xyz)?;
    // println!("      time for bvh: {:.2?}", time0.elapsed());
    // let time0 = Instant::now();
    let pix2tri = crate::raycast_trimesh::raycast3(
        tri2vtx,
        vtx2xyz,
        &bvhnodes,
        &aabbs,
        img_shape,
        transform_ndc2world,
    )?;
    // println!("      time for ray_cast: {:.2?}", time0.elapsed());
    // let time0 = Instant::now();
    // render depth
    let render = Layer {
        tri2vtx: tri2vtx.clone(),
        pix2tri: pix2tri.clone(),
        img_shape: *img_shape,
        transform_nbc2world: *transform_ndc2world,
    };
    vtx2xyz.apply_op1(render)
    // println!("      time for render depth: {:.2?}", time0.elapsed());
    // let time0 = Instant::now();
    // res
}
