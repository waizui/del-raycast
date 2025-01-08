use candle_core::{CpuStorage, Layout, Shape};
#[allow(unused_imports)]
use candle_core::{DType, Device, Tensor, Var};
use std::ops::Deref;

pub struct AntiAliasSilhouette {
    pix2tri: Tensor,
    edge2vtx_contour: Tensor,
    transform_world2pix: Tensor,
}

impl candle_core::CustomOp1 for AntiAliasSilhouette {
    fn name(&self) -> &'static str {
        "anti_alias_silhouette"
    }

    fn cpu_fwd(
        &self,
        vtx2xyz: &CpuStorage,
        l_vtx2xyz: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        assert_eq!(l_vtx2xyz.dim(1)?, 3);
        let img_shape = (self.pix2tri.dim(1)?, self.pix2tri.dim(0)?);
        get_cpu_slice_and_storage_from_tensor!(
            edge2vtx_contour,
            _s_edge2vtx_contour,
            self.edge2vtx_contour,
            u32
        );
        get_cpu_slice_and_storage_from_tensor!(pix2tri, _s_pix2tri, self.pix2tri, u32);
        get_cpu_slice_and_storage_from_tensor!(
            transform_world2pix,
            _s_transform_world2pix,
            self.transform_world2pix,
            f32
        );
        let transform_world2pix = arrayref::array_ref![transform_world2pix, 0, 16];
        let vtx2xyz = vtx2xyz.as_slice()?;
        let mut img = vec![0f32; img_shape.0 * img_shape.1];
        del_raycast_core::anti_aliased_silhouette::update_image(
            edge2vtx_contour,
            vtx2xyz,
            transform_world2pix,
            img_shape,
            &mut img,
            pix2tri,
        );
        Ok((CpuStorage::F32(img), (img_shape.1, img_shape.0).into()))
    }
}

#[test]
fn test_cpu() -> anyhow::Result<()> {
    let (tri2vtx, vtx2xyz) = {
        let (tri2vtx, vtx2xyz) =
            del_msh_core::trimesh3_primitive::sphere_yup::<u32, f32>(0.8, 64, 64);
        let num_tri = tri2vtx.len() / 3;
        let tri2vtx = Tensor::from_vec(tri2vtx, (num_tri, 3), &Device::Cpu)?;
        let num_vtx = vtx2xyz.len() / 3;
        let vtx2xyz = Var::from_vec(vtx2xyz, (num_vtx, 3), &Device::Cpu)?;
        (tri2vtx, vtx2xyz)
    };
    let (bvhnodes, bvhnode2aabb) = {
        let bvhdata = del_msh_candle::bvhnode2aabb::BvhForTriMesh::new(
            tri2vtx.dims2()?.0,
            vtx2xyz.dims2()?.1,
            &Device::Cpu,
        )?;
        bvhdata.compute(&tri2vtx, &vtx2xyz)?;
        (bvhdata.bvhnodes, bvhdata.bvhnode2aabb)
    };
    let img_asp = 1.5;
    let img_shape = (((16 * 6) as f32 * img_asp) as usize, 16 * 6);
    let cam_projection =
        del_geo_core::mat4_col_major::camera_perspective_blender(img_asp, 24f32, 0.3, 10.0, true);
    let cam_modelview =
        del_geo_core::mat4_col_major::camera_external_blender(&[0., 0., 2.0], 0., 0., 0.);
    // ----------------------
    let transform_world2ndc =
        del_geo_core::mat4_col_major::mult_mat(&cam_projection, &cam_modelview);
    let transform_ndc2world =
        del_geo_core::mat4_col_major::try_inverse(&transform_world2ndc).unwrap();
    let transform_world2pix = {
        let transform_ndc2pix = del_geo_core::mat3_col_major::from_transform_ndc2pix(img_shape);
        let transform_ndc2pix =
            del_geo_core::mat4_col_major::from_mat3_col_major_adding_z(&transform_ndc2pix);
        del_geo_core::mat4_col_major::mult_mat(&transform_ndc2pix, &transform_world2ndc)
    };
    let transform_ndc2world = Tensor::from_vec(transform_ndc2world.to_vec(), 16, &Device::Cpu)?;
    let transform_world2pix = Tensor::from_vec(transform_world2pix.to_vec(), 16, &Device::Cpu)?;
    let pix2tri = crate::pix2tri::from_trimesh3(
        &tri2vtx,
        &vtx2xyz,
        &bvhnodes,
        &bvhnode2aabb,
        img_shape,
        &transform_ndc2world,
    )?;
    let edge2vtx_contour =
        del_msh_candle::edge2vtx_trimesh3::contour(&tri2vtx, &vtx2xyz, &transform_world2ndc)?;
    let layer = AntiAliasSilhouette {
        edge2vtx_contour: edge2vtx_contour.clone(),
        pix2tri: pix2tri.clone(),
        transform_world2pix: transform_world2pix.clone(),
    };
    let img = vtx2xyz.apply_op1(layer)?;
    del_canvas::write_png_from_float_image_grayscale(
        "../target/del-raycast-candle__silhouette.png",
        img_shape,
        &img.flatten_all()?.to_vec1::<f32>()?,
    )?;
    #[cfg(feature = "cuda")]
    {
        let device = Device::cuda_if_available(0)?;
        let tri2vtx = tri2vtx.to_device(&device)?;
        let vtx2xyz = vtx2xyz.to_device(&device)?;
        let transform_ndc2world = transform_ndc2world.to_device(&device)?;
        let bvhdata =
            del_msh_candle::bvhnode2aabb::BvhForTriMesh::from_trimesh(&tri2vtx, &vtx2xyz)?;
        let pix2tri_cpu = pix2tri.flatten_all()?.to_vec1::<u32>()?;
        let pix2tri = Tensor::zeros(img_shape, DType::U32, &device)?;
        let layer = crate::pix2tri::Pix2Tri {
            bvhnodes: bvhdata.bvhnodes,
            bvhnode2aabb: bvhdata.bvhnode2aabb,
            transform_ndc2world,
        };
        pix2tri.inplace_op3(&tri2vtx, &vtx2xyz, &layer)?;
        let pix2tri_gpu = pix2tri.flatten_all()?.to_vec1::<u32>()?;
        pix2tri_cpu
            .iter()
            .zip(pix2tri_gpu.iter())
            .for_each(|(&a, &b)| {
                assert_eq!(a, b);
            })
    }
    Ok(())
}
