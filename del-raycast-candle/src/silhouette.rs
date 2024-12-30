#[allow(unused_imports)]
use candle_core::{DType, Device, Tensor, Var};
use std::ops::Deref;

pub fn anti_aliased_silhouette_update_image(
    edge2vtx_contour: &Tensor,
    vtx2xyz: &Tensor,
    transform_world2pix: &[f32; 16],
    pix2tri: &Tensor,
) -> anyhow::Result<Tensor> {
    let edge2vtx_contour = edge2vtx_contour.storage_and_layout().0;
    let edge2vtx_contour = match edge2vtx_contour.deref() {
        candle_core::Storage::Cpu(cpu_edge2vtx_contour) => cpu_edge2vtx_contour.as_slice::<u32>(),
        _ => panic!(),
    }?;
    //
    let vtx2xyz = vtx2xyz.storage_and_layout().0;
    let vtx2xyz = match vtx2xyz.deref() {
        candle_core::Storage::Cpu(cpu_vtx2xyz) => cpu_vtx2xyz.as_slice::<f32>(),
        _ => panic!(),
    }?;
    //
    let img_shape = pix2tri.dims2()?;
    let pix2tri = pix2tri.storage_and_layout().0;
    let pix2tri = match pix2tri.deref() {
        candle_core::Storage::Cpu(cpu_pix2tri) => cpu_pix2tri.as_slice::<u32>(),
        _ => panic!(),
    }?;
    //
    let mut img = vec![0f32; img_shape.0 * img_shape.1];
    del_raycast_core::anti_aliased_silhouette::update_image(
        edge2vtx_contour,
        vtx2xyz,
        transform_world2pix,
        img_shape,
        &mut img,
        pix2tri,
    );
    let img = Tensor::from_vec(img, img_shape, &candle_core::Device::Cpu)?;
    Ok(img)
}

#[test]
fn test_cpu() -> anyhow::Result<()> {
    let (tri2vtx, vtx2xyz) = {
        let (tri2vtx, vtx2xyz) =
            del_msh_core::trimesh3_primitive::sphere_yup::<u32, f32>(0.8, 64, 64);
        let num_tri = tri2vtx.len() / 3;
        let tri2vtx = Tensor::from_vec(tri2vtx, (num_tri, 3), &candle_core::Device::Cpu)?;
        let num_vtx = vtx2xyz.len() / 3;
        let vtx2xyz = Var::from_vec(vtx2xyz, (num_vtx, 3), &candle_core::Device::Cpu)?;
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
    let img_asp = 1.0;
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
    let pix2tri = crate::raycast_trimesh::pix2tri_for_trimesh3(
        &tri2vtx,
        &vtx2xyz,
        &bvhnodes,
        &bvhnode2aabb,
        img_shape,
        &transform_ndc2world,
    )?;
    let edge2vtx_contour =
        del_msh_candle::edge2vtx_trimesh3::contour(&tri2vtx, &vtx2xyz, &transform_world2ndc)?;
    let img = anti_aliased_silhouette_update_image(
        &edge2vtx_contour,
        &vtx2xyz,
        &transform_world2pix,
        &pix2tri,
    )?;
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
