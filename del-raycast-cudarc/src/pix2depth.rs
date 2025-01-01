use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
#[allow(clippy::too_many_arguments)]
pub fn pix2depth(
    device: &std::sync::Arc<CudaDevice>,
    img_shape: (usize, usize),
    pix2depth: &mut CudaSlice<f32>,
    pix2tri: &CudaSlice<u32>,
    tri2vtx: &CudaSlice<u32>,
    vtx2xyz: &CudaSlice<f32>,
    transform_ndc2world: &CudaSlice<f32>,
) -> std::result::Result<(), cudarc::driver::DriverError> {
    let transform_world2ndc = {
        let transform_ndc2world_cpu = device.dtoh_sync_copy(transform_ndc2world)?;
        let transform_ndc2world_cpu = arrayref::array_ref![&transform_ndc2world_cpu, 0, 16];
        let transform_world2ndc_cpu =
            del_geo_core::mat4_col_major::try_inverse(transform_ndc2world_cpu).unwrap();
        device.htod_sync_copy(&transform_world2ndc_cpu)?
    };
    let cfg = cudarc::driver::LaunchConfig::for_num_elems((img_shape.0 * img_shape.1) as u32);
    let num_tri = tri2vtx.len() / 3;
    let param = (
        pix2depth,
        pix2tri,
        num_tri as u32,
        tri2vtx,
        vtx2xyz,
        img_shape.0 as u32,
        img_shape.1 as u32,
        transform_ndc2world,
        &transform_world2ndc,
    );
    //unsafe { self.pix_to_tri.clone().launch(cfg,param) }.unwrap();
    let cuda_fn_pix2depth =
        del_cudarc::get_or_load_func(device, "pix2depth", del_raycast_cudarc_kernel::PIX2DEPTH)?;
    use cudarc::driver::LaunchAsync;
    unsafe { cuda_fn_pix2depth.launch(cfg, param) }?;
    Ok(())
}
