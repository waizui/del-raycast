use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};

#[allow(clippy::too_many_arguments)]
pub fn pix2tri(
    dev: &std::sync::Arc<CudaDevice>,
    img_shape: (usize, usize),
    pix2tri: &mut CudaSlice<u32>,
    tri2vtx: &CudaSlice<u32>,
    vtx2xyz: &CudaSlice<f32>,
    bvhnodes: &CudaSlice<u32>,
    bvhnode2aabb: &CudaSlice<f32>,
    transform_ndc2world: &CudaSlice<f32>,
) -> std::result::Result<(), cudarc::driver::DriverError> {
    let cfg = cudarc::driver::LaunchConfig::for_num_elems((img_shape.0 * img_shape.1) as u32);
    let num_tri = tri2vtx.len() / 3;
    let param = (
        pix2tri,
        num_tri as u32,
        tri2vtx,
        vtx2xyz,
        img_shape.0 as u32,
        img_shape.1 as u32,
        transform_ndc2world,
        bvhnodes,
        bvhnode2aabb,
    );
    //unsafe { self.pix_to_tri.clone().launch(cfg,param) }.unwrap();
    let pix_to_tri =
        del_cudarc::get_or_load_func(dev, "pix_to_tri", del_raycast_cudarc_kernel::PIX2TRI)?;
    use cudarc::driver::LaunchAsync;
    unsafe { pix_to_tri.launch(cfg, param) }?;
    Ok(())
}
