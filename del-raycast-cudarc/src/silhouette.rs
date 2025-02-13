use cudarc::driver::{CudaDevice, CudaSlice};
use del_cudarc::cudarc;
use del_cudarc::cudarc::driver::DeviceSlice;

pub fn compute_with_alias(
    dev: &std::sync::Arc<CudaDevice>,
    img_shape: (usize, usize),
    pix2tri: &CudaSlice<u32>,
) -> std::result::Result<CudaSlice<f32>, cudarc::driver::DriverError> {
    let mut img = dev.alloc_zeros::<f32>(img_shape.1 * img_shape.0)?;
    del_cudarc::util::set_value_at_mask(dev, &mut img, 1f32, pix2tri, u32::MAX, false)?;
    Ok(img)
}

pub fn remove_alias(
    dev: &std::sync::Arc<CudaDevice>,
    edge2vtx: &CudaSlice<u32>,
    img_shape: (usize, usize),
    pix2occu: &mut CudaSlice<f32>,
    pix2tri: &CudaSlice<u32>,
    vtx2xyz: &CudaSlice<f32>,
    transform_world2pix: &CudaSlice<f32>,
) -> std::result::Result<(), cudarc::driver::DriverError> {
    let num_edge = edge2vtx.len() / 2;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_edge as u32);
    let param = (
        num_edge as u32,
        edge2vtx,
        img_shape.0 as u32,
        img_shape.1 as u32,
        pix2occu,
        pix2tri,
        vtx2xyz,
        transform_world2pix,
    );
    let func = del_cudarc::get_or_load_func(dev, "fwd", del_raycast_cudarc_kernel::SILHOUETTE)?;
    use cudarc::driver::LaunchAsync;
    unsafe { func.launch(cfg, param) }?;
    Ok(())
}
