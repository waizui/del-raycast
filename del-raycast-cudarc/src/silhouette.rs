use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use del_cudarc::cudarc;

#[allow(clippy::too_many_arguments)]
pub fn wo_anti_alias(
    dev: &std::sync::Arc<CudaDevice>,
    img_shape: (usize, usize),
    pix2tri: &CudaSlice<u32>,
) -> std::result::Result<CudaSlice<f32>, cudarc::driver::DriverError> {
    let mut img = dev.alloc_zeros::<f32>(img_shape.1 * img_shape.0)?;
    del_cudarc::util::set_value_at_mask(dev, &mut img, 1f32, pix2tri, u32::MAX, false)?;
    Ok(img)
}
