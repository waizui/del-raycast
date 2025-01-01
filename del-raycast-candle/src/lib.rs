macro_rules! get_cpu_slice_from_tensor {
    ($slc: ident, $sto: ident, $tnsr: expr, $t: ty) => {
        let $sto = $tnsr.storage_and_layout().0;
        let $slc = match $sto.deref() {
            Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<$t>()?,
            _ => panic!(),
        };
    };
}

#[cfg(feature = "cuda")]
macro_rules! get_cuda_slice_from_tensor {
    ($slc: ident, $storage: ident, $layout: ident, $tnsr: expr, $t: ty) => {
        let ($storage, $layout) = $tnsr.storage_and_layout();
        let $slc = match $storage.deref() {
            Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<$t>()?,
            _ => panic!(),
        };
    };
}

pub mod gd_with_laplacian_reparam;
pub mod perturb_tensor;
pub mod pix2depth;
pub mod pix2tri;
pub mod raycast_trimesh;
pub mod render_meshtri2_vtxcolor;
pub mod silhouette;
