#[allow(unused_imports)]
use candle_core::{backend::BackendDevice, CudaStorage};
use candle_core::{CpuStorage, Layout, Storage, Tensor};
use std::ops::Deref;

pub struct Pix2Tri {
    pub bvhnodes: Tensor,
    pub bvhnode2aabb: Tensor,
    pub transform_ndc2world: Tensor,
}

impl candle_core::InplaceOp3 for Pix2Tri {
    fn name(&self) -> &'static str {
        "pix2tri"
    }
    fn cpu_fwd(
        &self,
        pix2tri: &mut CpuStorage,
        l_pix2tri: &Layout,
        tri2vtx: &CpuStorage,
        l_tri2vtx: &Layout,
        vtx2xyz: &CpuStorage,
        l_vtx2xyz: &Layout,
    ) -> candle_core::Result<()> {
        assert_eq!(l_tri2vtx.dim(1)?, 3);
        let num_tri = l_tri2vtx.dim(0)?;
        let num_dim = l_vtx2xyz.dim(1)?;
        assert_eq!(num_dim, 3); // todo: implement num_dim == 2
        assert_eq!(self.bvhnodes.dims2()?, (num_tri * 2 - 1, 3));
        assert_eq!(self.bvhnode2aabb.dims2()?, (num_tri * 2 - 1, 6));
        let img_shape = (l_pix2tri.dim(0)?, l_pix2tri.dim(1)?);
        let pix2tri = match pix2tri {
            CpuStorage::U32(v) => v,
            _ => panic!(),
        };
        let tri2vtx = match tri2vtx {
            CpuStorage::U32(v) => v,
            _ => panic!(),
        };
        let vtx2xyz = match vtx2xyz {
            CpuStorage::F32(v) => v,
            _ => panic!(),
        };
        let bvhnodes = self.bvhnodes.storage_and_layout().0;
        let bvhnodes = match bvhnodes.deref() {
            Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<u32>()?,
            _ => panic!(),
        };
        let bvhnode2aabb = self.bvhnode2aabb.storage_and_layout().0;
        let bvhnode2aabb = match bvhnode2aabb.deref() {
            Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
            _ => panic!(),
        };
        let storage = self.transform_ndc2world.storage_and_layout().0;
        let transform_ndc2world = match storage.deref() {
            Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
            _ => panic!(),
        };
        let transform_ndc2world = arrayref::array_ref!(transform_ndc2world, 0, 16);
        del_raycast_core::raycast_trimesh3::update_pix2tri(
            pix2tri,
            tri2vtx,
            vtx2xyz,
            bvhnodes,
            bvhnode2aabb,
            img_shape,
            transform_ndc2world,
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        pix2tri: &mut CudaStorage,
        l_pix2tri: &Layout,
        tri2vtx: &CudaStorage,
        l_tri2vtx: &Layout,
        vtx2xyz: &CudaStorage,
        l_vtx2xyz: &Layout,
    ) -> candle_core::Result<()> {
        use candle_core::cuda::CudaStorageSlice;
        assert_eq!(l_tri2vtx.dim(1)?, 3);
        assert_eq!(l_vtx2xyz.dim(1)?, 3); // todo: implement 2D
        use candle_core::cuda_backend::WrapErr;
        let img_shape = (l_pix2tri.dim(0)?, l_pix2tri.dim(1)?);
        //
        let CudaStorage { slice, device } = pix2tri;
        let (pix2tri, device_pix2tri) = match slice {
            CudaStorageSlice::U32(slice) => (slice, device),
            _ => panic!(),
        };
        //
        let CudaStorage { slice, device } = tri2vtx;
        let (tri2vtx, device_tri2vtx) = match slice {
            CudaStorageSlice::U32(slice) => (slice, device),
            _ => panic!(),
        };
        //
        let CudaStorage { slice, device } = vtx2xyz;
        let (vtx2xyz, device_vtx2xyz) = match slice {
            CudaStorageSlice::F32(slice) => (slice, device),
            _ => panic!(),
        };
        assert!(device_pix2tri.same_device(device_tri2vtx));
        assert!(device_pix2tri.same_device(device_vtx2xyz));
        //
        let storage = self.bvhnodes.storage_and_layout().0;
        let bvhnodes = match storage.deref() {
            Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<u32>()?,
            _ => panic!(),
        };
        //
        let storage = self.bvhnode2aabb.storage_and_layout().0;
        let bvhnode2aabb = match storage.deref() {
            Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<f32>()?,
            _ => panic!(),
        };
        //
        let storage = self.transform_ndc2world.storage_and_layout().0;
        let transform_ndc2world = match storage.deref() {
            Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<f32>()?,
            _ => panic!(),
        };
        del_raycast_cudarc::pix2tri::pix2tri(
            device,
            img_shape,
            pix2tri,
            tri2vtx,
            vtx2xyz,
            bvhnodes,
            bvhnode2aabb,
            transform_ndc2world,
        )
        .w()?;
        Ok(())
    }
}

pub fn from_trimesh3(
    tri2vtx: &Tensor,
    vtx2xyz: &Tensor,
    bvhnodes: &Tensor,
    bvhnode2aabb: &Tensor,
    img_shape: (usize, usize),    // (width, height)
    transform_ndc2world: &Tensor, // transform column major
) -> candle_core::Result<Tensor> {
    let device = tri2vtx.device();
    let pix2tri = Tensor::zeros(img_shape, candle_core::DType::U32, device)?;
    let layer = crate::pix2tri::Pix2Tri {
        bvhnodes: bvhnodes.clone(),
        bvhnode2aabb: bvhnode2aabb.clone(),
        transform_ndc2world: transform_ndc2world.clone(),
    };
    pix2tri.inplace_op3(tri2vtx, vtx2xyz, &layer)?;
    Ok(pix2tri)
}
