#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/pair.h>
#include "tri3.h"
#include "vec3.h"
#include "mat4_col_major.h"
#include "ray_for_pixel.h"

extern "C" {
__global__
void pix2depth(
  float* pix2depth,
  const uint32_t *pix2tri,
  const uint32_t num_tri,
  const uint32_t *tri2vtx,
  const float *vtx2xyz,
  const uint32_t img_w,
  const uint32_t img_h,
  const float *transform_ndc2world,
  const float *transform_world2ndc) {
    int i_pix = blockDim.x * blockIdx.x + threadIdx.x;
    if( i_pix >= img_w * img_h ){ return; }
    //
    auto ray = ray_for_pixel(i_pix, img_w, img_h, transform_ndc2world);
    const uint32_t i_tri = pix2tri[i_pix];
    if( i_tri == UINT32_MAX ){
        pix2depth[i_pix] = 0.f;
        return;
    }
    const uint32_t iv0 = tri2vtx[i_tri*3+0];
    const uint32_t iv1 = tri2vtx[i_tri*3+1];
    const uint32_t iv2 = tri2vtx[i_tri*3+2];
    const float* p0 = vtx2xyz + iv0*3;
    const float* p1 = vtx2xyz + iv1*3;
    const float* p2 = vtx2xyz + iv2*3;
    auto opt_depth = tri3::intersection_against_ray(p0, p1, p2, ray.first.data(), ray.second.data());
    if(!opt_depth){ return; }
    float depth = opt_depth.value();
    const auto pos_world = vec3::axpy(depth, ray.second.data(), ray.first.data());
    const auto pos_ndc = mat4_col_major::transform_homogeneous(transform_world2ndc, pos_world.data());
    const float depth_ndc = (pos_ndc[2] + 1.f) * 0.5f;
    pix2depth[i_pix] = depth_ndc;

  }
}