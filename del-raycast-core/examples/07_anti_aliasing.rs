fn main() -> anyhow::Result<()> {
    let (tri2vtx, vtx2xyz) = del_msh_core::trimesh3_primitive::sphere_yup::<u32, f32>(0.8, 64, 64);
    let bvhnodes = del_msh_core::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
    let bvhnode2aabb = del_msh_core::bvhnode2aabb3::from_uniform_mesh_with_bvh(
        0,
        &bvhnodes,
        Some((&tri2vtx, 3)),
        &vtx2xyz,
        None,
    );
    //
    let img_shape = {
        const TILE_SIZE: usize = 16;
        (TILE_SIZE * 12, TILE_SIZE * 12)
    };
    let cam_projection = del_geo_core::mat4_col_major::camera_perspective_blender(
        img_shape.0 as f32 / img_shape.1 as f32,
        24f32,
        0.3,
        10.0,
        true,
    );
    let cam_modelview =
        del_geo_core::mat4_col_major::camera_external_blender(&[0., 0., 2.0], 0., 0., 0.);
    let transform_world2ndc =
        del_geo_core::mat4_col_major::mult_mat(&cam_projection, &cam_modelview);
    let transform_ndc2world =
        del_geo_core::mat4_col_major::try_inverse(&transform_world2ndc).unwrap();

    let pix2tri = del_raycast_core::raycast_trimesh3::pix2tri(
        &tri2vtx,
        &vtx2xyz,
        &bvhnodes,
        &bvhnode2aabb,
        &img_shape,
        &transform_ndc2world,
    );

    {
        let img_out = del_raycast_core::raycast_trimesh3::render_normalmap_from_pix2tri(
            img_shape,
            &cam_modelview,
            &tri2vtx,
            &vtx2xyz,
            &pix2tri,
        );
        del_canvas::write_png_from_float_image_rgb(
            "target/07_anti_aliasing0.png",
            &img_shape,
            &img_out,
        )?;
    }

    {
        let transform_ndc2pix = del_geo_core::mat3_col_major::from_transform_ndc2pix(img_shape);
        // look horizontal
        let mut img_out = vec![0f32; img_shape.0 * img_shape.1 * 3];
        for ih in 0..img_shape.1 {
            for iw0 in 0..img_shape.0 - 1 {
                let iw1 = iw0 + 1;
                let itri0 = pix2tri[ih * img_shape.0 + iw0];
                let itri1 = pix2tri[ih * img_shape.0 + iw1];
                if itri0 != itri1 && (itri0 == u32::MAX || itri1 == u32::MAX) {
                    let c0 = [iw0 as f32 + 0.5, ih as f32 + 0.5];
                    let c1 = [iw1 as f32 + 0.5, ih as f32 + 0.5];
                    let (itri, which_in) = if itri0 == u32::MAX {
                        (itri1, 1)
                    } else {
                        (itri0, 0)
                    };
                    let tri = del_msh_core::trimesh3::to_tri3(&tri2vtx, &vtx2xyz, itri as usize);
                    let q0 = del_geo_core::mat4_col_major::transform_homogeneous(
                        &transform_world2ndc,
                        tri.p0,
                    )
                    .unwrap();
                    let q1 = del_geo_core::mat4_col_major::transform_homogeneous(
                        &transform_world2ndc,
                        tri.p1,
                    )
                    .unwrap();
                    let q2 = del_geo_core::mat4_col_major::transform_homogeneous(
                        &transform_world2ndc,
                        tri.p2,
                    )
                    .unwrap();
                    let r0 = del_geo_core::mat3_col_major::transform_homogeneous(
                        &transform_ndc2pix,
                        &[q0[0], q0[1]],
                    )
                    .unwrap();
                    let r1 = del_geo_core::mat3_col_major::transform_homogeneous(
                        &transform_ndc2pix,
                        &[q1[0], q1[1]],
                    )
                    .unwrap();
                    let r2 = del_geo_core::mat3_col_major::transform_homogeneous(
                        &transform_ndc2pix,
                        &[q2[0], q2[1]],
                    )
                    .unwrap();
                    let flg0 = del_geo_core::tri2::is_inside(&r0, &r1, &r2, &c0, -1.0);
                    let flg1 = del_geo_core::tri2::is_inside(&r0, &r1, &r2, &c1, -1.0);
                    if which_in == 0 {
                        assert!(flg0.is_some() && flg1.is_none());
                        let e0 = del_geo_core::edge2::intersection_edge2(&c0, &c1, &r1, &r2);
                        let e1 = del_geo_core::edge2::intersection_edge2(&c0, &c1, &r2, &r0);
                        let e2 = del_geo_core::edge2::intersection_edge2(&c0, &c1, &r0, &r1);
                        if let Some((rc, re)) = e0 {
                            assert!(rc >= 0. && rc <= 1.0);
                            if rc < 0.5 {
                                img_out[(ih * img_shape.0 + iw0) * 3] = 0.5 + rc;
                            } else {
                                img_out[(ih * img_shape.0 + iw0) * 3] = 1f32;
                                img_out[(ih * img_shape.0 + iw1) * 3] = rc - 0.5;
                            }
                        } else if let Some((rc, re)) = e1 {
                            assert!(rc >= 0. && rc <= 1.0);
                            if rc < 0.5 {
                                img_out[(ih * img_shape.0 + iw0) * 3] = 0.5 + rc;
                            } else {
                                img_out[(ih * img_shape.0 + iw0) * 3] = 1f32;
                                img_out[(ih * img_shape.0 + iw1) * 3] = rc - 0.5;
                            }
                        } else if let Some((rc, re)) = e2 {
                            assert!(rc >= 0. && rc <= 1.0);
                            if rc < 0.5 {
                                img_out[(ih * img_shape.0 + iw0) * 3] = 0.5 + rc;
                            } else {
                                img_out[(ih * img_shape.0 + iw0) * 3] = 1f32;
                                img_out[(ih * img_shape.0 + iw1) * 3] = rc - 0.5;
                            }
                        }
                    } else if which_in == 1 {
                        assert!(flg1.is_some() && flg0.is_none());
                    }
                }
            }
        }
        del_canvas::write_png_from_float_image_rgb(
            "target/07_anti_aliasing1.png",
            &img_shape,
            &img_out,
        )?;
    }

    Ok(())
}
