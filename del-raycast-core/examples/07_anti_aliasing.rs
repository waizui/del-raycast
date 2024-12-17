use del_geo_core::vec2::Vec2;
use del_geo_core::vec3::Vec3;

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
    let img_asp = 1.0;
    let img_shape_lowres = (((16 * 4) as f32 * img_asp) as usize, 16 * 4);
    let cam_projection =
        del_geo_core::mat4_col_major::camera_perspective_blender(img_asp, 24f32, 0.3, 10.0, true);
    let cam_modelview =
        del_geo_core::mat4_col_major::camera_external_blender(&[0., 0., 2.0], 0., 0., 0.);

    // ----------------------

    let transform_world2ndc =
        del_geo_core::mat4_col_major::mult_mat(&cam_projection, &cam_modelview);
    let transform_ndc2world =
        del_geo_core::mat4_col_major::try_inverse(&transform_world2ndc).unwrap();
    let transform_world2pix_lowres = {
        let transform_ndc2pix =
            del_geo_core::mat3_col_major::from_transform_ndc2pix(img_shape_lowres);
        let transform_ndc2pix =
            del_geo_core::mat4_col_major::from_mat3_col_major_adding_z(&transform_ndc2pix);
        del_geo_core::mat4_col_major::mult_mat(&transform_ndc2pix, &transform_world2ndc)
    };

    let pix2tri = del_raycast_core::raycast_trimesh3::pix2tri(
        &tri2vtx,
        &vtx2xyz,
        &bvhnodes,
        &bvhnode2aabb,
        &img_shape_lowres,
        &transform_ndc2world,
    );

    {
        let img_out = del_raycast_core::raycast_trimesh3::render_normalmap_from_pix2tri(
            img_shape_lowres,
            &cam_modelview,
            &tri2vtx,
            &vtx2xyz,
            &pix2tri,
        );
        del_canvas::write_png_from_float_image_rgb(
            "target/07_anti_aliasing_normalmap.png",
            &img_shape_lowres,
            &img_out,
        )?;
    }

    let edge2vtx_contour = {
        let edge2vtx = del_msh_core::edge2vtx::from_triangle_mesh(&tri2vtx, vtx2xyz.len() / 3);
        let edge2tri = del_msh_core::edge2elem::from_edge2vtx_of_tri2vtx(
            &edge2vtx,
            &tri2vtx,
            vtx2xyz.len() / 3,
        );
        del_msh_core::edge2vtx::contour_for_triangle_mesh(
            &tri2vtx,
            &vtx2xyz,
            &transform_world2ndc,
            &edge2vtx,
            &edge2tri,
        )
    };

    let mut img_lowres = vec![0f32; img_shape_lowres.0 * img_shape_lowres.1 * 3];
    for node2vtx in edge2vtx_contour.chunks(2) {
        let (i0_vtx, i1_vtx) = (node2vtx[0], node2vtx[1]);
        let p0 = del_msh_core::vtx2xyz::to_xyz(&vtx2xyz, i0_vtx as usize).p;
        let p1 = del_msh_core::vtx2xyz::to_xyz(&vtx2xyz, i1_vtx as usize).p;
        let q0 = p0
            .transform_homogeneous(&transform_world2pix_lowres)
            .unwrap()
            .xy();
        let q1 = p1
            .transform_homogeneous(&transform_world2pix_lowres)
            .unwrap()
            .xy();
        let v01 = q1.sub(&q0);
        let is_horizontal = v01[0].abs() < v01[1].abs();
        let list_pix = del_geo_core::edge2::overlapping_pixels_dda(img_shape_lowres, &q0, &q1);
        for &i_pix in list_pix.iter() {
            let (iw1, ih1) = (i_pix % img_shape_lowres.0, i_pix / img_shape_lowres.0);
            if is_horizontal {
                let is_in0 = if iw1 != 0 {
                    pix2tri[ih1 * img_shape_lowres.0 + iw1 - 1] != u32::MAX
                } else {
                    false
                };
                let is_in1 = pix2tri[ih1 * img_shape_lowres.0 + iw1] != u32::MAX;
                let is_in2 = if iw1 != img_shape_lowres.0 - 1 {
                    pix2tri[ih1 * img_shape_lowres.0 + iw1 + 1] != u32::MAX
                } else {
                    false
                };
                let c0 = [iw1 as f32 - 0.5, ih1 as f32 + 0.5];
                let c1 = [iw1 as f32 + 0.5, ih1 as f32 + 0.5];
                let c2 = [iw1 as f32 + 1.5, ih1 as f32 + 0.5];
                if is_in0 && !is_in1 {
                    if let Some((rc, re)) =
                        del_geo_core::edge2::intersection_edge2(&c0, &c1, &q1, &q0)
                    {
                        assert!(rc >= 0. && rc <= 1.0);
                        let rc = 1.0 - rc;
                        if rc < 0.5 {
                            img_lowres[(ih1 * img_shape_lowres.0 + iw1 - 1) * 3] = 0.5 + rc;
                        } else {
                            img_lowres[(ih1 * img_shape_lowres.0 + iw1 - 1) * 3 + 1] = 1f32;
                            img_lowres[(ih1 * img_shape_lowres.0 + iw1) * 3 + 1] = rc - 0.5;
                        }
                    }
                }
                if is_in1 && !is_in2 {
                    if let Some((rc, re)) =
                        del_geo_core::edge2::intersection_edge2(&c1, &c2, &q0, &q1)
                    {
                        assert!(rc >= 0. && rc <= 1.0);
                        let rc = 1.0 - rc;
                        if rc < 0.5 {
                            img_lowres[(ih1 * img_shape_lowres.0 + iw1) * 3] = 0.5 + rc;
                        } else {
                            img_lowres[(ih1 * img_shape_lowres.0 + iw1) * 3 + 1] = 1f32;
                            img_lowres[(ih1 * img_shape_lowres.0 + iw1 + 1) * 3 + 1] = rc - 0.5;
                        }
                    }
                }
            } else {
            }
        }
    }
    del_canvas::write_png_from_float_image_rgb(
        "target/07_anti_aliasing1.png",
        &img_shape_lowres,
        &img_lowres,
    )?;

    /*
    {
        // look horizontal
        let mut img_out = vec![0f32; img_shape_lowres.0 * img_shape_lowres.1 * 3];
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
                    use del_geo_core::vec3::Vec3;
                    let r0 = tri
                        .p0
                        .transform_homogeneous(&transform_world2pix)
                        .unwrap()
                        .xy();
                    let r1 = tri
                        .p1
                        .transform_homogeneous(&transform_world2pix)
                        .unwrap()
                        .xy();
                    let r2 = tri
                        .p2
                        .transform_homogeneous(&transform_world2pix)
                        .unwrap()
                        .xy();
                    dbg!(r0, r1, r2, c0, c1);
                    let flg0 = del_geo_core::tri2::is_inside(&r0, &r1, &r2, &c0, -1.0);
                    let flg1 = del_geo_core::tri2::is_inside(&r0, &r1, &r2, &c1, -1.0);
                    if which_in == 0 {
                        // assert!(flg0.is_some() && flg1.is_none());
                        let e0 = del_geo_core::edge2::intersection_edge2(&c0, &c1, &r1, &r2);
                        let e1 = del_geo_core::edge2::intersection_edge2(&c0, &c1, &r2, &r0);
                        let e2 = del_geo_core::edge2::intersection_edge2(&c0, &c1, &r0, &r1);
                        dbg!((e0.is_some(), e1.is_some(), e2.is_some()));
                        if let Some((rc, re)) = e0 {
                            assert!(rc >= 0. && rc <= 1.0);
                            let rc = 1.0 - rc;
                            dbg!((rc, iw0, ih));
                            if rc < 0.5 {
                                dbg!("hoge0");
                                img_out[(ih * img_shape.0 + iw0) * 3] = 0.5 + rc;
                            } else {
                                img_out[(ih * img_shape.0 + iw0) * 3 + 1] = 1f32;
                                img_out[(ih * img_shape.0 + iw1) * 3 + 1] = rc - 0.5;
                            }
                        } else if let Some((rc, re)) = e1 {
                            let rc = 1.0 - rc;
                            assert!(rc >= 0. && rc <= 1.0);
                            dbg!((rc, iw0, ih));
                            if rc < 0.5 {
                                dbg!("hoge1");
                                img_out[(ih * img_shape.0 + iw0) * 3] = 0.5 + rc;
                            } else {
                                img_out[(ih * img_shape.0 + iw0) * 3 + 1] = 1f32;
                                img_out[(ih * img_shape.0 + iw1) * 3 + 1] = rc - 0.5;
                            }
                        } else if let Some((rc, re)) = e2 {
                            assert!(rc >= 0. && rc <= 1.0);
                            let rc = 1.0 - rc;
                            dbg!((rc, iw0, ih));
                            if rc < 0.5 {
                                dbg!("hoge2");
                                img_out[(ih * img_shape.0 + iw0) * 3] = 0.5 + rc;
                            } else {
                                img_out[(ih * img_shape.0 + iw0) * 3 + 1] = 1f32;
                                img_out[(ih * img_shape.0 + iw1) * 3 + 1] = rc - 0.5;
                            }
                        }
                    } else if which_in == 1 {
                        // assert!(flg1.is_some() && flg0.is_none());
                    }
                }
            }
        }
    }
     */

    Ok(())
}
