

fn main() -> anyhow::Result<()> {
    let (tri2vtx, vtx2xyz) = del_msh_core::trimesh3_primitive::sphere_yup::<u32, f32>(0.8, 64, 64);

    /*
    let (tri2vtx, vtx2xyz) = {
        let mut obj = del_msh_core::io_obj::WavefrontObj::<u32, f32>::new();
        obj.load("asset/spot/spot_triangulated.obj")?;
        (obj.idx2vtx_xyz, obj.vtx2xyz)
    };
     */

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
    let img_shape_lowres = (((16 * 6) as f32 * img_asp) as usize, 16 * 6);
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
        use del_geo_core::vec3::Vec3;
        let (i0_vtx, i1_vtx) = (node2vtx[0], node2vtx[1]);
        let q0 = del_msh_core::vtx2xyz::to_xyz(&vtx2xyz, i0_vtx as usize)
            .p
            .transform_homogeneous(&transform_world2pix_lowres)
            .unwrap()
            .xy();
        let q1 = del_msh_core::vtx2xyz::to_xyz(&vtx2xyz, i1_vtx as usize)
            .p
            .transform_homogeneous(&transform_world2pix_lowres)
            .unwrap()
            .xy();
        let v01 = del_geo_core::vec2::sub(&q1,&q0);
        let is_horizontal = v01[0].abs() < v01[1].abs();
        let list_pix = del_geo_core::edge2::overlapping_pixels_dda(img_shape_lowres, &q0, &q1);
        for &i_pix in list_pix.iter() {
            let (iw1, ih1) = (i_pix % img_shape_lowres.0, i_pix / img_shape_lowres.0);
            let list_i_pix = [
                ih1 * img_shape_lowres.0 + iw1,       // c
                ih1 * img_shape_lowres.0 + iw1 - 1,   // w
                ih1 * img_shape_lowres.0 + iw1 + 1,   // e
                (ih1 - 1) * img_shape_lowres.0 + iw1, // s
                (ih1 + 1) * img_shape_lowres.0 + iw1, // n
            ];
            let list_pos_c = [
                [iw1 as f32 + 0.5, ih1 as f32 + 0.5], // c
                [iw1 as f32 - 0.5, ih1 as f32 + 0.5], // w
                [iw1 as f32 + 1.5, ih1 as f32 + 0.5], // e
                [iw1 as f32 + 0.5, ih1 as f32 - 0.5], // s
                [iw1 as f32 + 0.5, ih1 as f32 + 1.5], // n
            ];
            let list_index = if is_horizontal {
                [(0, 1), (1, 0), (0, 2), (2, 0)]
            } else {
                [(0, 3), (3, 0), (0, 4), (4, 0)]
            };
            for (idx0, idx1) in list_index {
                let i_pix0 = list_i_pix[idx0];
                let i_pix1 = list_i_pix[idx1];
                let is_in0 = pix2tri[i_pix0] != u32::MAX;
                let is_in1 = pix2tri[i_pix1] != u32::MAX;
                let c0 = list_pos_c[idx0];
                let c1 = list_pos_c[idx1];
                if !is_in0 || is_in1 {
                    continue;
                } // zero in && one out
                let Some((rc, re)) = del_geo_core::edge2::intersection_edge2(&c0, &c1, &q1, &q0)
                else {
                    continue;
                };
                assert!(rc >= 0. && rc <= 1.0);
                if rc < 0.5 {
                    img_lowres[i_pix0 * 3] = 0.5 + rc;
                } else {
                    img_lowres[i_pix1 * 3] = rc - 0.5;
                }
            }
        }
    }
    del_canvas::write_png_from_float_image_rgb(
        "target/07_anti_aliasing_lowres.png",
        &img_shape_lowres,
        &img_lowres,
    )?;

    let expansion_ratio = 9;
    let (img_shape_hires, mut img_hires) =
        del_canvas::expand_image(img_shape_lowres, &img_lowres, expansion_ratio);
    let transform_world2pix_hires = {
        let transform_ndc2pix =
            del_geo_core::mat3_col_major::from_transform_ndc2pix(img_shape_hires);
        let transform_ndc2pix =
            del_geo_core::mat4_col_major::from_mat3_col_major_adding_z(&transform_ndc2pix);
        del_geo_core::mat4_col_major::mult_mat(&transform_ndc2pix, &transform_world2ndc)
    };
    for node2vtx in edge2vtx_contour.chunks(2) {
        let (i0_vtx, i1_vtx) = (node2vtx[0], node2vtx[1]);
        let p0 = del_msh_core::vtx2xyz::to_xyz(&vtx2xyz, i0_vtx as usize).p;
        let p1 = del_msh_core::vtx2xyz::to_xyz(&vtx2xyz, i1_vtx as usize).p;
        use del_geo_core::vec3::Vec3;
        let q0 = p0
            .transform_homogeneous(&transform_world2pix_hires)
            .unwrap()
            .xy();
        let q1 = p1
            .transform_homogeneous(&transform_world2pix_hires)
            .unwrap()
            .xy();
        use slice_of_array::SliceNestExt;
        del_canvas::rasterize::line::draw_dda(
            img_hires.nest_mut(),
            img_shape_hires.0,
            &q0,
            &q1,
            [0., 1., 1.],
        );
    }
    del_canvas::write_png_from_float_image_rgb(
        "target/07_anti_aliasing_hires.png",
        &img_shape_hires,
        &img_hires,
    )?;
    Ok(())
}
