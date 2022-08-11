use crate::game::*;

define_class! {
    /// The game world.
    pub class World {
        mesh_renderer: MeshRenderer,
        location: Location,
    }
}

impl World {
    pub fn new(gfx: &Gfx, world_program: Program) -> Self {
        // Create mesh to be put in mesh_renderer
        let mut proto_mesh = ProtoMesh::new();
        proto_mesh.add_box(
            Orientation::identity(),
            vector!(1.0, 1.0, 1.0),
            &BoxSides::new_uniform(&BoxSide {
                color: Color::RED,
                tex_coords: (vector!(0.0, 0.0), vector!(1.0, 1.0)),
            }),
        );
        let mesh = gfx.create_mesh::<DebugVertex>(&proto_mesh);

        Self {
            mesh_renderer: MeshRenderer::new(mesh, world_program),
            location: Location::new(
                vector!(0.0, 0.0, 0.0),
                Quaternion::new_identity(),
                vector!(1.0, 1.0, 1.0),
            ),
        }
    }
}
