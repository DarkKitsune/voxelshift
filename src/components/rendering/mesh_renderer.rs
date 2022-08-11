use crate::{
    gfx::{Mesh, Program},
    vertex::DebugVertex,
};

/// A component that marks a node to be rendered as a mesh.
pub struct MeshRenderer {
    pub mesh: Mesh<DebugVertex>,
    pub program: Program,
}

impl MeshRenderer {
    pub fn new(mesh: Mesh<DebugVertex>, program: Program) -> Self {
        Self { mesh, program }
    }
}
