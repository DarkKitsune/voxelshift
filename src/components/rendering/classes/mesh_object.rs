use multiverse_ecs::define_class;

use crate::components::{common::location::Location, rendering::mesh_renderer::MeshRenderer};

define_class! {
    pub class MeshObject {
        location: Location,
        mesh_renderer: MeshRenderer,
    }
}

impl MeshObject {
    pub fn new(location: Location, mesh_renderer: MeshRenderer) -> Self {
        Self {
            location,
            mesh_renderer,
        }
    }
}
