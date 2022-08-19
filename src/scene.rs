use std::{any::Any, collections::HashMap, time::Duration};

use ggmath::{
    prelude::{Matrix3x3, Matrix4x4},
    vector_alias::{Vector2, Vector3}, quaternion::Quaternion,
};
use ggutil::prelude::Handle;
use glfw::{Action, Key, Modifiers};
use multiverse_ecs::universe::{NodesIter, Universe};

use crate::{
    colors::Color,
    components::{common::location::Location, *},
    game::{Program, ProgramTemplate},
    gfx::{Framebuffer, Gfx, RenderCamera},
    render_uniforms,
};

/// A builder for creating a `Scene`.
pub struct SceneBuilder {
    clear_color: Option<Color>,
    on_init: Option<fn(&mut Scene, &Gfx)>,
    on_update: Option<fn(&mut Scene, Duration)>,
    on_render: Option<fn(&mut Scene, &Gfx, &Framebuffer, &RenderCamera, Vector2<u32>, Duration)>,
    on_key: Option<fn(&mut Scene, Key, Action, Modifiers)>,
    on_mouse_move: Option<fn(&mut Scene, Vector2<f32>)>,
}

impl SceneBuilder {
    /// Creates a new scene builder.
    pub fn new() -> Self {
        Self {
            clear_color: Some(Color::BLACK),
            on_init: None,
            on_update: None,
            on_render: None,
            on_key: None,
            on_mouse_move: None,
        }
    }

    /// Sets the clear color of the scene.
    pub fn with_clear_color(mut self, color: Color) -> Self {
        self.clear_color = Some(color);
        self
    }

    /// Sets the init event callback.
    pub fn on_init(mut self, on_init: fn(&mut Scene, &Gfx)) -> Self {
        self.on_init = Some(on_init);
        self
    }

    /// Sets the update event callback.
    pub fn on_update(mut self, on_update: fn(&mut Scene, Duration)) -> Self {
        self.on_update = Some(on_update);
        self
    }

    /// Sets the render event callback.
    pub fn on_render(
        mut self,
        on_render: fn(&mut Scene, &Gfx, &Framebuffer, &RenderCamera, Vector2<u32>, Duration),
    ) -> Self {
        self.on_render = Some(on_render);
        self
    }

    /// Sets the key event callback.
    pub fn on_key(mut self, on_key: fn(&mut Scene, Key, Action, Modifiers)) -> Self {
        self.on_key = Some(on_key);
        self
    }

    /// Sets the mouse move event callback.
    pub fn on_mouse_move(mut self, on_mouse_move: fn(&mut Scene, Vector2<f32>)) -> Self {
        self.on_mouse_move = Some(on_mouse_move);
        self
    }

    /// Builds the scene.
    pub fn build(self) -> Scene {
        Scene::new(
            self.clear_color,
            self.on_init,
            self.on_update,
            self.on_render,
            self.on_key,
            self.on_mouse_move,
        )
    }
}

/// A `Scene` holds and manipulates the app's resources and state.
pub struct Scene {
    universe: Universe,
    resources: HashMap<String, Box<dyn Any>>,
    clear_color: Option<Color>,
    on_init: Option<fn(&mut Scene, &Gfx)>,
    on_update: Option<fn(&mut Scene, Duration)>,
    on_render: Option<fn(&mut Scene, &Gfx, &Framebuffer, &RenderCamera, Vector2<u32>, Duration)>,
    on_key: Option<fn(&mut Scene, Key, Action, Modifiers)>,
    on_mouse_move: Option<fn(&mut Scene, Vector2<f32>)>,
    last_mouse_position: Option<Vector2<f32>>,
    anchor_location: Option<Location>,
}

impl Scene {
    /// Creates a new scene.
    fn new(
        clear_color: Option<Color>,
        on_init: Option<fn(&mut Scene, &Gfx)>,
        on_update: Option<fn(&mut Scene, Duration)>,
        on_render: Option<fn(&mut Scene, &Gfx, &Framebuffer, &RenderCamera, Vector2<u32>, Duration)>,
        on_key: Option<fn(&mut Scene, Key, Action, Modifiers)>,
        on_mouse_move: Option<fn(&mut Scene, Vector2<f32>)>,
    ) -> Self {
        Self {
            universe: Universe::new(),
            resources: HashMap::new(),
            clear_color,
            on_init,
            on_update,
            on_render,
            on_key,
            on_mouse_move,
            last_mouse_position: None,
            anchor_location: None,
        }
    }

    /// Calls the init event on the scene.
    pub fn __init(&mut self, gfx: &Gfx) {
        self.on_init
            .as_ref()
            .expect("No on_init function set in the scene")(self, gfx);
    }

    /// Calls the update event on the scene.
    pub fn __update(&mut self, frame_delta: Duration) {
        self.on_update
            .as_ref()
            .expect("No on_update function set in the scene")(self, frame_delta);
    }

    /// Calls the render event on the scene.
    pub fn __render(
        &mut self,
        gfx: &Gfx,
        framebuffer: &Framebuffer,
        window_size: Vector2<u32>,
        frame_delta: Duration,
    ) {
        // Clear color and depth
        if let Some(clear_color) = self.clear_color {
            gfx.clear_color(framebuffer, clear_color);
        }

        // Render from nodes in universe
        if let Some((camera_node, camera)) = self.universe.nodes().with_component::<Camera>().next()
        {
            let absolute_camera_location = self.get_absolute_location(camera_node.handle().clone());
            let absolute_camera_rotation_matrix = Matrix3x3::from(absolute_camera_location.delocalize_rotation(camera.rotation()));

            // Set the camera
            let absolute_camera_position = absolute_camera_location.position();
            let absolute_camera_target = if let Some(camera_target_handle) = camera.target() {
                Some(self.get_absolute_location(camera_target_handle.clone()))
            } else {
                None
            };
            let absolute_camera_direction = absolute_camera_target
                .map(|target| (target.position() - absolute_camera_position).normalized())
                .unwrap_or_else(|| -absolute_camera_rotation_matrix.z_axis());
            let aspect_ratio = window_size.x() as f32 / window_size.y() as f32;
            let render_camera = RenderCamera::new(
                absolute_camera_position,
                absolute_camera_direction,
                absolute_camera_rotation_matrix.y_axis(),
                camera.projection_matrix(aspect_ratio),
            );

            self.on_render
                .as_ref()
                .expect("No on_render function set in the scene")(
                self,
                gfx,
                framebuffer,
                &render_camera,
                window_size,
                frame_delta,
            );

            // Render mesh objects
            for (node, mesh_renderer) in self.universe.nodes().with_component::<MeshRenderer>() {
                let absolute_mesh_location = self.get_absolute_location(node.handle().clone());
                let mesh = &mesh_renderer.mesh;
                let program = &mesh_renderer.program;
                gfx.render_mesh(
                    framebuffer,
                    program,
                    mesh,
                    1,
                    Some(&render_camera),
                    render_uniforms! [
                        mesh_position: absolute_mesh_location.position(),
                    ],
                );
            }
        }
    }

    /// Calls the key event on the scene.
    pub fn __key_action(&mut self, key: Key, action: Action, modifiers: Modifiers) {
        self.on_key
            .as_ref()
            .expect("No on_key function set in the scene")(self, key, action, modifiers);
    }

    /// Calls the mouse move event on the scene.
    pub fn __mouse_move(&mut self, position: Vector2<f32>) {
        if let Some(last_mouse_position) = self.last_mouse_position {
            let delta = position - last_mouse_position;
            self.on_mouse_move
                .as_ref()
                .expect("No on_mouse_move function set in the scene")(self, delta);
        }
        self.last_mouse_position = Some(position);
    }

    /// Set the scene's clear color
    pub fn set_clear_color(&mut self, color: Color) {
        self.clear_color = Some(color);
    }

    /// Get the scene's clear color
    pub fn clear_color(&self) -> Option<Color> {
        self.clear_color
    }

    /// Get a reference to the scene's universe
    pub fn universe(&self) -> &Universe {
        &self.universe
    }

    /// Get a mutable reference to the scene's universe
    pub fn universe_mut(&mut self) -> &mut Universe {
        &mut self.universe
    }

    /// Insert a resource into the scene
    pub fn insert(&mut self, name: impl Into<String>, resource: impl Any) {
        self.resources.insert(name.into(), Box::new(resource));
    }

    /// Get a reference to a resource from the scene
    /// Panics if the resource does not exist
    pub fn get<T: Any>(&self, name: &str) -> &T {
        self.resources
            .get(name)
            .and_then(|resource| resource.downcast_ref())
            .unwrap_or_else(|| panic!("Resource not found: {}", name))
    }

    /// Get a reference to a resource from the scene
    /// Returns `None` if the resource does not exist
    pub fn try_get<T: Any>(&self, name: &str) -> Option<&T> {
        self.resources
            .get(name)
            .and_then(|resource| resource.downcast_ref())
    }

    /// Get a program from the scene (this is a clonable smart pointer),
    /// as previously created with `Scene::create_program()`
    /// Panics if the program does not exist
    pub fn get_program(&self, name: &str) -> Program {
        self.get::<Program>(name).clone()
    }

    /// Get a program from the scene (this is a clonable smart pointer),
    /// as previously created with `Scene::create_program()`
    /// Returns `None` if the program does not exist
    pub fn try_get_program(&self, name: &str) -> Option<Program> {
        self.try_get::<Program>(name).cloned()
    }

    /// Insert a program as a resources into the scene
    pub fn create_program(
        &mut self,
        gfx: &Gfx,
        name: impl Into<String>,
        template: &impl ProgramTemplate,
    ) {
        self.insert(name, template.create_program(gfx));
    }

    /// Gets the location of a node, local to the given location
    pub fn get_location_local_to_location(&self, node_handle: Handle, local_to: Location) -> Location {
        let node = self
            .universe
            .node(&node_handle)
            .expect("Invalid handle passed as \"node_handle\"");

        let node_location = node
            .component::<Location>()
            .expect("Node pointed to by \"node_handle\" does not have a Location component");
        
        // Calculate the absolute location of the node
        let absolute_location = if let Some(parent) = node.parent() {
            // If the parent has a location, delocalize the location from it
            self.get_location_local_to_location(parent.clone(), local_to.clone()).delocalize_location(node_location.clone())
        } else {
            node_location.clone()
        };

        // Make the absolute location local to local_to
        local_to.localize_location(absolute_location)
    }

    /// Gets the location of a node, local to the scene anchor
    pub fn get_absolute_location(&self, node_handle: Handle) -> Location {
        self.get_location_local_to_location(node_handle, self.anchor_location.clone().unwrap_or_default())
    }

    pub fn set_anchor_location(&mut self, location: Option<Location>) {
        self.anchor_location = location;
    }

    pub fn anchor_location(&self) -> Option<Location> {
        self.anchor_location.clone()
    }
}
