use crate::game::*;

use super::{
    player::Player, programs::world_program::WorldProgram, world::World,
    world_generator::BasicWorldGenerator,
};

// ============================================================================
//     Scene events
// ============================================================================

/// Build the scene
pub fn build_scene() -> Scene {
    SceneBuilder::new()
        .with_clear_color(Color::CYAN)
        .on_init(on_init)
        .on_update(on_update)
        .on_render(on_render)
        .on_key(on_key)
        .on_mouse_move(on_mouse_move)
        .build()
}

/// Scene init event
pub fn on_init(scene: &mut Scene, gfx: &Gfx) {
    // Initialize the scene's resources
    init_resources(scene, gfx);

    // Initialize the game universe, returning the handles
    // to important nodes in the universe
    let node_handles = init_universe(scene, gfx);

    // Store important node handles in the scene as a resource
    scene.insert("node handles", node_handles);
}

/// Scene update event
pub fn on_update(scene: &mut Scene, frame_delta: Duration) {
    let node_handles = scene.get::<NodeHandles>("node handles").clone();
    update_scene(scene, &node_handles, frame_delta);
}

/// Scene render event
pub fn on_render(
    scene: &mut Scene,
    gfx: &Gfx,
    framebuffer: &Framebuffer,
    render_camera: &RenderCamera,
    window_size: Vector2<u32>,
    frame_delta: Duration,
) {
    // Render world chunk meshes
    let drawn_meshes: Vec<_> = scene
        .universe_mut()
        .nodes_mut()
        .with_class::<World>()
        .map(|node| {
            let world = node.class_as_mut::<World>().unwrap();
            (world.program().clone(), world)
        })
        .flat_map(|(program, world)| {
            world.chunk_meshes(gfx).filter_map(move |(location, mesh)| {
                mesh.map(|mesh| (program.clone(), (location, mesh)))
            })
        })
        .collect();
    for (program, (location, mesh)) in drawn_meshes {
        gfx.render_mesh(
            framebuffer,
            &program,
            mesh,
            1,
            Some(&render_camera),
            render_uniforms! [
                mesh_position: location.position().convert_to::<f32>().unwrap(),
            ],
        );
    }
}

/// Scene key event
pub fn on_key(scene: &mut Scene, key: Key, action: Action, modifiers: Modifiers) {
    let node_handles = scene.get::<NodeHandles>("node handles").clone();
    scene_key_action(scene, &node_handles.player, key, action, modifiers)
}

/// Scene mouse move event
pub fn on_mouse_move(scene: &mut Scene, mouse_delta: Vector2<f64>) {
    let node_handles = scene.get::<NodeHandles>("node handles").clone();
    scene_mouse_move(scene, &node_handles.player, mouse_delta)
}

// ============================================================================
//     Scene initialization
// ============================================================================

/// Initialize scene resources
fn init_resources(scene: &mut Scene, gfx: &Gfx) {
    // Create world program
    scene.create_program(gfx, "world program", &WorldProgram);
}

/// Initialize the game universe
fn init_universe(scene: &mut Scene, gfx: &Gfx) -> NodeHandles {
    // Get the world program from the scene
    let world_program = scene.get_program("world program");

    // Create the world node
    let world = scene.universe_mut().create_node(
        None,
        World::new(gfx, world_program, BasicWorldGenerator::new(12345)),
    );

    // Create the player node
    let player = scene
        .universe_mut()
        .create_node(None, Player::new(vector!(0.0, 0.0, 2.0), std::f64::consts::PI * 0.5));

    NodeHandles { world, player }
}

// ============================================================================
//     Scene update
// ============================================================================

/// Update the scene
fn update_scene(scene: &mut Scene, node_handles: &NodeHandles, frame_delta: Duration) {
    // Update the player
    player_code::update_player(
        scene,
        &node_handles.player,
        &node_handles.world,
        frame_delta,
    );
}

// ============================================================================
//     Scene input events
// ============================================================================

/// Scene key event
fn scene_key_action(
    scene: &mut Scene,
    player_handle: &Handle,
    key: Key,
    action: Action,
    modifiers: Modifiers,
) {
    player_code::player_key_action(scene, player_handle, key, action, modifiers);
}

/// Scene mouse move event
fn scene_mouse_move(scene: &mut Scene, player_handle: &Handle, mouse_delta: Vector2<f64>) {
    player_code::player_mouse_motion(scene, player_handle, mouse_delta);
}

/// Keeps track of handles pointing to important nodes in the game universe
#[derive(Clone)]
struct NodeHandles {
    world: Handle,
    player: Handle,
}
