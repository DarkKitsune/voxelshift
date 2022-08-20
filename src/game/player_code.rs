use crate::game::*;

use super::{
    player::Player,
    world::{DynWorldGenerator, VoxelUnits, World, VOXELS_PER_METER},
};

const PLAYER_EYE_HEIGHT: f64 = 1.7;

pub fn update_player(
    scene: &mut Scene,
    player_handle: &Handle,
    world_handle: &Handle,
    frame_delta: Duration,
) {
    let (player_physics_step, player_position, player_voxel_position) = {
        let player_node = scene
            .universe_mut()
            .node_mut(player_handle)
            .expect("Player node not found");
        let player_class = player_node
            .class_as_mut::<Player>()
            .expect("Player node has no Player component");
        // Simulate player movement
        let player_physics_step = player_class.simulate_movement(frame_delta);
        // Return player positions
        (
            player_physics_step,
            player_class.position(),
            player_class.voxel_position(),
        )
    };
    // Set player on ground
    if player_physics_step.position_changed {
        let ground_position = scene
            .universe()
            .node(world_handle)
            .expect("World node not found")
            .class_as::<World>()
            .expect("World node is not a World")
            .set_on_ground(player_position);
        let player_node = scene
            .universe_mut()
            .node_mut(player_handle)
            .expect("Player node not found");
        let player_class = player_node
            .class_as_mut::<Player>()
            .expect("Player node has no Player component");
        player_class
            .set_position(ground_position + vector!(0.0, PLAYER_EYE_HEIGHT, 0.0));
    }
    // Extend and remove chunks
    let world = scene
        .universe_mut()
        .node_mut(world_handle)
        .expect("World node not found");
    let world_class = world
        .class_as_mut::<World>()
        .expect("World node is not a World");
    world_class.remove_old_chunks();
    world_class.extend_life_in_view(player_voxel_position, VoxelUnits::from(20i64))
}

pub fn player_key_action(
    scene: &mut Scene,
    player_handle: &Handle,
    key: Key,
    action: Action,
    modifiers: Modifiers,
) {
    if let Some(player_node) = scene.universe_mut().node_mut(player_handle) {
        let player_class = player_node
            .class_as_mut::<Player>()
            .expect("Player node has no Player component");
        // Simulate a key press in the player component
        player_class.key_action(key, action, modifiers);
    }
}

pub fn player_mouse_motion(scene: &mut Scene, player_handle: &Handle, mouse_delta: Vector2<f64>) {
    if let Some(player_node) = scene.universe_mut().node_mut(player_handle) {
        let player_class = player_node
            .class_as_mut::<Player>()
            .expect("Player node has no Player component");
        // Simulate a mouse action in the player component
        player_class.move_mouse(mouse_delta, 0.03);
    }
}
