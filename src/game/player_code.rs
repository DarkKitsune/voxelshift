use crate::game::*;

use super::player::Player;

pub fn update_player(scene: &mut Scene, player_handle: &Handle, frame_delta: Duration) {
    if let Some(player_node) = scene.universe_mut().node_mut(player_handle) {
        let player_class = player_node.class_as_mut::<Player>().expect("Player node has no Player component");
        let player_location = player_class.location().clone();
        // Simulate player movement
        player_class.simulate_movement(&player_location, frame_delta);
    }
}

pub fn player_key_action(scene: &mut Scene, player_handle: &Handle, key: Key, action: Action, modifiers: Modifiers) {
    if let Some(player_node) = scene.universe_mut().node_mut(player_handle) {
        let player_class = player_node.class_as_mut::<Player>().expect("Player node has no Player component");
        // Simulate a key press in the player component
        player_class.key_action(key, action, modifiers);
    }
}

pub fn player_mouse_motion(scene: &mut Scene, player_handle: &Handle, mouse_delta: Vector2<f32>) {
    if let Some(player_node) = scene.universe_mut().node_mut(player_handle) {
        let player_class = player_node.class_as_mut::<Player>().expect("Player node has no Player component");
        // Simulate a mouse action in the player component
        player_class.move_mouse(mouse_delta, 0.03);
    }
}