use crate::game::*;

use super::{
    player::Player,
    world::{World, WORLD_VIEW_DISTANCE_METERS},
};

pub fn update_player(
    scene: &mut Scene,
    player_handle: &Handle,
    world_handle: &Handle,
    frame_delta: Duration,
) {
    let (player_physics_step, player_position, player_height, player_look_direction) = {
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
            player_class.height(),
            player_class.look_direction(),
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
        player_class.set_position(ground_position + vector!(0.0, player_height * 0.5, 0.0));
    }

    // Extend and remove chunks
    let average_chunk_generation_time = {
        let world = scene
            .universe_mut()
            .node_mut(world_handle)
            .expect("World node not found");
        let world_class = world
            .class_as_mut::<World>()
            .expect("World node is not a World");
        world_class.remove_old_chunks();
        world_class.extend_life_in_view(
            player_position,
            WORLD_VIEW_DISTANCE_METERS,
            Some(player_look_direction),
        )
    };

    // Chunk generation time measurements
    if let Some(average_chunk_generation_time) = average_chunk_generation_time {
        // Get the current average chunk generation time from the scene
        let (mut total_average_chunk_generation_time, mut total_average_chunk_generation_samples) =
            if let Some((
                total_average_chunk_generation_time,
                total_average_chunk_generation_samples,
            )) = scene.try_get::<(f64, usize)>("total average chunk generation time")
            {
                (
                    *total_average_chunk_generation_time,
                    *total_average_chunk_generation_samples,
                )
            } else {
                (0.0, 0)
            };
        
        // Update it
        total_average_chunk_generation_time += average_chunk_generation_time;
        total_average_chunk_generation_samples += 1;

        // Put the updated time back into the scene
        scene.insert(
            "total average chunk generation time",
            (
                total_average_chunk_generation_time,
                total_average_chunk_generation_samples,
            ),
        );

        // Print the timing information
        println!(
            "Average chunk generation time:\t{} ms\t|\tTotal:{} ms",
            average_chunk_generation_time * 1000.0,
            total_average_chunk_generation_time * 1000.0
                / total_average_chunk_generation_samples as f64
        );
    }
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
