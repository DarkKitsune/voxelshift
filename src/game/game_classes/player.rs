use game_components::physics::Physics;

use crate::game::*;

use super::world::{VoxelPosition, VoxelUnits};

const PLAYER_ACCELERATION: f32 = 30.0;
const PLAYER_FRICTION: f32 = 10.0;
const PLAYER_PITCH_LIMIT: f32 = std::f32::consts::PI * 0.3;

define_class! {
    pub class Player {
        location: Location,
        camera: Camera,
        physics: Physics,
        key_state: PlayerKeyState,
        look: Look,
    }
}

impl Player {
    /// Creates a new player.
    pub fn new(position: Vector3<f32>, fov: f32) -> Self {
        Self {
            location: Location::new(position, Quaternion::identity(), Vector3::one()),
            camera: Camera::new_perspective(None, fov, 0.05, 100.0),
            physics: Physics {
                velocity: Vector3::zero(),
            },
            key_state: PlayerKeyState {
                forward: false,
                backward: false,
                left: false,
                right: false,
            },
            look: Look::new(),
        }
    }

    /// Simulate a step of player movement.
    /// Returns whether the player has moved
    pub fn simulate_movement(&mut self, player_location: &Location, frame_delta: Duration) -> bool {
        let delta_f32 = frame_delta.as_secs_f32();
        // Get the player-local motion vector from the key state
        let local_motion_vector = self.key_state.to_local_motion_vector().unwrap_or_default();
        // Make the motion vector local to the player's parent node
        let motion_vector = player_location.delocalize_direction(local_motion_vector)
            * player_location.delocalize_scale(vector!(
                PLAYER_ACCELERATION,
                PLAYER_ACCELERATION,
                PLAYER_ACCELERATION
            ));
        // Add the motion vector to the player's velocity
        self.physics.velocity = self.physics.velocity + motion_vector * delta_f32;
        // Update the player's location
        let player_moved = if self.physics.velocity.length_squared() > 0.0 {
            self.location.translate(self.physics.velocity * delta_f32);
            true
        } else {
            false
        };
        // Apply friction to the player's velocity
        self.physics.velocity =
            self.physics.velocity * (1.0 - (PLAYER_FRICTION * delta_f32).min(1.0));
        // Return whether the player moved
        player_moved
    }

    /// Simulate a key action.
    pub fn key_action(&mut self, key: Key, action: Action, _modifiers: Modifiers) {
        match key {
            Key::W => self.key_state.forward = action != Action::Release,
            Key::S => self.key_state.backward = action != Action::Release,
            Key::A => self.key_state.left = action != Action::Release,
            Key::D => self.key_state.right = action != Action::Release,
            _ => (),
        }
    }

    /// Simulate mouse movement.
    pub fn move_mouse(&mut self, mouse_delta: Vector2<f32>, sensitivity: f32) {
        self.look.add_yaw(mouse_delta.x() * sensitivity);
        self.look.add_pitch(mouse_delta.y() * sensitivity);

        self.location.set_rotation(self.look.to_location_rotation());
        self.camera.set_rotation(self.look.to_local_rotation());
    }

    /// Get the player location.
    pub fn location(&self) -> &Location {
        &self.location
    }

    pub fn world_position(&self) -> Vector3<f64> {
        let position = self.location.position();
        vector!(
            position.x() as f64,
            position.y() as f64,
            position.z() as f64
        )
    }

    pub fn set_world_position(&mut self, position: Vector3<f64>) {
        self.location.set_position(vector!(
            position.x() as f32,
            position.y() as f32,
            position.z() as f32
        ));
    }

    pub fn voxel_position(&self) -> VoxelPosition {
        let position = self.location.position();
        vector!(
            VoxelUnits::from(position.x().floor() as i64),
            VoxelUnits::from(position.y().floor() as i64),
            VoxelUnits::from(position.z().floor() as i64)
        )
    }
}

/// Tracks the state of the player's keyboard inputs.
pub struct PlayerKeyState {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
}

impl PlayerKeyState {
    /// Convert the key state into a normalized motion vector local to player space.
    /// Returns `None` if no keys are pressed.
    fn to_local_motion_vector(&self) -> Option<Vector3<f32>> {
        if !self.forward && !self.backward && !self.left && !self.right {
            None
        } else {
            Some(
                vector!(
                    (if self.right { 1.0 } else { 0.0 }) - (if self.left { 1.0 } else { 0.0 }),
                    0.0,
                    (if self.backward { 1.0 } else { 0.0 })
                        - (if self.forward { 1.0 } else { 0.0 }),
                )
                .normalized(),
            )
        }
    }
}

/// Component for handling the player's ability to look around.
pub struct Look {
    /// The player's current pitch angle.
    pub pitch: f32,
    /// The player's current yaw angle.
    pub yaw: f32,
}

impl Look {
    /// Create a new look component.
    pub fn new() -> Self {
        Self {
            pitch: 0.0,
            yaw: 0.0,
        }
    }

    /// Add to the yaw angle.
    pub fn add_yaw(&mut self, amount: f32) {
        let new_yaw = (self.yaw + amount) % std::f32::consts::TAU;
        if new_yaw < 0.0 {
            self.yaw = new_yaw + std::f32::consts::TAU;
        } else {
            self.yaw = new_yaw;
        }
    }

    /// Add to the pitch angle.
    pub fn add_pitch(&mut self, amount: f32) {
        self.pitch = (self.pitch + amount).clamp(-PLAYER_PITCH_LIMIT, PLAYER_PITCH_LIMIT);
    }

    /// Convert to a rotation local to the player.
    pub fn to_local_rotation(&self) -> Quaternion<f32> {
        Quaternion::from_rotation_x(self.pitch)
    }

    /// Get the rotation the player's location should have.
    pub fn to_location_rotation(&self) -> Quaternion<f32> {
        Quaternion::from_rotation_y(self.yaw)
    }
}
