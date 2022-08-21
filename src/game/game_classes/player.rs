use game_components::physics::{Physics, PhysicsStepInfo};

use crate::game::*;

use super::world::{VoxelPosition, VoxelUnits};

const PLAYER_ACCELERATION: f64 = 55.0;
const PLAYER_FRICTION: f64 = 10.0;
const PLAYER_PITCH_LIMIT: f64 = std::f64::consts::PI * 0.3;
const PLAYER_HEIGHT: f64 = 1.75;
const PLAYER_EYE_HEIGHT: f64 = PLAYER_HEIGHT * 0.5 - 0.06;

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
    pub fn new(position: Vector3<f64>, fov: f64) -> Self {
        Self {
            location: Location::new(position, Quaternion::identity(), Vector3::one()),
            camera: Camera::new_perspective(None, vector!(0.0, PLAYER_EYE_HEIGHT, 0.0), Quaternion::identity(), fov, 0.05, 100.0),
            physics: Physics {
                mass: 88.7,
                velocity: Vector3::zero(),
                friction: PLAYER_FRICTION,
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
    /// Returns physics information on what has changed during this step.
    pub fn simulate_movement(&mut self, frame_delta: Duration) -> PhysicsStepInfo {
        let delta_f64 = frame_delta.as_secs_f64();
        // Get the player-local motion vector from the key state
        let local_motion_vector = self.key_state.to_local_motion_vector().unwrap_or_default();
        // Convert the player-local motion vector to a world-local acceleration vector
        let acceleration_vector = self.location.delocalize_direction(local_motion_vector)
            * self.location.delocalize_scale(vector!(
                PLAYER_ACCELERATION,
                PLAYER_ACCELERATION,
                PLAYER_ACCELERATION
            ));
        // Apply the acceleration to the player physics.
        self.physics.accelerate(acceleration_vector, delta_f64);
        // Update the physics
        let physics_step = self.physics.simulate(&mut self.location, delta_f64);
        // Return physics information
        physics_step
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
    pub fn move_mouse(&mut self, mouse_delta: Vector2<f64>, sensitivity: f64) {
        self.look.add_yaw(mouse_delta.x() * sensitivity);
        self.look.add_pitch(mouse_delta.y() * sensitivity);

        self.location.set_rotation(self.look.to_location_rotation());
        self.camera.set_rotation(self.look.to_local_rotation());
    }

    /// Get the position of the player in meters.
    pub fn position(&self) -> Vector3<f64> {
        self.location.position()
    }

    /// Set the position of the player in meters.
    pub fn set_position(&mut self, position: Vector3<f64>) {
        self.location.set_position(position);
    }

    pub fn voxel_position(&self) -> VoxelPosition {
        self.location.position().map(|v| VoxelUnits::from_meters(*v))
    }

    pub fn local_eye_position(&self) -> Vector3<f64> {
        self.camera.translation()
    }

    pub fn height(&self) -> f64 {
        PLAYER_HEIGHT
    }

    pub fn look_direction(&self) -> Vector3<f64> {
        -Vector3::unit_z().rotated_by(&self.location.rotation())
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
    fn to_local_motion_vector(&self) -> Option<Vector3<f64>> {
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
    pitch: f64,
    /// The player's current yaw angle.
    yaw: f64,
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
    pub fn add_yaw(&mut self, amount: f64) {
        let new_yaw = (self.yaw + amount) % std::f64::consts::TAU;
        if new_yaw < 0.0 {
            self.yaw = new_yaw + std::f64::consts::TAU;
        } else {
            self.yaw = new_yaw;
        }
    }

    /// Add to the pitch angle.
    pub fn add_pitch(&mut self, amount: f64) {
        self.pitch = (self.pitch + amount).clamp(-PLAYER_PITCH_LIMIT, PLAYER_PITCH_LIMIT);
    }

    /// Convert to a rotation local to the player.
    pub fn to_local_rotation(&self) -> Quaternion<f64> {
        Quaternion::from_rotation_x(self.pitch)
    }

    /// Get the rotation the player's location should have.
    pub fn to_location_rotation(&self) -> Quaternion<f64> {
        Quaternion::from_rotation_y(self.yaw)
    }
}
