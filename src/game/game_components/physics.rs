use crate::game::*;

pub struct Physics {
    /// Mass in kilograms
    pub mass: f64,
    /// Velocity in meters per second
    pub velocity: Vector3<f64>,
    /// Friction
    pub friction: f64,
}

impl Default for Physics {
    fn default() -> Self {
        Self {
            mass: 1.0,
            velocity: Vector::zero(),
            friction: 0.0,
        }
    }
}

impl Physics {
    /// Simulate physics on the given `Location`.
    /// Returns information on what has changed during this physics step.
    pub fn simulate(&mut self, location: &mut Location, delta_seconds: f64) -> PhysicsStepInfo {
        // Exit early if delta_seconds is 0 or less
        if delta_seconds < std::f64::EPSILON {
            return Default::default();
        }

        // Translate the location by the velocity
        let position_changed = if self.velocity.length_squared() > 0.0 {
            let delta_velocity = self.velocity * delta_seconds;
            location.translate(delta_velocity);
            true
        } else {
            false
        };

        let friction_applied = if self.friction > 0.0 {
            // Apply friction to the velocity
            let friction_applied = (self.friction * delta_seconds).min(1.0);
            self.velocity = self.velocity * (1.0 - friction_applied);
            friction_applied
        } else {
            0.0
        };

        // Return info on what has changed
        PhysicsStepInfo {
            position_changed,
            friction_applied,
            ..Default::default()
        }
    }

    /// Apply acceleration (meters per second squared) to the velocity.
    pub fn accelerate(&mut self, acceleration: Vector3<f64>, delta_seconds: f64) {
        // Exit early if delta_seconds is 0 or less
        if delta_seconds < std::f64::EPSILON {
            return Default::default();
        }

        // Add the acceleration to the velocity
        self.velocity = self.velocity + acceleration * delta_seconds;
    }

    /// Apply force (in newtons) to the center of the physics object.
    pub fn apply_force(&mut self, force: Vector3<f64>, delta_seconds: f64) {
        self.accelerate(force / self.mass, delta_seconds)
    }

    /// Apply friction to the center of the physics object.
    pub fn apply_friction(&mut self, friction: f64, delta_seconds: f64) {
        self.velocity = self.velocity * (1.0 - (friction * delta_seconds).min(1.0));
    }
}

pub struct PhysicsStepInfo {
    /// Whether the position changed during the physics step.
    pub position_changed: bool,
    /// Whether the rotation changed during the physics step.
    pub rotation_changed: bool,
    /// How much friction was applied during the physics step (0.0 to 1.0 with 1.0 being a full stop).
    pub friction_applied: f64,
}

impl Default for PhysicsStepInfo {
    fn default() -> Self {
        Self {
            position_changed: false,
            rotation_changed: false,
            friction_applied: 0.0,
        }
    }
}
