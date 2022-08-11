use ggmath::prelude::{Quaternion, Vector3, Matrix3x3};

/// Represents a location in 3D space.
#[derive(Clone, Debug)]
pub struct Location {
    /// The location's position vector
    pub position: Vector3<f32>,
    /// The location's rotation quaternion
    pub rotation: Quaternion<f32>,
    /// The location's scale vector
    pub scale: Vector3<f32>,
}

impl Location {
    /// Creates a new location with the given position, rotation, and scale.
    pub fn new(position: Vector3<f32>, rotation: Quaternion<f32>, scale: Vector3<f32>) -> Location {
        Location {
            position,
            rotation,
            scale,
        }
    }

    /// Delocalizes a direction that is local to this location,
    /// making it local to the parent of the owner of this location instead.
    pub fn delocalize_direction(&self, direction: Vector3<f32>) -> Vector3<f32> {
        Matrix3x3::from(self.rotation) * direction
    }

    /// Delocalizes a position that is local to this location,
    /// making it local to the parent of the owner of this location instead.
    pub fn delocalize_position(&self, position: Vector3<f32>) -> Vector3<f32> {
        Matrix3x3::from(self.rotation) * position + self.position
    }

    /// Delocalizes a rotation that is local to this location,
    /// making it local to the parent of the owner of this location instead.
    pub fn delocalize_rotation(&self, rotation: Quaternion<f32>) -> Quaternion<f32> {
        self.rotation.and_then(&rotation)
    }

    /// Delocalizes a scale that is local to this location,
    /// making it local to the parent of the owner of this location instead.
    pub fn delocalize_scale(&self, scale: Vector3<f32>) -> Vector3<f32> {
        self.scale * scale
    }

    /// Delocalizes a location that is local to this location,
    /// making it local to the parent of the owner of this location instead.
    pub fn delocalize_location(&self, location: Location) -> Location {
        Location {
            position: self.delocalize_position(location.position),
            rotation: self.delocalize_rotation(location.rotation),
            scale: self.delocalize_scale(location.scale),
        }
    }
}

impl Default for Location {
    fn default() -> Self {
        Location {
            position: Vector3::zero(),
            rotation: Quaternion::new_identity(),
            scale: Vector3::one(),
        }
    }
}
