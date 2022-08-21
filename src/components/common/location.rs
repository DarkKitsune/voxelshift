use ggmath::prelude::*;

/// Represents a location in 3D space.
#[derive(Clone, Debug)]
pub struct Location {
    /// The location's position vector
    position: Vector3<f64>,
    /// The location's rotation quaternion
    rotation: Quaternion<f64>,
    /// The location's scale vector
    scale: Vector3<f64>,
}

impl Location {
    /// Creates a new location with the given position, rotation, and scale.
    pub fn new(position: Vector3<f64>, rotation: Quaternion<f64>, scale: Vector3<f64>) -> Location {
        #[cfg(debug_assertions)]
        {
            if position.x().is_nan() || position.y().is_nan() || position.z().is_nan() {
                panic!("Location::new: position is NaN");
            }
            if rotation.x().is_nan()
                || rotation.y().is_nan()
                || rotation.z().is_nan()
                || rotation.w().is_nan()
            {
                panic!("Location::new: rotation is NaN");
            }
            if scale.x().is_nan() || scale.y().is_nan() || scale.z().is_nan() {
                panic!("Location::new: scale is NaN");
            }
            if rotation.length_squared() < f64::EPSILON {
                panic!(
                    "Attempted to set the create a Location with a zero-length rotation quaternion"
                );
            }
        }
        Location {
            position,
            rotation,
            scale,
        }
    }

    /// Returns the location's position vector.
    pub fn position(&self) -> Vector3<f64> {
        self.position
    }

    /// Sets the location's position vector.
    pub fn set_position(&mut self, position: Vector3<f64>) {
        #[cfg(debug_assertions)]
        if position.x().is_nan() || position.y().is_nan() || position.z().is_nan() {
            panic!("Location::new: position is NaN");
        }
        self.position = position;
    }

    /// Changes the location's position vector by the given amount.
    pub fn translate(&mut self, amount: Vector3<f64>) {
        self.set_position(self.position() + amount);
    }

    /// Returns the location's rotation quaternion.
    pub fn rotation(&self) -> Quaternion<f64> {
        self.rotation
    }

    /// Sets the location's rotation quaternion.
    pub fn set_rotation(&mut self, rotation: Quaternion<f64>) {
        #[cfg(debug_assertions)]
        {
            if rotation.x().is_nan()
                || rotation.y().is_nan()
                || rotation.z().is_nan()
                || rotation.w().is_nan()
            {
                panic!("Location::new: rotation is NaN");
            }
            if rotation.length_squared() < f64::EPSILON {
                panic!("Attempted to set the location's rotation to a zero-length quaternion");
            }
        }
        self.rotation = rotation;
    }

    /// Changes the location's rotation quaternion by the given amount.
    pub fn rotate(&mut self, amount: Quaternion<f64>) {
        self.set_rotation(self.rotation().and_then(&amount));
    }

    /// Returns the location's scale vector.
    pub fn scale(&self) -> Vector3<f64> {
        self.scale
    }

    /// Sets the location's scale vector.
    pub fn set_scale(&mut self, scale: Vector3<f64>) {
        #[cfg(debug_assertions)]
        if scale.x().is_nan() || scale.y().is_nan() || scale.z().is_nan() {
            panic!("Location::new: scale is NaN");
        }
        self.scale = scale;
    }

    /// Returns the location's rotation matrix.
    pub fn rotation_matrix(&self) -> Matrix3x3<f64> {
        self.rotation.to_matrix()
    }

    /// Returns the location's transformation matrix.
    pub fn transformation_matrix(&self) -> Matrix4x4<f64> {
        Matrix4x4::from(self.rotation())
            .and_then(&Matrix4x4::new_scale(&self.scale()))
            .and_then(&Matrix4x4::new_translation(&self.position()))
    }

    /// Delocalizes a direction that is local to this location,
    /// making it local to the parent of the owner of this location instead.
    pub fn delocalize_direction(&self, direction: Vector3<f64>) -> Vector3<f64> {
        direction.rotated_by(&self.rotation)
    }

    pub fn localize_direction(&self, direction: Vector3<f64>) -> Vector3<f64> {
        direction.rotated_by(&self.rotation.inverted())
    }

    /// Delocalizes a position that is local to this location,
    /// making it local to the parent of the owner of this location instead.
    pub fn delocalize_position(&self, position: Vector3<f64>) -> Vector3<f64> {
        position.rotated_by(&self.rotation) + self.position
    }

    pub fn localize_position(&self, position: Vector3<f64>) -> Vector3<f64> {
        (position - self.position).rotated_by(&self.rotation.inverted())
    }

    /// Delocalizes a rotation that is local to this location,
    /// making it local to the parent of the owner of this location instead.
    pub fn delocalize_rotation(&self, rotation: Quaternion<f64>) -> Quaternion<f64> {
        rotation.and_then(&self.rotation)
    }

    pub fn localize_rotation(&self, rotation: Quaternion<f64>) -> Quaternion<f64> {
        rotation.and_then(&self.rotation.inverted())
    }

    /// Delocalizes a scale that is local to this location,
    /// making it local to the parent of the owner of this location instead.
    pub fn delocalize_scale(&self, scale: Vector3<f64>) -> Vector3<f64> {
        self.scale * scale
    }

    pub fn localize_scale(&self, scale: Vector3<f64>) -> Vector3<f64> {
        self.scale / scale
    }

    /// Delocalizes a location that is local to this location,
    /// making it local to the parent of the owner of this location instead.
    pub fn delocalize_location(&self, location: Location) -> Location {
        Location::new(
            self.delocalize_position(location.position),
            self.delocalize_rotation(location.rotation),
            self.delocalize_scale(location.scale),
        )
    }

    pub fn localize_location(&self, location: Location) -> Location {
        Location::new(
            self.localize_position(location.position),
            self.localize_rotation(location.rotation),
            self.localize_scale(location.scale),
        )
    }
}

impl Default for Location {
    fn default() -> Self {
        Location {
            position: Vector3::zero(),
            rotation: Quaternion::identity(),
            scale: Vector3::one(),
        }
    }
}
