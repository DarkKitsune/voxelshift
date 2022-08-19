use ggmath::prelude::*;
use ggutil::prelude::Handle;

#[derive(Clone, Debug)]
pub struct Camera {
    target_node: Option<Handle>,
    rotation: Quaternion<f32>,
    camera_projection: CameraProjection,
}

impl Camera {
    /// Create a new camera
    fn new(target_node: Option<Handle>, camera_projection: CameraProjection) -> Self {
        Self {
            target_node,
            rotation: Quaternion::identity(),
            camera_projection,
        }
    }

    /// Create a new perspective camera
    pub fn new_perspective(target_node: Option<Handle>, fov: f32, near: f32, far: f32) -> Self {
        Self::new(target_node, CameraProjection::Perspective {
            fov,
            near,
            far,
        })
    }

    /// Create a new orthographic camera
    pub fn new_orthographic(target_node: Option<Handle>, size: OrthographicSize, near: f32, far: f32) -> Self {
        Self::new(target_node, CameraProjection::Orthographic {
            size,
            near,
            far,
        })
    }

    /// Set the camera target node
    pub fn set_target(&mut self, target_node: Option<Handle>) {
        self.target_node = target_node;
    }

    /// Get the camera target node
    pub fn target(&self) -> Option<&Handle> {
        self.target_node.as_ref()
    }

    /// Set the camera rotation
    pub fn set_rotation(&mut self, rotation: Quaternion<f32>) {
        self.rotation = rotation;
    }

    /// Get the camera rotation
    pub fn rotation(&self) -> Quaternion<f32> {
        self.rotation
    }

    /// Create a projection matrix for the camera
    pub fn projection_matrix(&self, aspect_ratio: f32) -> Matrix4x4<f32> {
        match self.camera_projection.clone() {
            CameraProjection::Perspective { fov, near, far } =>
                Matrix4x4::new_projection_perspective(fov, aspect_ratio, near, far),
            CameraProjection::Orthographic { size, near, far } =>
                Matrix4x4::new_projection_orthographic(size.size(aspect_ratio), near, far),
        }
    }
}

#[derive(Clone, Debug)]
pub enum CameraProjection {
    Perspective {
        fov: f32,
        near: f32,
        far: f32,
    },
    Orthographic {
        size: OrthographicSize,
        near: f32,
        far: f32,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum OrthographicSize {
    Fixed(Vector2<f32>),
    FixedWidth(f32),
    FixedHeight(f32),
}

impl OrthographicSize {
    /// Return the width of the orthographic camera projection in world units
    /// Respects the aspect ratio of the render target.
    pub fn width(&self, aspect_ratio: f32) -> f32 {
        match self {
            Self::Fixed(size) => size.x(),
            Self::FixedWidth(width) => *width,
            Self::FixedHeight(height) => *height * aspect_ratio,
        }
    }

    /// Returns the height of the orthographic camera projection in world units,
    /// Respects the aspect ratio of the render target.
    pub fn height(&self, aspect_ratio: f32) -> f32 {
        match self {
            Self::Fixed(size) => size.y(),
            Self::FixedWidth(width) => *width / aspect_ratio,
            Self::FixedHeight(height) => *height,
        }
    }

    /// Return the size of the orthographic camera projection in world units
    /// Respects the aspect ratio of the render target.
    pub fn size(&self, aspect_ratio: f32) -> Vector2<f32> {
        vector!(self.width(aspect_ratio), self.height(aspect_ratio))
    }
}