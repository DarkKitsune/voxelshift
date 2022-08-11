use ggmath::quaternion::Quaternion;
use ggutil::prelude::Handle;

pub struct Camera {
    pub target_node: Option<Handle>,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub rotation: Quaternion<f32>,
}
