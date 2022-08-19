use crate::game::*;

#[derive(Debug)]
pub struct Voxel {
    color: Color,
}

impl Voxel {
    pub fn new(color: Color) -> Self {
        Self { color }
    }
}

impl Voxel {
    pub fn color(&self) -> Color {
        self.color
    }
}
