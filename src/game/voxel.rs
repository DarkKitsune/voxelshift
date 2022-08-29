use crate::{game::*, vertex_struct};

#[derive(Debug)]
pub struct Voxel {
    class: VoxelClass,
    normal: Vector3<f32>,
}

impl Voxel {
    pub fn new(class: VoxelClass, normal: Vector3<f32>, seed: u64) -> Self {
        let mut normal_lcg = NormalLcg::new(seed);
        let normal = if let Some(randomization) = class.normal_randomization() {
            let next_normal = normal_lcg.next_normal_f32();
            (normal + next_normal * randomization).normalized()
        } else {
            normal
        };

        Self { class, normal }
    }
}

impl Voxel {
    pub fn color(&self) -> Color {
        self.class.color()
    }

    pub fn normal(&self) -> Vector3<f32> {
        self.normal
    }

    pub fn is_transparent(&self) -> bool {
        self.class.is_transparent()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VoxelClass {
    Grass,
    Water,
}

impl VoxelClass {
    pub fn color(&self) -> Color {
        match self {
            VoxelClass::Grass => Color::new(0.35, 0.8, 0.1, 1.0),
            VoxelClass::Water => Color::new(0.0, 0.1, 0.9, 1.0),
        }
    }

    pub fn is_transparent(&self) -> bool {
        match self {
            _ => false,
        }
    }

    pub fn normal_randomization(&self) -> Option<f32> {
        match self {
            VoxelClass::Grass => Some(0.2),
            _ => None,
        }
    }
}

vertex_struct! {
    /// Basic debug vertex with a position, normal, and color.
    #[derive(Debug)]
    pub struct VoxelVertex {
        position: Vector3<f32> = |proto| proto.position(),
        color: Vector4<f32> = |proto| proto.color_or_panic().into(),
        normal: Vector3<f32> = |proto| proto.normal().unwrap_or_default(),
    }
}
