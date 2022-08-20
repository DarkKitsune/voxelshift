use crate::game::*;

use super::{
    voxel::Voxel,
    world::{VoxelPosition, VoxelPositionExt, WorldGenerator},
};

const MAX_ELEVATION: f64 = 50.0;
const MIN_ELEVATION: f64 = -50.0;

pub struct BasicWorldGenerator {
    seed: u64,
    elevation_noise: Noise<2>,
}

impl BasicWorldGenerator {
    pub fn new(seed: u64) -> Self {
        let elevation_noise = Noise::<2>::new(seed, 5, 4.0, 4.0, 0.1);
        Self {
            seed,
            elevation_noise,
        }
    }
}

impl WorldGenerator for BasicWorldGenerator {
    fn new_constructor(&self) -> Box<dyn FnMut(world::VoxelPosition) -> Option<Voxel>> {
        let seed = self.seed;
        let color_noise = Noise::<3>::new(seed.wrapping_mul(12345), 4, 4.0, 3.0, 0.4);
        let elevation_noise = self.elevation_noise.clone();
        Box::new(move |voxel_position: VoxelPosition| {
            let voxel_center = voxel_position.center_position();
            let base_elevation = elevation_noise.sample_f64(voxel_center.xz());
            let elevation = MIN_ELEVATION.lerp(MAX_ELEVATION, base_elevation);
            if voxel_center.y() < elevation {
                let r = 0.2 + 0.8 * color_noise.sample_f64(voxel_center);
                let g = 0.7 + 0.3 * color_noise.sample_f64(voxel_center + 2000.0);
                let b = 0.2 + 0.8 * color_noise.sample_f64(voxel_center + 4000.0);
                Some(Voxel::new(Color::new(r as f32, g as f32, b as f32, 1.0)))
            } else {
                None
            }
        })
    }

    fn elevation_at(&self, xz: Vector2<f64>) -> f64 {
        let base_elevation = self.elevation_noise.sample_f64(xz);
        MIN_ELEVATION.lerp(MAX_ELEVATION, base_elevation)
    }
}
