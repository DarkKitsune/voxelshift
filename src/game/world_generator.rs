use crate::game::*;

use super::{
    voxel::{Voxel, VoxelClass},
    world::{VoxelPosition, VoxelPositionExt, WorldGenerator},
};

const MAX_ELEVATION: f64 = 200.0;
const MIN_ELEVATION: f64 = -170.0;
const WATER_LEVEL: f64 = 0.0;

pub struct BasicWorldGenerator {
    seed: u64,
    elevation_noise: Noise<2>,
}

impl BasicWorldGenerator {
    pub fn new(seed: u64) -> Self {
        let elevation_noise = Noise::<2>::new(seed, 6, 5.0, 3.0, 0.03);
        Self {
            seed,
            elevation_noise,
        }
    }
}

impl WorldGenerator for BasicWorldGenerator {
    fn new_constructor(&self) -> Box<dyn FnMut(world::VoxelPosition) -> Option<Voxel>> {
        let elevation_noise = self.elevation_noise.clone();
        Box::new(move |voxel_position: VoxelPosition| {
            let voxel_center = voxel_position.center_position();
            let center_elevation = elevation_at(&elevation_noise, voxel_center.xz());

            let voxel_info = if voxel_center.y() <= center_elevation + 0.5 && voxel_center.y() > center_elevation - 0.5 {
                let normal = average_normal(&elevation_noise, voxel_center, center_elevation, 1.5, 2);
                
                Some((VoxelClass::Grass, normal.convert_to().unwrap()))
            } else if voxel_center.y() < WATER_LEVEL {
                Some((VoxelClass::Water, vector!(0.0, 1.0, 0.0)))
            } else {
                None
            };
            
            if let Some((voxel_class, normal)) = voxel_info {
                let seed = voxel_position.map(|&c| c.into_i64()).to_seed();
                Some(Voxel::new(voxel_class, normal, seed))
            }
            else {
                None
            }
        })
    }

    fn elevation_at(&self, xz: Vector2<f64>) -> f64 {
        let base_elevation = self.elevation_noise.sample_f64(xz);
        MIN_ELEVATION.lerp(MAX_ELEVATION, base_elevation)
    }
}

fn elevation_at(elevation_noise: &Noise<2>, xz: Vector2<f64>) -> f64 {
    let base_elevation = elevation_noise.sample_f64(xz);
    MIN_ELEVATION.lerp(MAX_ELEVATION, base_elevation)
}

fn average_normal(elevation_noise: &Noise<2>, voxel_center: Vector3<f64>, center_elevation: f64, distance: f64, iterations: usize) -> Vector3<f64> {
    let mut total = vector!(0.0, 0.0, 0.0);
    for iteration in 1..=iterations {
        let distance = if iteration % 2 == 0 {
            distance * (iteration as f64 / iterations as f64)
        }
        else {
            -distance * (iteration as f64 / iterations as f64)
        };
        let p_center = vector!(voxel_center.x(), center_elevation, voxel_center.z());
        let elevation_x = elevation_at(elevation_noise, voxel_center.xz() + vector!(distance, 0.0));
        let p_x = vector!(voxel_center.x() + distance, elevation_x, voxel_center.z());
        let elevation_z = elevation_at(elevation_noise, voxel_center.xz() + vector!(0.0, distance));
        let p_z = vector!(voxel_center.x(), elevation_z, voxel_center.z() + distance);
        let normal = (p_z - p_center).cross(&(p_x - p_center)).normalized();
        total = total + normal;
    }
    (total / iterations as f64).normalized()
}