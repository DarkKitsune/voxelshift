use std::sync::Arc;

use num::ToPrimitive;

use crate::game::*;

use super::{chunk_blueprint::VoxelBlueprint, world::{ChunkPosition, VOXELS_PER_CHUNK, VoxelPosition, VoxelPositionExt, VoxelUnits, ChunkPositionExt, meters_to_voxels}};

pub struct GenSchema {
    instructions: Vec<GenInstruction>,
}

impl GenSchema {
    pub fn new(instructions: impl IntoIterator<Item = GenInstruction>) -> Self {
        Self {
            instructions: instructions.into_iter().collect(),
        }
    }

    pub fn compile(&self, seed: u64) -> CompiledGenSchema {
        CompiledGenSchema::new(&self.instructions, seed)
    }
}

#[derive(Debug, Clone)]
pub enum GenInstruction {
    Terrain {
        elevation_bounds: (f64, f64),
        smoothness: f64,
    },
    LiquidVolume {
        y_level: VoxelUnits,
        min_volume: usize,
    },
}

impl GenInstruction {
    fn compile(&self, seed: u64) -> CompiledInstruction {
        match self {
            &Self::Terrain { elevation_bounds, smoothness } =>
                CompiledInstruction::Terrain {
                    elevation_bounds,
                    noise: Noise::<2>::new(seed, 6, 6.0, smoothness, 0.01),
                },
            &Self::LiquidVolume { y_level, min_volume } =>
                CompiledInstruction::LiquidVolume {
                    y_level,
                    min_volume,
                },
        }
    }
}

struct CompiledGenSchemaBase {
    instructions: Vec<CompiledInstruction>,
}

#[derive(Clone)]
pub struct CompiledGenSchema {
    base: Arc<CompiledGenSchemaBase>,
}

impl CompiledGenSchema {
    fn new<'a>(instructions: impl IntoIterator<Item = &'a GenInstruction> + 'a, seed: u64) -> Self {
        let mut instructions: Vec<_> = instructions.into_iter().enumerate().map(|(idx, i)| i.compile(seed.wrapping_mul((idx + 1) as u64))).collect();
        instructions.sort_by_key(CompiledInstruction::priority);
        Self {
            base: Arc::new(CompiledGenSchemaBase {
                instructions,
            }),
        }
    }

    pub fn generate(&self, chunk_position: ChunkPosition) -> Vec<VoxelBlueprint> {
        let mut voxels = vec![VoxelBlueprint::Air; VOXELS_PER_CHUNK.into_i64().pow(3) as usize];
        for instruction in &self.base.instructions {
            instruction.apply(chunk_position, &mut voxels);
        }
        voxels
    }

    pub fn elevation_at_voxel_position(&self, voxel_position: Vector2<f64>) -> Option<f64> {
        self.base.instructions.iter().find_map(|i| i.elevation_at_voxel_position(voxel_position))
    }

    pub fn elevation_at_world_position(&self, world_position: Vector2<f64>) -> Option<f64> {
        self.elevation_at_voxel_position(world_position.map(|c| meters_to_voxels(*c)))
    }
}

pub enum CompiledInstruction {
    Terrain {
        elevation_bounds: (f64, f64),
        noise: Noise<2>,
    },
    LiquidVolume {
        y_level: VoxelUnits,
        min_volume: usize,
    },
}

impl CompiledInstruction {
    fn priority(&self) -> usize {
        match self {
            Self::Terrain { .. } => 0,
            Self::LiquidVolume { .. } => 1,
        }
    }

    fn apply(&self, chunk_position: ChunkPosition, voxels: &mut Vec<VoxelBlueprint>) {
        match self {
            Self::Terrain { elevation_bounds, noise } => {
                let min_elevation = elevation_bounds.0.to_f64().unwrap();
                let max_elevation = elevation_bounds.1.to_f64().unwrap();
                let columns: Vec<_> = for_each_column(chunk_position).collect();
                for (xz, column) in columns {
                    let column_center = xz.map(|c| c.into_f64() + 0.5);
                    let elevation = min_elevation.lerp(max_elevation, noise.sample_f64(column_center));
                    for (y, idx) in column {
                        let voxel_center = vector!(xz.x(), y, xz.y()).center_position();
                        if voxel_center.y() < elevation {
                            voxels[idx] = VoxelBlueprint::TerrainFiller;
                        }
                    }
                }
            },
            Self::LiquidVolume { y_level, min_volume } => {
                let mut volume = 0;
                let mut indices = Vec::new();
                for (_xz, column) in for_each_column(chunk_position) {
                    for (y, idx) in column {
                        if y < *y_level && matches!(voxels[idx], VoxelBlueprint::Air) {
                            indices.push(idx);
                            volume += 1;
                        }
                    }
                }
                if volume >= *min_volume {
                    for idx in indices {
                        voxels[idx] = VoxelBlueprint::Liquid;
                    }
                }
            },
        }
    }

    pub fn elevation_at_voxel_position(&self, voxel_position: Vector2<f64>) -> Option<f64> {
        match self {
            Self::Terrain { elevation_bounds, noise } => {
                let min_elevation = elevation_bounds.0.to_f64().unwrap();
                let max_elevation = elevation_bounds.1.to_f64().unwrap();
                let elevation = min_elevation.lerp(max_elevation, noise.sample_f64(voxel_position));
                Some(elevation)
            },
            Self::LiquidVolume { .. } => None,
        }
    }
}

fn for_each_voxel(voxels: &mut [VoxelBlueprint], chunk_position: ChunkPosition) -> impl Iterator<Item = (VoxelPosition, &mut VoxelBlueprint)> {
    const VOXELS_PER_CHUNK_I64: i64 = VOXELS_PER_CHUNK.into_i64();
    let chunk_voxel_position = chunk_position.to_voxel_position();
    voxels.iter_mut().enumerate().map(move |(idx, voxel)| {
        let idx = idx as i64;
        let x = idx % VOXELS_PER_CHUNK_I64;
        let y = (idx / VOXELS_PER_CHUNK_I64) % VOXELS_PER_CHUNK_I64;
        let z = idx / (VOXELS_PER_CHUNK_I64 * VOXELS_PER_CHUNK_I64);
        let voxel_position = chunk_voxel_position + vector!(x.into(), y.into(), z.into());
        (voxel_position, voxel)
    })
}

fn for_each_column<'a>(chunk_position: ChunkPosition) -> impl Iterator<Item = (Vector2<VoxelUnits>, impl Iterator<Item = (VoxelUnits, usize)>)> + 'a {
    const VOXELS_PER_CHUNK_I64: i64 = VOXELS_PER_CHUNK.into_i64();
    let chunk_voxel_position = chunk_position.to_voxel_position();
    (0 .. VOXELS_PER_CHUNK_I64).flat_map(move |x| {
        (0 .. VOXELS_PER_CHUNK_I64).map(move |z| {
            let column_position = vector!(x.into(), z.into());
            (
                chunk_voxel_position.xz() + column_position,
                (0..VOXELS_PER_CHUNK_I64).map(move |y| (chunk_voxel_position.y() + y.into(), (z + y * VOXELS_PER_CHUNK_I64 + x * VOXELS_PER_CHUNK_I64.pow(2)) as usize)),
            )
        })
    })
}