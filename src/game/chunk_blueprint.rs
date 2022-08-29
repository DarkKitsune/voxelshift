use ggmath::vector;

use super::{
    gen_schema::CompiledGenSchema,
    world::{VOXELS_PER_CHUNK}, chunk::{ChunkPosition, VoxelPosition, VoxelUnits},
};

const VOXELS_PER_CHUNK_I64: i64 = VOXELS_PER_CHUNK.into_i64();
const VOXELS_PER_CHUNK_USIZE: usize = VOXELS_PER_CHUNK_I64 as usize;
pub const CHUNK_LIFE_SECONDS: u64 = 9;

#[derive(Debug)]
pub struct ChunkBlueprint {
    voxels: Vec<VoxelBlueprint>,
}

impl ChunkBlueprint {
    pub fn new(chunk_position: ChunkPosition, schema: &CompiledGenSchema) -> Self {
        Self {
            voxels: schema.generate(chunk_position),
        }
    }

    pub fn get(&self, position_in_chunk: VoxelPosition) -> VoxelBlueprint {
        let idx = (position_in_chunk.x().into_i64() * VOXELS_PER_CHUNK_I64.pow(2)
            + position_in_chunk.y().into_i64() * VOXELS_PER_CHUNK_I64
            + position_in_chunk.z().into_i64()) as usize;
        self.voxels.get(idx).copied().unwrap_or(VoxelBlueprint::Air)
    }

    pub fn may_be_visible(&self, position_in_chunk: VoxelPosition) -> bool {
        let idx = (position_in_chunk.x().into_i64() * VOXELS_PER_CHUNK_I64.pow(2)
            + position_in_chunk.y().into_i64() * VOXELS_PER_CHUNK_I64
            + position_in_chunk.z().into_i64()) as usize;
        position_in_chunk.x() == VoxelUnits::from_u64(0)
            || position_in_chunk.x() == VOXELS_PER_CHUNK - VoxelUnits::from_u64(1)
            || position_in_chunk.y() == VoxelUnits::from_u64(0)
            || position_in_chunk.y() == VOXELS_PER_CHUNK - VoxelUnits::from_u64(1)
            || position_in_chunk.z() == VoxelUnits::from_u64(0)
            || position_in_chunk.z() == VOXELS_PER_CHUNK - VoxelUnits::from_u64(1)
            || self.voxels[idx - 1].may_be_transparent()
            || self.voxels[idx + 1].may_be_transparent()
            || self.voxels[idx - VOXELS_PER_CHUNK_USIZE].may_be_transparent()
            || self.voxels[idx + VOXELS_PER_CHUNK_USIZE].may_be_transparent()
            || self.voxels[idx - VOXELS_PER_CHUNK_USIZE.pow(2)].may_be_transparent()
            || self.voxels[idx + VOXELS_PER_CHUNK_USIZE.pow(2)].may_be_transparent()
    }

    pub fn to_group<'a>(&'a self, neighbors: [&'a ChunkBlueprint; 6]) -> ChunkBlueprintGroup<'a> {
        ChunkBlueprintGroup {
            main: self,
            neighbors,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VoxelBlueprint {
    Air,
    TerrainFiller,
    Liquid,
    Other,
}

impl VoxelBlueprint {
    fn may_be_transparent(self) -> bool {
        match self {
            Self::Air => true,
            Self::TerrainFiller => false,
            Self::Liquid => true,
            Self::Other => true,
        }
    }
}

pub struct ChunkBlueprintGroup<'a> {
    main: &'a ChunkBlueprint,
    neighbors: [&'a ChunkBlueprint; 6],
}

impl<'a> ChunkBlueprintGroup<'a> {
    pub fn get(&self, position_in_chunk: VoxelPosition) -> VoxelBlueprint {
        if position_in_chunk.x() < VoxelUnits::from_i64(0) {
            self.neighbors[0].get(
                (position_in_chunk
                    + vector!(
                        VOXELS_PER_CHUNK,
                        VoxelUnits::from_i64(0),
                        VoxelUnits::from_i64(0)
                    ))
                .into(),
            )
        } else if position_in_chunk.x() >= VOXELS_PER_CHUNK {
            self.neighbors[1].get(
                (position_in_chunk
                    - vector!(
                        VOXELS_PER_CHUNK,
                        VoxelUnits::from_i64(0),
                        VoxelUnits::from_i64(0)
                    ))
                .into(),
            )
        } else if position_in_chunk.y() < VoxelUnits::from_i64(0) {
            self.neighbors[2].get(
                (position_in_chunk
                    + vector!(
                        VoxelUnits::from_i64(0),
                        VOXELS_PER_CHUNK,
                        VoxelUnits::from_i64(0)
                    ))
                .into(),
            )
        } else if position_in_chunk.y() >= VOXELS_PER_CHUNK {
            self.neighbors[3].get(
                (position_in_chunk
                    - vector!(
                        VoxelUnits::from_i64(0),
                        VOXELS_PER_CHUNK,
                        VoxelUnits::from_i64(0)
                    ))
                .into(),
            )
        } else if position_in_chunk.z() < VoxelUnits::from_i64(0) {
            self.neighbors[4].get(
                (position_in_chunk
                    + vector!(
                        VoxelUnits::from_i64(0),
                        VoxelUnits::from_i64(0),
                        VOXELS_PER_CHUNK
                    ))
                .into(),
            )
        } else if position_in_chunk.z() >= VOXELS_PER_CHUNK {
            self.neighbors[5].get(
                (position_in_chunk
                    - vector!(
                        VoxelUnits::from_i64(0),
                        VoxelUnits::from_i64(0),
                        VOXELS_PER_CHUNK
                    ))
                .into(),
            )
        } else {
            self.main.get(position_in_chunk.into())
        }
    }

    pub fn may_be_visible(&self, position_in_chunk: VoxelPosition) -> bool {
        if position_in_chunk.x() < VoxelUnits::from_i64(0) {
            self.neighbors[0].may_be_visible(
                (position_in_chunk
                    + vector!(
                        VOXELS_PER_CHUNK,
                        VoxelUnits::from_i64(0),
                        VoxelUnits::from_i64(0)
                    ))
                .into(),
            )
        } else if position_in_chunk.x() >= VOXELS_PER_CHUNK {
            self.neighbors[1].may_be_visible(
                (position_in_chunk
                    - vector!(
                        VOXELS_PER_CHUNK,
                        VoxelUnits::from_i64(0),
                        VoxelUnits::from_i64(0)
                    ))
                .into(),
            )
        } else if position_in_chunk.y() < VoxelUnits::from_i64(0) {
            self.neighbors[2].may_be_visible(
                (position_in_chunk
                    + vector!(
                        VoxelUnits::from_i64(0),
                        VOXELS_PER_CHUNK,
                        VoxelUnits::from_i64(0)
                    ))
                .into(),
            )
        } else if position_in_chunk.y() >= VOXELS_PER_CHUNK {
            self.neighbors[3].may_be_visible(
                (position_in_chunk
                    - vector!(
                        VoxelUnits::from_i64(0),
                        VOXELS_PER_CHUNK,
                        VoxelUnits::from_i64(0)
                    ))
                .into(),
            )
        } else if position_in_chunk.z() < VoxelUnits::from_i64(0) {
            self.neighbors[4].may_be_visible(
                (position_in_chunk
                    + vector!(
                        VoxelUnits::from_i64(0),
                        VoxelUnits::from_i64(0),
                        VOXELS_PER_CHUNK
                    ))
                .into(),
            )
        } else if position_in_chunk.z() >= VOXELS_PER_CHUNK {
            self.neighbors[5].may_be_visible(
                (position_in_chunk
                    - vector!(
                        VoxelUnits::from_i64(0),
                        VoxelUnits::from_i64(0),
                        VOXELS_PER_CHUNK
                    ))
                .into(),
            )
        } else {
            self.main.may_be_visible(position_in_chunk.into())
        }
    }
}
