use std::{
    collections::HashMap,
    sync::RwLock,
};

use crate::game::{
    chunk_blueprint::ChunkBlueprint,
    gen_schema::CompiledGenSchema,
    voxel::VoxelVertex,
    *, chunk::{VoxelUnits, ChunkPosition, Chunk, voxels_to_meters, chunks_to_meters, ChunkPositionExt},
};

pub const VOXELS_PER_CHUNK: VoxelUnits = VoxelUnits::from_i64(60i64);
pub const VOXELS_PER_METER: VoxelUnits = VoxelUnits::from_i64(4i64);
pub const WORLD_VIEW_DISTANCE_METERS: f64 = 70.0;
pub const WORLD_VIEW_DISTANCE_VERTICAL_RATIO: f64 = 0.3;
const EXTEND_IN_VIEW_IS_SPHERICAL: bool = true;

define_class! {
    /// The game world.
    pub class World {
        location: Location,
        world_chunks: Chunks,
        schema: CompiledGenSchema,
        world_program: Program,
    }
}

impl World {
    /// Create a new world.
    pub fn new(world_program: Program, schema: &CompiledGenSchema) -> Self {
        // Create mesh to be put in mesh_renderer
        Self {
            location: Location::new(
                vector!(0.0, 0.0, 0.0),
                Quaternion::identity(),
                vector!(1.0, 1.0, 1.0),
            ),
            world_chunks: Chunks::new(),
            schema: schema.clone(),
            world_program,
        }
    }

    /// Remove chunks that have expired.
    pub fn remove_old_chunks(&mut self) {
        self.world_chunks.remove_old_chunks();
    }

    /// Extend the life of all chunks in a given area. Units are in meters.
    pub fn extend_life_in_view(
        &mut self,
        center: Vector3<f64>,
        distance: f64,
        direction: Option<Vector3<f64>>,
    ) -> Option<f64> {
        self.world_chunks.extend_life_in_view(
            center,
            distance,
            WORLD_VIEW_DISTANCE_VERTICAL_RATIO,
            direction,
            &self.schema,
        )
    }

    pub fn create_chunk_meshes(&mut self, gfx: &Gfx) {
        self.world_chunks.create_chunk_meshes(gfx)
    }

    pub fn using_chunk_meshes<'a, F: FnMut(&[(Location, Option<&Mesh<VoxelVertex>>)]) + 'a>(
        &'a self,
        gfx: &'a Gfx,
        f: F,
    ) {
        self.world_chunks.using_chunk_meshes(gfx, f)
    }

    pub fn program(&self) -> &Program {
        &self.world_program
    }

    pub fn set_on_ground(&self, position: Vector3<f64>) -> Vector3<f64> {
        vector!(
            position.x(),
            voxels_to_meters(
                self.schema
                    .elevation_at_world_position(position.xz())
                    .unwrap_or_default()
            ),
            position.z(),
        )
    }
}

#[derive(Debug)]
pub struct Chunks {
    chunks: RwLock<HashMap<ChunkPosition, Chunk>>,
    blueprints: RwLock<HashMap<ChunkPosition, ChunkBlueprint>>,
}

impl Chunks {
    pub fn new() -> Self {
        Self {
            chunks: RwLock::new(HashMap::new()),
            blueprints: RwLock::new(HashMap::new()),
        }
    }

    pub fn using_chunk<F: FnOnce(Option<&Chunk>)>(&self, chunk_position: ChunkPosition, f: F) {
        f(self.chunks.read().unwrap().get(&chunk_position))
    }

    pub fn using_chunk_mut<F: FnOnce(Option<&mut Chunk>)>(
        &self,
        chunk_position: ChunkPosition,
        f: F,
    ) {
        f(self.chunks.write().unwrap().get_mut(&chunk_position))
    }

    pub fn chunk_exists(&self, chunk_position: ChunkPosition) -> bool {
        self.chunks.read().unwrap().contains_key(&chunk_position)
    }

    pub fn using_blueprint<F: FnOnce(&ChunkBlueprint)>(&self, chunk_position: ChunkPosition, f: F) {
        if !self
            .blueprints
            .read()
            .unwrap()
            .contains_key(&chunk_position)
        {
            panic!("Chunk blueprint does not exist: {:?}", chunk_position);
        }
        f(self
            .blueprints
            .read()
            .unwrap()
            .get(&chunk_position)
            .unwrap())
    }

    fn create_blueprint(&self, chunk_position: ChunkPosition, schema: &CompiledGenSchema) {
        if self
            .blueprints
            .read()
            .unwrap()
            .contains_key(&chunk_position)
        {
            return;
        }
        let blueprint = ChunkBlueprint::new(chunk_position, schema);
        self.blueprints
            .write()
            .unwrap()
            .insert(chunk_position, blueprint);
    }

    fn remove_old_chunks(&mut self) {
        self.chunks
            .write()
            .unwrap()
            .retain(|_, chunk| chunk.is_alive());
        let chunks_read = self.chunks.read().unwrap();
        self.blueprints
            .write()
            .unwrap()
            .retain(|chunk_position, _| chunks_read.contains_key(chunk_position));
    }

    fn extend_life(
        &mut self,
        chunk_position: ChunkPosition,
        now: Instant,
        schema: &CompiledGenSchema,
    ) -> Option<f64> {
        let mut returned_generation_time = None;
        if self.chunk_exists(chunk_position) {
            self.using_chunk_mut(chunk_position, |chunk| chunk.unwrap().extend_life(now));
        } else {
            self.create_blueprint(chunk_position, schema);
            self.create_blueprint(chunk_position - Vector::unit_x(), schema);
            self.create_blueprint(chunk_position + Vector::unit_x(), schema);
            self.create_blueprint(chunk_position - Vector::unit_y(), schema);
            self.create_blueprint(chunk_position + Vector::unit_y(), schema);
            self.create_blueprint(chunk_position - Vector::unit_z(), schema);
            self.create_blueprint(chunk_position + Vector::unit_z(), schema);
            self.using_blueprint(chunk_position, |main_blueprint| {
                self.using_blueprint(chunk_position - Vector::unit_x(), |mx_blueprint| {
                    self.using_blueprint(chunk_position + Vector::unit_x(), |px_blueprint| {
                        self.using_blueprint(chunk_position - Vector::unit_y(), |my_blueprint| {
                            self.using_blueprint(
                                chunk_position + Vector::unit_y(),
                                |py_blueprint| {
                                    self.using_blueprint(
                                        chunk_position - Vector::unit_z(),
                                        |mz_blueprint| {
                                            self.using_blueprint(
                                                chunk_position + Vector::unit_z(),
                                                |pz_blueprint| {
                                                    let (chunk, generation_time) = Chunk::new(
                                                        chunk_position.clone(),
                                                        now,
                                                        main_blueprint,
                                                        [
                                                            mx_blueprint,
                                                            px_blueprint,
                                                            my_blueprint,
                                                            py_blueprint,
                                                            mz_blueprint,
                                                            pz_blueprint,
                                                        ],
                                                    );
                                                    returned_generation_time =
                                                        Some(generation_time);
                                                    self.chunks
                                                        .write()
                                                        .unwrap()
                                                        .insert(chunk_position, chunk);
                                                },
                                            )
                                        },
                                    )
                                },
                            )
                        })
                    })
                })
            })
        }
        returned_generation_time
    }

    /// Extend the lifetime of chunks in a given area. Units in meters.
    fn extend_life_in_view(
        &mut self,
        center: Vector3<f64>,
        distance: f64,
        vertical_ratio: f64,
        direction: Option<Vector3<f64>>,
        schema: &CompiledGenSchema,
    ) -> Option<f64> {
        if vertical_ratio <= 0.0 {
            return None;
        }
        let direction = direction.map(|d| d.normalized());
        let now = Instant::now();
        let mut total_generation_time = 0.0;
        let mut total_chunks = 0;
        if EXTEND_IN_VIEW_IS_SPHERICAL {
            if let Some(direction) = direction {
                let chunk_meters = chunks_to_meters(1.0);
                for chunk_position in
                    self.chunk_positions_in_sphere(center, distance, vertical_ratio)
                {
                    let min = chunk_position.to_meters();
                    let corners = [
                        min,
                        min + vector!(chunk_meters, 0.0, 0.0),
                        min + vector!(0.0, chunk_meters, 0.0),
                        min + vector!(chunk_meters, chunk_meters, 0.0),
                        min + vector!(0.0, 0.0, chunk_meters),
                        min + vector!(chunk_meters, 0.0, chunk_meters),
                        min + vector!(0.0, chunk_meters, chunk_meters),
                        min + chunk_meters,
                    ];

                    if corners
                        .into_iter()
                        .any(|corner| (corner - center).normalized().dot(&direction) > 0.0)
                    {
                        if let Some(generation_time) = self.extend_life(chunk_position, now, schema)
                        {
                            total_generation_time += generation_time;
                            total_chunks += 1;
                        }
                    }
                }
            } else {
                for chunk_position in
                    self.chunk_positions_in_sphere(center, distance, vertical_ratio)
                {
                    if let Some(generation_time) = self.extend_life(chunk_position, now, schema) {
                        total_generation_time += generation_time;
                        total_chunks += 1;
                    }
                }
            }
        } else {
            if let Some(direction) = direction {
                let chunk_meters = chunks_to_meters(1.0);
                for chunk_position in self.chunk_positions_in_box(
                    ChunkPosition::from_meters(center - distance * vertical_ratio),
                    ChunkPosition::from_meters(center + distance * vertical_ratio),
                ) {
                    let min = chunk_position.to_meters();
                    let corners = [
                        min,
                        min + vector!(chunk_meters, 0.0, 0.0),
                        min + vector!(0.0, chunk_meters, 0.0),
                        min + vector!(chunk_meters, chunk_meters, 0.0),
                        min + vector!(0.0, 0.0, chunk_meters),
                        min + vector!(chunk_meters, 0.0, chunk_meters),
                        min + vector!(0.0, chunk_meters, chunk_meters),
                        min + chunk_meters,
                    ];

                    if corners
                        .into_iter()
                        .any(|corner| (corner - center).normalized().dot(&direction) > 0.0)
                    {
                        if let Some(generation_time) = self.extend_life(chunk_position, now, schema)
                        {
                            total_generation_time += generation_time;
                            total_chunks += 1;
                        }
                    }
                }
            } else {
                for chunk_position in self.chunk_positions_in_box(
                    ChunkPosition::from_meters(center - distance * vertical_ratio),
                    ChunkPosition::from_meters(center + distance * vertical_ratio),
                ) {
                    if let Some(generation_time) = self.extend_life(chunk_position, now, schema) {
                        total_generation_time += generation_time;
                        total_chunks += 1;
                    }
                }
            }
        }
        if total_chunks > 0 {
            Some(total_generation_time / total_chunks as f64)
        } else {
            None
        }
    }

    fn chunk_positions_in_box(
        &self,
        min: ChunkPosition,
        max: ChunkPosition,
    ) -> impl Iterator<Item = ChunkPosition> {
        (min.z()..=max.z()).flat_map(move |z| {
            (min.y()..=max.y()).flat_map(move |y| {
                (min.x()..=max.x()).map(move |x| {
                    let chunk_position = vector!(x, y, z);
                    chunk_position
                })
            })
        })
    }

    /// Get the positions of all chunks found in a spherical area.
    /// Units are in meters.
    fn chunk_positions_in_sphere(
        &self,
        center: Vector3<f64>,
        distance: f64,
        vertical_ratio: f64,
    ) -> impl Iterator<Item = ChunkPosition> {
        let inv_vertical_ratio = vector!(1.0, 1.0 / vertical_ratio, 1.0);
        let half_chunk = chunks_to_meters(0.5);
        let min = ChunkPosition::from_meters(center - distance);
        let max = ChunkPosition::from_meters(center + distance);
        self.chunk_positions_in_box(min, max)
            .filter_map(move |chunk_position| {
                if ((chunk_position.to_meters() + half_chunk - center) * inv_vertical_ratio)
                    .length_squared()
                    <= distance * distance
                {
                    Some(chunk_position)
                } else {
                    None
                }
            })
    }

    fn create_chunk_meshes(&mut self, gfx: &Gfx) {
        let mut chunks_read = self.chunks.write().unwrap();
        for chunk in chunks_read.values_mut() {
            chunk.generate_mesh(gfx);
        }
    }

    fn using_chunk_meshes<'a, F: FnMut(&[(Location, Option<&Mesh<VoxelVertex>>)]) + 'a>(
        &'a self,
        gfx: &'a Gfx,
        mut f: F,
    ) {
        let chunks_read = self.chunks.read().unwrap();
        let chunks_read_vec: Vec<_> = chunks_read
            .iter()
            .map(|(chunk_position, chunk)| {
                let chunk_meter_position = chunk_position.to_meters();
                let voxel_to_meter = voxels_to_meters(1.0);
                (
                    Location::new(
                        chunk_meter_position,
                        Quaternion::identity(),
                        Vector::from_scalar(voxel_to_meter),
                    ),
                    chunk.get_mesh(gfx),
                )
            })
            .collect();
        f(&chunks_read_vec)
    }
}
