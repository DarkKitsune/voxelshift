use std::{
    collections::HashMap,
    convert::identity,
    iter::{Step, Sum}, ops::{Div, Mul}, sync::RwLock,
};

use auto_ops::impl_op_ex;
use ggmath::init_array;
use num::{One, Zero};

use crate::{game::{voxel::{Voxel, VoxelVertex, VoxelClass}, *, chunk_blueprint::{ChunkBlueprint, VoxelBlueprint, ChunkBlueprintGroup}, gen_schema::CompiledGenSchema}};

pub const VOXELS_PER_CHUNK: VoxelUnits = VoxelUnits(60);
pub const VOXELS_PER_METER: VoxelUnits = VoxelUnits(4);
pub const WORLD_VIEW_DISTANCE_METERS: f64 = 70.0;
pub const WORLD_VIEW_DISTANCE_VERTICAL_RATIO: f64 = 0.3;
const CHUNK_LIFE_SECONDS: u64 = 9;
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
    pub fn new(
        world_program: Program,
        schema: &CompiledGenSchema,
    ) -> Self {
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
    pub fn extend_life_in_view(&mut self, center: Vector3<f64>, distance: f64, direction: Option<Vector3<f64>>) {
        self.world_chunks
            .extend_life_in_view(center, distance, WORLD_VIEW_DISTANCE_VERTICAL_RATIO, direction, &self.schema);
    }

    pub fn create_chunk_meshes(&mut self, gfx: &Gfx) -> (f64, usize) {
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
            voxels_to_meters(self.schema.elevation_at_world_position(position.xz()).unwrap_or_default()),
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

    pub fn using_chunk_mut<F: FnOnce(Option<&mut Chunk>)>(&self, chunk_position: ChunkPosition, f: F) {
        f(self.chunks.write().unwrap().get_mut(&chunk_position))
    }

    pub fn chunk_exists(&self, chunk_position: ChunkPosition) -> bool {
        self.chunks.read().unwrap().contains_key(&chunk_position)
    }

    pub fn using_blueprint<F: FnOnce(&ChunkBlueprint)>(&self, chunk_position: ChunkPosition, f: F) {
        if !self.blueprints.read().unwrap().contains_key(&chunk_position) {
            panic!("Chunk blueprint does not exist: {:?}", chunk_position);
        }
        f(self.blueprints.read().unwrap().get(&chunk_position).unwrap())
    }

    fn create_blueprint(&self, chunk_position: ChunkPosition, schema: &CompiledGenSchema) {
        if self.blueprints.read().unwrap().contains_key(&chunk_position) {
            return;
        }
        let blueprint = ChunkBlueprint::new(chunk_position, schema);
        self.blueprints.write().unwrap().insert(chunk_position, blueprint);
    }

    fn remove_old_chunks(&mut self) {
        self.chunks.write().unwrap().retain(|_, chunk| chunk.is_alive());
        let chunks_read = self.chunks.read().unwrap();
        self.blueprints.write().unwrap().retain(|chunk_position, _| chunks_read.contains_key(chunk_position));
    }

    fn extend_life(
        &mut self,
        chunk_position: ChunkPosition,
        now: Instant,
        schema: &CompiledGenSchema,
    ) {
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
                            self.using_blueprint(chunk_position + Vector::unit_y(), |py_blueprint| {
                                self.using_blueprint(chunk_position - Vector::unit_z(), |mz_blueprint| {
                                    self.using_blueprint(chunk_position + Vector::unit_z(), |pz_blueprint| {
                                        self.chunks.write().unwrap().insert(
                                            chunk_position,
                                            Chunk::new(
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
                                                ]
                                            ),
                                        );
                                    })
                                })
                            })
                        })
                    })
                })
            })
        }
    }

    /// Extend the lifetime of chunks in a given area. Units in meters.
    fn extend_life_in_view(
        &mut self,
        center: Vector3<f64>,
        distance: f64,
        vertical_ratio: f64,
        direction: Option<Vector3<f64>>,
        schema: &CompiledGenSchema,
    ) {
        if vertical_ratio <= 0.0 {
            return;
        }
        let direction = direction.map(|d| d.normalized());
        let now = Instant::now();
        if EXTEND_IN_VIEW_IS_SPHERICAL {
            if let Some(direction) = direction {
                let chunk_meters = chunks_to_meters(1.0);
                for chunk_position in self.chunk_positions_in_sphere(center, distance, vertical_ratio) {
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
                        self.extend_life(chunk_position, now, schema);
                    }
                }
            }
            else {
                for chunk_position in self.chunk_positions_in_sphere(center, distance, vertical_ratio) {
                    self.extend_life(chunk_position, now, schema);
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
                        self.extend_life(chunk_position, now, schema);
                    }
                }
            }
            else {
                for chunk_position in self.chunk_positions_in_box(
                    ChunkPosition::from_meters(center - distance * vertical_ratio),
                    ChunkPosition::from_meters(center + distance * vertical_ratio),
                ) {
                    self.extend_life(chunk_position, now, schema);
                }
            }
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
                if ((chunk_position.to_meters() + half_chunk - center) * inv_vertical_ratio).length_squared() <= distance * distance {
                    Some(chunk_position)
                } else {
                    None
                }
            })
    }

    fn create_chunk_meshes(&mut self, gfx: &Gfx) -> (f64, usize) {
        let mut total_time = 0.0;
        let mut total_count = 0;
        let mut chunks_read = self.chunks.write().unwrap();
        for chunk in chunks_read.values_mut() {
            let start = Instant::now();
            chunk.generate_mesh(gfx);
            total_time += start.elapsed().as_secs_f64();
            total_count += 1;
        }
        (total_time, total_count)
    }

    fn using_chunk_meshes<'a, F: FnMut(&[(Location, Option<&Mesh<VoxelVertex>>)]) + 'a>(
        &'a self,
        gfx: &'a Gfx,
        mut f: F,
    ) {
        let chunks_read = self.chunks.read().unwrap();
        let chunks_read_vec: Vec<_> = chunks_read.iter().map(|(chunk_position, chunk)| {
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
        }).collect();
        f(&chunks_read_vec)
    }
}

#[derive(Debug)]
pub struct Chunk {
    last_extended: Instant,
    voxels: Voxels,
    mesh: Option<Option<Mesh<VoxelVertex>>>,
}

impl Chunk {
    fn new(
        position: ChunkPosition,
        now: Instant,
        blueprint: &ChunkBlueprint,
        neighbor_blueprints: [&ChunkBlueprint; 6],
    ) -> Self {
        Self {
            last_extended: now,
            voxels: Voxels::new(
                position,
                blueprint.to_group(neighbor_blueprints),
                VoxelClass::Grass,
                VoxelClass::Water
            ), // TODO: make the voxel classes configurable
            mesh: None,
        }
    }

    fn extend_life(&mut self, now: Instant) {
        self.last_extended = now;
    }

    fn is_alive(&self) -> bool {
        self.last_extended.elapsed().as_secs() < CHUNK_LIFE_SECONDS
    }

    fn get_mesh(&self, gfx: &Gfx) -> Option<&Mesh<VoxelVertex>> {
        self.mesh.as_ref().and_then(|mesh| mesh.as_ref())
    }

    fn generate_mesh(&mut self, gfx: &Gfx) {
        if !self.voxels.needs_mesh() {
            return;
        }

        let mut proto_mesh = ProtoMesh::new(PrimitiveType::Points);

        let voxels = &self.voxels;
        let points = (VoxelUnits(0)..VOXELS_PER_CHUNK).flat_map(
            |z| (VoxelUnits(0)..VOXELS_PER_CHUNK).flat_map(
                move |y| {
                    (VoxelUnits(0)..VOXELS_PER_CHUNK).filter_map(
                        move |x| {
                            let voxel_position = vector!(x, y, z);
                            if let Some(voxel) = voxels.get(voxel_position) {
                                let mz = voxels.get(voxel_position - Vector::unit_z());
                                let pz = voxels.get(voxel_position + Vector::unit_z());
                                let my = voxels.get(voxel_position - Vector::unit_y());
                                let py = voxels.get(voxel_position + Vector::unit_y());
                                let mx = voxels.get(voxel_position - Vector::unit_x());
                                let px = voxels.get(voxel_position + Vector::unit_x());
                                if mz.is_none() || unsafe { mz.unwrap_unchecked() }.is_transparent()
                                    || pz.is_none() || unsafe { pz.unwrap_unchecked() }.is_transparent()
                                    || my.is_none() || unsafe { my.unwrap_unchecked() }.is_transparent()
                                    || py.is_none() || unsafe { py.unwrap_unchecked() }.is_transparent()
                                    || mx.is_none() || unsafe { mx.unwrap_unchecked() }.is_transparent()
                                    || px.is_none() || unsafe { px.unwrap_unchecked() }.is_transparent()
                                {
                                    let vertex_position = voxel_position.map(|c| c.0 as f32) + 0.5;
                                    let color = voxel.color();
                                    let normal = voxel.normal();
                                    Some(
                                        ProtoVertex::new(vertex_position)
                                            .with_color(color)
                                            .with_normal(normal)
                                    )
                                }
                                else {
                                    None
                                }
                            } else {
                                None
                            }
                        }
                    )
                },
            ),
        );
        
        proto_mesh.add_points(points);

        if proto_mesh.elements().is_empty() {
            self.mesh = Some(None);
        } else {
            self.mesh = Some(Some(gfx.create_mesh::<VoxelVertex>(&proto_mesh)));
        }

        self.voxels.mark_mesh_needed(false);
    }
}

#[derive(Debug)]
struct Voxels {
    needs_mesh: bool,
    voxels: Vec<Option<Voxel>>,
}

impl Voxels {
    fn new(
        chunk_position: ChunkPosition,
        blueprints: ChunkBlueprintGroup,
        terrain_filler: VoxelClass,
        liquid: VoxelClass,
    ) -> Self {
        let seed = chunk_position.x().into_i64()
            .wrapping_mul(chunk_position.y().into_i64())
            .wrapping_mul(chunk_position.z().into_i64()) as u64;
        let voxels = (0..VOXELS_PER_CHUNK.0.pow(3) as usize).map(|idx| {
            let idx_voxel_units = VoxelUnits(idx as i64);
            let z = idx_voxel_units % VOXELS_PER_CHUNK;
            let y = (idx_voxel_units / VOXELS_PER_CHUNK) % VOXELS_PER_CHUNK;
            let x = (idx_voxel_units / VOXELS_PER_CHUNK) / VOXELS_PER_CHUNK;
            let voxel_position_in_chunk = vector!(x, y, z);
            let voxel_blueprint = blueprints.get(voxel_position_in_chunk);
            match voxel_blueprint {
                VoxelBlueprint::Air => None,
                VoxelBlueprint::TerrainFiller => {
                    if blueprints.may_be_visible(voxel_position_in_chunk) {
                        let normal = blueprint_normal_using_neighbors(&blueprints, voxel_position_in_chunk, VoxelBlueprint::TerrainFiller);
                        Some(Voxel::new(terrain_filler, normal, seed.wrapping_mul((idx + 1) as u64)))
                    }
                    else {
                        None
                    }
                },
                VoxelBlueprint::Liquid => Some(Voxel::new(liquid, Vector::unit_z(), seed.wrapping_mul((idx + 1) as u64))),
                VoxelBlueprint::Other => None,
            }
        }).collect();
        Self {
            needs_mesh: true,
            voxels,
        }
    }

    fn position_to_index(position: VoxelPosition) -> Option<usize> {
        if position.x().0 < 0 || position.x().0 >= VOXELS_PER_CHUNK.0 {
            return None;
        }
        if position.y().0 < 0 || position.y().0 >= VOXELS_PER_CHUNK.0 {
            return None;
        }
        if position.z().0 < 0 || position.z().0 >= VOXELS_PER_CHUNK.0 {
            return None;
        }
        let index = (position.x().0 * VOXELS_PER_CHUNK.0 * VOXELS_PER_CHUNK.0)
            + (position.y().0 * VOXELS_PER_CHUNK.0)
            + position.z().0;
        Some(index as usize)
    }

    fn get(&self, position_in_chunk: VoxelPosition) -> Option<&Voxel> {
        Self::position_to_index(position_in_chunk)
            .and_then(|index| self.voxels[index].as_ref())
    }

    fn mark_mesh_needed(&mut self, value: bool) {
        self.needs_mesh = value;
    }

    fn needs_mesh(&self) -> bool {
        self.needs_mesh
    }
}

/// A distance or component of a location in chunk units.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkUnits(i64);

impl ChunkUnits {
    pub const fn into_i64(self) -> i64 {
        self.0
    }

    pub fn to_meters(self) -> f64 {
        chunks_to_meters(self.0 as f64)
    }

    pub fn from_meters(meters: f64) -> Self {
        ChunkUnits(meters_to_chunks(meters).floor() as i64)
    }
}

impl From<i64> for ChunkUnits {
    fn from(i: i64) -> Self {
        Self(i)
    }
}

impl Into<i64> for ChunkUnits {
    fn into(self) -> i64 {
        self.0
    }
}

impl From<u64> for ChunkUnits {
    fn from(i: u64) -> Self {
        Self(i.try_into().expect("ChunkUnits::from: u64 too large"))
    }
}

impl From<usize> for ChunkUnits {
    fn from(i: usize) -> Self {
        Self(i.try_into().expect("ChunkUnits::from: usize too large"))
    }
}

impl Into<usize> for ChunkUnits {
    fn into(self) -> usize {
        self.0
            .try_into()
            .expect("ChunkUnits::into: i64 does not fit into a usize")
    }
}

impl From<VoxelUnits> for ChunkUnits {
    fn from(voxel_units: VoxelUnits) -> Self {
        ((voxel_units.0 as f64 / VOXELS_PER_CHUNK.0 as f64).floor() as i64).into()
    }
}

impl Step for ChunkUnits {
    fn steps_between(start: &Self, end: &Self) -> Option<usize> {
        if start.0 < end.0 {
            Some(
                (end.0 - start.0)
                    .try_into()
                    .expect("ChunkUnits::steps_between: i64 does not fit into a usize"),
            )
        } else {
            None
        }
    }

    fn forward_checked(start: Self, count: usize) -> Option<Self> {
        Some(Self(
            start.0
                + i64::try_from(count)
                    .expect("ChunkUnits::forward_checked: usize does not fine into an i64"),
        ))
    }

    fn backward_checked(start: Self, count: usize) -> Option<Self> {
        Some(Self(
            start.0
                - i64::try_from(count)
                    .expect("ChunkUnits::backward_checked: usize does not fine into an i64"),
        ))
    }
}

impl Sum for ChunkUnits {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self(0), |acc, x| Self(acc.0 + x.0))
    }
}

impl_op_ex!(+ |a: ChunkUnits, b: ChunkUnits| -> ChunkUnits {
    ChunkUnits(a.0 + b.0)
});
impl_op_ex!(-|a: ChunkUnits, b: ChunkUnits| -> ChunkUnits { ChunkUnits(a.0 - b.0) });
impl_op_ex!(*|a: ChunkUnits, b: ChunkUnits| -> ChunkUnits { ChunkUnits(a.0 * b.0) });
impl_op_ex!(/ |a: ChunkUnits, b: ChunkUnits| -> ChunkUnits {
    ChunkUnits(a.0 / b.0)
});
impl_op_ex!(% |a: ChunkUnits, b: ChunkUnits| -> ChunkUnits {
    ChunkUnits(a.0 % b.0)
});
impl_op_ex!(+= |a: &mut ChunkUnits, b: ChunkUnits| {
    *a = *a + b;
});
impl_op_ex!(-= |a: &mut ChunkUnits, b: ChunkUnits| {
    *a = *a - b;
});
impl_op_ex!(*= |a: &mut ChunkUnits, b: ChunkUnits| {
    *a = *a * b;
});
impl_op_ex!(/= |a: &mut ChunkUnits, b: ChunkUnits| {
    *a = *a / b;
});
impl_op_ex!(%= |a: &mut ChunkUnits, b: ChunkUnits| {
    *a = *a % b;
});

impl Zero for ChunkUnits {
    fn zero() -> Self {
        Self(0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }

    fn set_zero(&mut self) {
        self.0 = 0;
    }
}

impl One for ChunkUnits {
    fn one() -> Self {
        Self(1)
    }
}

/// A position in chunk units.
pub type ChunkPosition = Vector3<ChunkUnits>;

pub trait ChunkPositionExt {
    fn to_voxel_position(&self) -> VoxelPosition;
    fn to_meters(&self) -> Vector3<f64>;
    fn from_meters(v: Vector3<f64>) -> Self;
}

impl ChunkPositionExt for ChunkPosition {
    fn to_voxel_position(&self) -> VoxelPosition {
        vector!(self.x().into(), self.y().into(), self.z().into())
    }

    fn to_meters(&self) -> Vector3<f64> {
        self.map(|x| x.to_meters())
    }

    fn from_meters(v: Vector3<f64>) -> Self {
        v.map(|x| ChunkUnits::from_meters(*x))
    }
}

/// A distance or component of a location in voxel units.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VoxelUnits(i64);

impl VoxelUnits {
    pub const fn into_i64(self) -> i64 {
        self.0
    }

    pub const fn into_f64(self) -> f64 {
        self.into_i64() as f64
    }

    pub fn to_meters(self) -> f64 {
        voxels_to_meters(self.into_f64())
    }

    pub fn from_meters(meters: f64) -> Self {
        Self(meters_to_voxels(meters).floor() as i64)
    }
}

impl From<i64> for VoxelUnits {
    fn from(i: i64) -> Self {
        Self(i)
    }
}

impl Into<i64> for VoxelUnits {
    fn into(self) -> i64 {
        self.0
    }
}

impl From<u64> for VoxelUnits {
    fn from(i: u64) -> Self {
        Self(i.try_into().expect("VoxelUnits::from: u64 too large"))
    }
}

impl From<usize> for VoxelUnits {
    fn from(i: usize) -> Self {
        Self(i.try_into().expect("VoxelUnits::from: usize too large"))
    }
}

impl Into<usize> for VoxelUnits {
    fn into(self) -> usize {
        self.0
            .try_into()
            .expect("ChunkUnits::into: i64 does not fit into a usize")
    }
}

impl From<ChunkUnits> for VoxelUnits {
    fn from(v: ChunkUnits) -> Self {
        (v.0 * VOXELS_PER_CHUNK.0).into()
    }
}

impl_op_ex!(+ |a: VoxelUnits, b: VoxelUnits| -> VoxelUnits {
    VoxelUnits(a.0 + b.0)
});
impl_op_ex!(-|a: VoxelUnits, b: VoxelUnits| -> VoxelUnits { VoxelUnits(a.0 - b.0) });
impl_op_ex!(*|a: VoxelUnits, b: VoxelUnits| -> VoxelUnits { VoxelUnits(a.0 * b.0) });
impl_op_ex!(/ |a: VoxelUnits, b: VoxelUnits| -> VoxelUnits {
    VoxelUnits(a.0 / b.0)
});
impl_op_ex!(% |a: VoxelUnits, b: VoxelUnits| -> VoxelUnits {
    VoxelUnits(a.0 % b.0)
});
impl_op_ex!(+= |a: &mut VoxelUnits, b: VoxelUnits| {
    *a = *a + b;
});
impl_op_ex!(-= |a: &mut VoxelUnits, b: VoxelUnits| {
    *a = *a - b;
});
impl_op_ex!(*= |a: &mut VoxelUnits, b: VoxelUnits| {
    *a = *a * b;
});
impl_op_ex!(/= |a: &mut VoxelUnits, b: VoxelUnits| {
    *a = *a / b;
});
impl_op_ex!(%= |a: &mut VoxelUnits, b: VoxelUnits| {
    *a = *a % b;
});

impl Zero for VoxelUnits {
    fn zero() -> Self {
        Self(0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }

    fn set_zero(&mut self) {
        self.0 = 0;
    }
}

impl One for VoxelUnits {
    fn one() -> Self {
        Self(1)
    }
}

impl Step for VoxelUnits {
    fn steps_between(start: &Self, end: &Self) -> Option<usize> {
        if start.0 < end.0 {
            Some(
                (end.0 - start.0)
                    .try_into()
                    .expect("VoxelUnits::steps_between: i64 does not fit into a usize"),
            )
        } else {
            None
        }
    }

    fn forward_checked(start: Self, count: usize) -> Option<Self> {
        Some(Self(
            start.0
                + i64::try_from(count)
                    .expect("VoxelUnits::forward_checked: usize does not fine into an i64"),
        ))
    }

    fn backward_checked(start: Self, count: usize) -> Option<Self> {
        Some(Self(
            start.0
                - i64::try_from(count)
                    .expect("VoxelUnits::backward_checked: usize does not fine into an i64"),
        ))
    }
}

/// A position in voxel units.
pub type VoxelPosition = Vector3<VoxelUnits>;

pub trait VoxelPositionExt {
    fn to_chunk_position(&self) -> (ChunkPosition, VoxelPosition);
    fn center_position(&self) -> Vector3<f64>;
    fn position_in_chunk(&self) -> VoxelPosition;
    fn to_meters(&self) -> Vector3<f64>;
    fn from_meters(v: Vector3<f64>) -> Self;
}

impl VoxelPositionExt for VoxelPosition {
    fn to_chunk_position(&self) -> (ChunkPosition, VoxelPosition) {
        let float_chunk = vector!(self.x().0 as f64, self.y().0 as f64, self.z().0 as f64)
            / VOXELS_PER_CHUNK.0 as f64;
        let floored_float_chunk = float_chunk.map(|f| f.floor());
        (
            floored_float_chunk.map(|c| (*c as i64).into()),
            ((float_chunk - floored_float_chunk) * VOXELS_PER_CHUNK.0 as f64).map(|c| (*c as i64).into()).into(),
        )
    }

    fn center_position(&self) -> Vector3<f64> {
        self.map(|v| v.0 as f64) + 0.5
    }

    fn position_in_chunk(&self) -> VoxelPosition {
        self.to_chunk_position().1
    }

    fn to_meters(&self) -> Vector3<f64> {
        self.map(|v| v.to_meters())
    }

    fn from_meters(v: Vector3<f64>) -> Self {
        v.map(|v| VoxelUnits::from_meters(*v))
    }
}

pub fn voxels_to_meters<T: Div<f64, Output = T>>(position: T) -> T {
    position / VOXELS_PER_METER.0 as f64
}

pub fn meters_to_voxels<T: Mul<f64, Output = T>>(position: T) -> T {
    position * VOXELS_PER_METER.0 as f64
}

pub fn chunks_to_voxels<T: Mul<f64, Output = T>>(position: T) -> T {
    position * VOXELS_PER_CHUNK.0 as f64
}

pub fn voxels_to_chunks<T: Div<f64, Output = T>>(position: T) -> T {
    position / VOXELS_PER_CHUNK.0 as f64
}

pub fn chunks_to_meters<T: Div<f64, Output = T> + Mul<f64, Output = T>>(position: T) -> T {
    voxels_to_meters(chunks_to_voxels(position))
}

pub fn meters_to_chunks<T: Div<f64, Output = T> + Mul<f64, Output = T>>(position: T) -> T {
    voxels_to_chunks(meters_to_voxels(position))
}

fn blueprint_normal_using_neighbors(blueprints: &ChunkBlueprintGroup, position_in_chunk: VoxelPosition, matching: VoxelBlueprint) -> Vector3<f32> {
    let normal =
        // -Z
        if blueprints.get(position_in_chunk - Vector::unit_z()) == matching {
            Vector::unit_z()
        }
        else {
            Vector::zero()
        }
        // +Z
        + if blueprints.get(position_in_chunk + Vector::unit_z()) == matching {
            -Vector::unit_z()
        }
        else {
            Vector::zero()
        }
        // -Y
        + if blueprints.get(position_in_chunk - Vector::unit_y()) == matching {
            Vector::unit_y()
        }
        else {
            Vector::zero()
        }
        // +Y
        + if blueprints.get(position_in_chunk + Vector::unit_y()) == matching {
            -Vector::unit_y()
        }
        else {
            Vector::zero()
        }
        // -X
        + if blueprints.get(position_in_chunk - Vector::unit_x()) == matching {
            Vector::unit_x()
        }
        else {
            Vector::zero()
        }
        // +X
        + if blueprints.get(position_in_chunk + Vector::unit_x()) == matching {
            -Vector::unit_x()
        }
        else {
            Vector::zero()
        }
        // -Z -Y -X
        + if blueprints.get(position_in_chunk - VoxelUnits(1)) == matching {
            Vector::one() * 0.7071
        }
        else {
            Vector::zero()
        }
        // -Z -Y +X
        + if blueprints.get(position_in_chunk - Vector::unit_z() - Vector::unit_y() + Vector::unit_x()) == matching {
            (Vector::unit_z() + Vector::unit_y() - Vector::unit_x()).normalized() * 0.7071
        }
        else {
            Vector::zero()
        }
        // -Z +Y -X
        + if blueprints.get(position_in_chunk - Vector::unit_z() + Vector::unit_y() - Vector::unit_x()) == matching {
            (Vector::unit_z() - Vector::unit_y() + Vector::unit_x()).normalized() * 0.7071
        }
        else {
            Vector::zero()
        }
        // -Z +Y +X
        + if blueprints.get(position_in_chunk - Vector::unit_z() + Vector::unit_y() + Vector::unit_x()) == matching {
            (Vector::unit_z() - Vector::unit_y() - Vector::unit_x()).normalized() * 0.7071
        }
        else {
            Vector::zero()
        }
        // +Z -Y -X
        + if blueprints.get(position_in_chunk + Vector::unit_z() - Vector::unit_y() - Vector::unit_x()) == matching {
            (-Vector::unit_z() + Vector::unit_y() + Vector::unit_x()).normalized() * 0.7071
        }
        else {
            Vector::zero()
        }
        // +Z -Y +X
        + if blueprints.get(position_in_chunk + Vector::unit_z() - Vector::unit_y() + Vector::unit_x()) == matching {
            (-Vector::unit_z() + Vector::unit_y() - Vector::unit_x()).normalized() * 0.7071
        }
        else {
            Vector::zero()
        }
        // +Z +Y -X
        + if blueprints.get(position_in_chunk + Vector::unit_z() + Vector::unit_y() - Vector::unit_x()) == matching {
            (-Vector::unit_z() - Vector::unit_y() + Vector::unit_x()).normalized() * 0.7071
        }
        else {
            Vector::zero()
        }
        // +Z +Y +X
        + if blueprints.get(position_in_chunk + VoxelUnits(1)) == matching {
            -Vector::one() * 0.7071
        }
        else {
            Vector::zero()
        };
    
    normal.normalized()
}