use std::{
    collections::HashMap,
    convert::identity,
    iter::{Step, Sum}, ops::{Div, Mul},
};

use auto_ops::impl_op_ex;
use ggmath::init_array;
use num::{One, Zero};

use crate::{game::{voxel::Voxel, *}, vertex::WorldVertex};

pub const VOXELS_PER_CHUNK: VoxelUnits = VoxelUnits(25);
pub const VOXELS_PER_METER: VoxelUnits = VoxelUnits(3);
pub const WORLD_VIEW_DISTANCE_METERS: f64 = 50.0;
pub const WORLD_VIEW_DISTANCE_VERTICAL_RATIO: f64 = 0.25;
const CHUNK_LIFE_SECONDS: u64 = 9;
const EXTEND_IN_VIEW_IS_SPHERICAL: bool = true;

define_class! {
    /// The game world.
    pub class World {
        location: Location,
        world_chunks: Chunks,
        world_generator: DynWorldGenerator,
        world_program: Program,
    }
}

impl World {
    /// Create a new world.
    pub fn new(
        world_program: Program,
        world_generator: impl WorldGenerator + 'static,
    ) -> Self {
        // Create mesh to be put in mesh_renderer
        Self {
            location: Location::new(
                vector!(0.0, 0.0, 0.0),
                Quaternion::identity(),
                vector!(1.0, 1.0, 1.0),
            ),
            world_chunks: Chunks::new(),
            world_generator: DynWorldGenerator::new(world_generator),
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
            .extend_life_in_view(center, distance, WORLD_VIEW_DISTANCE_VERTICAL_RATIO, direction, &self.world_generator);
    }

    pub fn chunk_meshes<'a>(
        &'a mut self,
        gfx: &'a Gfx,
    ) -> impl Iterator<Item = (Location, Option<&'a Mesh<WorldVertex>>)> + 'a {
        self.world_chunks.meshes(gfx)
    }

    pub fn program(&self) -> &Program {
        &self.world_program
    }

    pub fn set_on_ground(&self, position: Vector3<f64>) -> Vector3<f64> {
        vector!(
            position.x(),
            voxels_to_meters(self.world_generator.elevation_at(meters_to_voxels(position).xz())),
            position.z(),
        )
    }
}

#[derive(Debug)]
pub struct Chunks {
    chunks: HashMap<ChunkPosition, Chunk>,
}

impl Chunks {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
        }
    }

    pub fn get(&self, chunk_position: ChunkPosition) -> Option<&Chunk> {
        self.chunks.get(&chunk_position)
    }

    pub fn get_mut(&mut self, chunk_position: ChunkPosition) -> Option<&mut Chunk> {
        self.chunks.get_mut(&chunk_position)
    }

    fn remove_old_chunks(&mut self) {
        self.chunks.retain(|_, chunk| chunk.is_alive());
    }

    fn extend_life(
        &mut self,
        chunk_position: ChunkPosition,
        now: Instant,
        world_generator: &DynWorldGenerator,
    ) {
        if let Some(chunk) = self.get_mut(chunk_position) {
            chunk.extend_life(now);
        } else {
            self.chunks.insert(
                chunk_position,
                Chunk::new(
                    chunk_position.clone(),
                    now,
                    world_generator.as_ref().new_constructor(),
                ),
            );
        }
    }

    /// Extend the lifetime of chunks in a given area. Units in meters.
    fn extend_life_in_view(
        &mut self,
        center: Vector3<f64>,
        distance: f64,
        vertical_ratio: f64,
        direction: Option<Vector3<f64>>,
        world_generator: &DynWorldGenerator,
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
                        self.extend_life(chunk_position, now, world_generator);
                    }
                }
            }
            else {
                for chunk_position in self.chunk_positions_in_sphere(center, distance, vertical_ratio) {
                    self.extend_life(chunk_position, now, world_generator);
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
                        self.extend_life(chunk_position, now, world_generator);
                    }
                }
            }
            else {
                for chunk_position in self.chunk_positions_in_box(
                    ChunkPosition::from_meters(center - distance * vertical_ratio),
                    ChunkPosition::from_meters(center + distance * vertical_ratio),
                ) {
                    self.extend_life(chunk_position, now, world_generator);
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

    fn chunks_in_box(
        &self,
        min: ChunkPosition,
        max: ChunkPosition,
    ) -> impl Iterator<Item = (ChunkPosition, Option<&Chunk>)> {
        self.chunk_positions_in_box(min, max)
            .map(|chunk_position| (chunk_position, self.get(chunk_position)))
    }

    /// Get all of the chunks found in a spherical area.
    /// Units are in meters.
    fn chunks_in_sphere(
        &self,
        center_chunk: Vector3<f64>,
        distance: f64,
        vertical_ratio: f64,
    ) -> impl Iterator<Item = (ChunkPosition, Option<&Chunk>)> {
        self.chunk_positions_in_sphere(center_chunk, distance, vertical_ratio)
            .map(|chunk_position| (chunk_position, self.get(chunk_position)))
    }

    fn meshes<'a>(
        &'a mut self,
        gfx: &'a Gfx,
    ) -> impl Iterator<Item = (Location, Option<&'a Mesh<WorldVertex>>)> + 'a {
        self.chunks.iter_mut().map(|(chunk_position, chunk)| {
            let chunk_meter_position = chunk_position.to_meters();
            let voxel_to_meter = voxels_to_meters(1.0);
            (
                Location::new(
                    chunk_meter_position,
                    Quaternion::identity(),
                    Vector::from_scalar(voxel_to_meter),
                ),
                chunk.mesh(gfx),
            )
        })
    }
}

#[derive(Debug)]
pub struct Chunk {
    last_extended: Instant,
    voxels: Voxels,
    mesh: Option<Mesh<WorldVertex>>,
}

impl Chunk {
    fn new(
        position: ChunkPosition,
        now: Instant,
        constructor: impl FnMut(VoxelPosition) -> Option<Voxel>,
    ) -> Self {
        Self {
            last_extended: now,
            voxels: Voxels::new(position, constructor),
            mesh: None,
        }
    }

    fn extend_life(&mut self, now: Instant) {
        self.last_extended = now;
    }

    fn is_alive(&self) -> bool {
        self.last_extended.elapsed().as_secs() < CHUNK_LIFE_SECONDS
    }

    fn mesh(&mut self, gfx: &Gfx) -> Option<&Mesh<WorldVertex>> {
        if self.mesh.is_none() {
            self.generate_mesh(gfx);
        }
        self.mesh.as_ref()
    }

    fn generate_mesh(&mut self, gfx: &Gfx) {
        let mut proto_mesh = ProtoMesh::new(PrimitiveType::Points);

        let voxels = &self.voxels;
        let points = (VoxelUnits(0)..VOXELS_PER_CHUNK).flat_map(
            |z| (VoxelUnits(0)..VOXELS_PER_CHUNK).flat_map(
                move |y| {
                    (VoxelUnits(0)..VOXELS_PER_CHUNK).filter_map(
                        move |x| {
                            let voxel_position = vector!(x, y, z);
                            if let Some(voxel) = voxels.get(voxel_position) {
                                if voxels.get(voxel_position - Vector::unit_z()).is_none()
                                    || voxels.get(voxel_position + Vector::unit_z()).is_none()
                                    || voxels.get(voxel_position - Vector::unit_y()).is_none()
                                    || voxels.get(voxel_position + Vector::unit_y()).is_none()
                                    || voxels.get(voxel_position - Vector::unit_x()).is_none()
                                    || voxels.get(voxel_position + Vector::unit_x()).is_none()
                                {
                                    let vertex_position = voxel_position.map(|c| c.0 as f32) + 0.5;
                                    let color = voxel.color();
                                    Some(
                                        ProtoVertex::new(vertex_position)
                                            .with_color(color)
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
            self.mesh = None;
        } else {
            self.mesh = Some(gfx.create_mesh::<WorldVertex>(&proto_mesh));
        }
    }
}

#[derive(Debug)]
struct Voxels {
    voxels: Vec<Option<Voxel>>,
}

impl Voxels {
    fn new(
        chunk_position: ChunkPosition,
        mut constructor: impl FnMut(VoxelPosition) -> Option<Voxel>,
    ) -> Self {
        let chunk_position_in_voxels = chunk_position.to_voxel_position();
        let mut voxels = Vec::new();
        for x in VoxelUnits(0)..VOXELS_PER_CHUNK {
            for y in VoxelUnits(0)..VOXELS_PER_CHUNK {
                for z in VoxelUnits(0)..VOXELS_PER_CHUNK {
                    voxels.push(constructor(chunk_position_in_voxels + vector!(x, y, z)));
                }
            }
        }
        Self { voxels }
    }

    fn position_to_index(&self, position: VoxelPosition) -> Option<usize> {
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
        self.position_to_index(position_in_chunk)
            .and_then(|index| self.voxels[index].as_ref())
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

/// Impl for world generator types.
pub trait WorldGenerator {
    /// Create a voxel constructor function.
    fn new_constructor(&self) -> Box<dyn FnMut(VoxelPosition) -> Option<Voxel>>;
    fn elevation_at(&self, xz: Vector2<f64>) -> f64;
}

pub struct DynWorldGenerator {
    world_generator: Box<dyn WorldGenerator>,
}

impl DynWorldGenerator {
    pub fn new(world_generator: impl WorldGenerator + 'static) -> Self {
        Self {
            world_generator: Box::new(world_generator),
        }
    }

    pub fn as_ref(&self) -> &dyn WorldGenerator {
        self.world_generator.as_ref()
    }

    /// Sample the elevation at the given X and Z position.
    pub fn elevation_at(&self, xz: Vector2<f64>) -> f64 {
        self.world_generator.elevation_at(xz)
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