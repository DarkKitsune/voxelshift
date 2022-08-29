use std::{iter::{Step, Sum}, ops::{Div, Mul}};

use auto_ops::impl_op_ex;
use num::{One, Zero};

use crate::game::*;

use super::{voxel::{VoxelVertex, VoxelClass, Voxel}, chunk_blueprint::{ChunkBlueprint, CHUNK_LIFE_SECONDS, ChunkBlueprintGroup, VoxelBlueprint}, world::{VOXELS_PER_CHUNK, VOXELS_PER_METER}};

#[derive(Debug)]
pub struct Chunk {
    last_extended: Instant,
    voxels: Voxels,
    mesh: Option<Option<Mesh<VoxelVertex>>>,
}

impl Chunk {
    pub fn new(
        position: ChunkPosition,
        now: Instant,
        blueprint: &ChunkBlueprint,
        neighbor_blueprints: [&ChunkBlueprint; 6],
    ) -> (Self, f64) {
        let start = Instant::now();
        let voxels = Voxels::new(
            position,
            blueprint.to_group(neighbor_blueprints),
            VoxelClass::Grass,
            VoxelClass::Water,
        );
        (
            Self {
                last_extended: now,
                voxels, // TODO: make the voxel classes configurable
                mesh: None,
            },
            start.elapsed().as_secs_f64(),
        )
    }

    pub fn extend_life(&mut self, now: Instant) {
        self.last_extended = now;
    }

    pub fn is_alive(&self) -> bool {
        self.last_extended.elapsed().as_secs() < CHUNK_LIFE_SECONDS
    }

    pub fn get_mesh(&self, gfx: &Gfx) -> Option<&Mesh<VoxelVertex>> {
        self.mesh.as_ref().and_then(|mesh| mesh.as_ref())
    }

    pub fn generate_mesh(&mut self, gfx: &Gfx) {
        if !self.voxels.needs_mesh() {
            return;
        }

        let mut proto_mesh = ProtoMesh::new(PrimitiveType::Points);

        let voxels = &self.voxels;
        let points = (VoxelUnits(0)..VOXELS_PER_CHUNK).flat_map(|z| {
            (VoxelUnits(0)..VOXELS_PER_CHUNK).flat_map(move |y| {
                (VoxelUnits(0)..VOXELS_PER_CHUNK).filter_map(move |x| {
                    let voxel_position = vector!(x, y, z);
                    if let Some(voxel) = voxels.get(voxel_position) {
                        let mz = voxels.get(voxel_position - Vector::unit_z());
                        let pz = voxels.get(voxel_position + Vector::unit_z());
                        let my = voxels.get(voxel_position - Vector::unit_y());
                        let py = voxels.get(voxel_position + Vector::unit_y());
                        let mx = voxels.get(voxel_position - Vector::unit_x());
                        let px = voxels.get(voxel_position + Vector::unit_x());
                        if mz.is_none()
                            || unsafe { mz.unwrap_unchecked() }.is_transparent()
                            || pz.is_none()
                            || unsafe { pz.unwrap_unchecked() }.is_transparent()
                            || my.is_none()
                            || unsafe { my.unwrap_unchecked() }.is_transparent()
                            || py.is_none()
                            || unsafe { py.unwrap_unchecked() }.is_transparent()
                            || mx.is_none()
                            || unsafe { mx.unwrap_unchecked() }.is_transparent()
                            || px.is_none()
                            || unsafe { px.unwrap_unchecked() }.is_transparent()
                        {
                            let vertex_position = voxel_position.map(|c| c.0 as f32) + 0.5;
                            let color = voxel.color();
                            let normal = voxel.normal();
                            Some(
                                ProtoVertex::new(vertex_position)
                                    .with_color(color)
                                    .with_normal(normal),
                            )
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
            })
        });

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
        let seed = chunk_position
            .x()
            .into_i64()
            .wrapping_mul(chunk_position.y().into_i64())
            .wrapping_mul(chunk_position.z().into_i64()) as u64;
        let voxels = (0..VOXELS_PER_CHUNK.0.pow(3) as usize)
            .map(|idx| {
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
                            let normal = blueprint_normal_using_neighbors(
                                &blueprints,
                                voxel_position_in_chunk,
                                VoxelBlueprint::TerrainFiller,
                            );
                            Some(Voxel::new(
                                terrain_filler,
                                normal,
                                seed.wrapping_mul((idx + 1) as u64),
                            ))
                        } else {
                            None
                        }
                    }
                    VoxelBlueprint::Liquid => Some(Voxel::new(
                        liquid,
                        Vector::unit_z(),
                        seed.wrapping_mul((idx + 1) as u64),
                    )),
                    VoxelBlueprint::Other => None,
                }
            })
            .collect();
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
        Self::position_to_index(position_in_chunk).and_then(|index| self.voxels[index].as_ref())
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

    pub const fn from_i64(i: i64) -> Self {
        Self(i)
    }

    pub const fn from_u64(i: u64) -> Self {
        Self(i as i64)
    }

    pub const fn from_f64(f: f64) -> Self {
        Self(f as i64)
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
            ((float_chunk - floored_float_chunk) * VOXELS_PER_CHUNK.0 as f64)
                .map(|c| (*c as i64).into())
                .into(),
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

fn blueprint_normal_using_neighbors(
    blueprints: &ChunkBlueprintGroup,
    position_in_chunk: VoxelPosition,
    matching: VoxelBlueprint,
) -> Vector3<f32> {
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
