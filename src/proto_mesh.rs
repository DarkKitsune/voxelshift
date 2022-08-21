use ggmath::{
    prelude::{Matrix, Matrix3x3},
    quaternion::Quaternion,
    vector,
    vector_alias::{Vector2, Vector3},
};

use crate::{colors::Color, vertex::Vertex, game::PrimitiveType};

/// A prototype mesh
#[derive(Debug, Clone)]
pub struct ProtoMesh {
    primitive_type: PrimitiveType,
    vertices: Vec<ProtoVertex>,
    elements: Vec<u32>,
}

impl ProtoMesh {
    /// Create a new empty `ProtoMesh`
    pub fn new(primitive_type: PrimitiveType) -> ProtoMesh {
        ProtoMesh {
            primitive_type,
            vertices: Vec::new(),
            elements: Vec::new(),
        }
    }

    /// Concat a `ProtoMesh` onto this one
    pub fn concat(&mut self, other: &ProtoMesh) {
        if self.primitive_type() != other.primitive_type() {
            panic!("Cannot concat ProtoMeshs with different primitive types");
        }
        let offset = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        self.elements
            .extend(other.elements.iter().map(|&i| i + offset as u32));
    }

    /// Extend the `ProtoMesh` with the given vertices and elements
    pub fn extend(
        &mut self,
        vertices: impl IntoIterator<Item = ProtoVertex>,
        elements: impl IntoIterator<Item = u32>,
    ) {
        let old_length = self.elements().len();
        let offset = self.vertices.len() as u32;
        self.vertices.extend(vertices);
        self.elements
            .extend(elements.into_iter().map(|i| i + offset as u32));
        self.primitive_type().check_element_count(self.elements().len() - old_length);
    }

    /// Get the primitive type of this `ProtoMesh`
    pub fn primitive_type(&self) -> PrimitiveType {
        self.primitive_type
    }

    /// Gets the vertices of this `ProtoMesh`
    pub fn vertices(&self) -> &[ProtoVertex] {
        &self.vertices
    }

    /// Gets the elements of this `ProtoMesh`
    pub fn elements(&self) -> &[u32] {
        &self.elements
    }

    /// Gets the vertices of this `ProtoMesh` converted to the given type
    pub fn convert_vertices<'a, V: Vertex + 'a>(&'a self) -> impl Iterator<Item = V> + 'a {
        self.vertices.iter().map(V::from_proto_vertex)
    }

    /// Adds a rectangle to the `ProtoMesh`
    pub fn add_rectangle(
        &mut self,
        orientation: Orientation,
        size: Vector2<f32>,
        color: Color,
        tex_coords: (Vector2<f32>, Vector2<f32>),
    ) {
        if self.primitive_type() != PrimitiveType::Triangles {
            panic!("Can only add rectangles to triangle meshes");
        }
        
        let center = orientation.center();
        let half_size = size * 0.5;
        let axes = orientation
            .axes()
            .cloned()
            .unwrap_or_else(|| Matrix::identity());
        let x = axes.x_axis();
        let y = axes.y_axis();
        let forward = -axes.z_axis();
        let vertices = [
            // -x -y
            ProtoVertex {
                position: center - x * half_size.x() - y * half_size.y(),
                normal: Some(forward),
                color: Some(color),
                tex_coord: Some(tex_coords.0),
            },
            // -x +y
            ProtoVertex {
                position: center - x * half_size.x() + y * half_size.y(),
                normal: Some(forward),
                color: Some(color),
                tex_coord: Some(vector!(tex_coords.0.x(), tex_coords.1.y())),
            },
            // +x +y
            ProtoVertex {
                position: center + x * half_size.x() + y * half_size.y(),
                normal: Some(forward),
                color: Some(color),
                tex_coord: Some(vector!(tex_coords.1.x(), tex_coords.1.y())),
            },
            // +x -y
            ProtoVertex {
                position: center + x * half_size.x() - y * half_size.y(),
                normal: Some(forward),
                color: Some(color),
                tex_coord: Some(vector!(tex_coords.1.x(), tex_coords.0.y())),
            },
        ];
        let elements = [
            0, 1, 2, // -x -y, -x +y, +x +y
            2, 3, 0, // +x +y, +x -y, -x -y
        ];
        self.extend(vertices, elements);
    }

    pub fn add_box(&mut self, orientation: Orientation, size: Vector3<f32>, sides: &BoxSides) {
        if self.primitive_type() != PrimitiveType::Triangles {
            panic!("Can only add rectangles to triangle meshes");
        }

        let center = orientation.center();
        let half_size = size * 0.5;
        let axes = orientation
            .axes()
            .cloned()
            .unwrap_or_else(|| Matrix::identity());
        let x = axes.x_axis();
        let y = axes.y_axis();
        let z = axes.z_axis();

        // Rectangle on the -x side
        // Rectangle's X axis is the -Y axis of the box
        // Rectangle's Y axis is the -Z axis of the box
        // Rectangle's Z axis is the -X axis of the box
        if let Some(side) = sides.x.minus.as_ref() {
            self.add_rectangle(
                Orientation::new(
                    center - x * half_size.x(),
                    Some(Matrix::new([
                        [-y.x(), -y.y(), -y.z()],
                        [-z.x(), -z.y(), -z.z()],
                        [-x.x(), -x.y(), -x.z()],
                    ])),
                ),
                vector![size.y(), size.z()],
                side.color,
                side.tex_coords,
            );
        }

        // Rectangle on the +x side
        // Rectangle's X axis is the +Y axis of the box
        // Rectangle's Y axis is the -Z axis of the box
        // Rectangle's Z axis is the +X axis of the box
        if let Some(side) = sides.x.plus.as_ref() {
            self.add_rectangle(
                Orientation::new(
                    center + x * half_size.x(),
                    Some(Matrix::new([
                        [y.x(), y.y(), y.z()],
                        [-z.x(), -z.y(), -z.z()],
                        [x.x(), x.y(), x.z()],
                    ])),
                ),
                vector![size.y(), size.z()],
                side.color,
                side.tex_coords,
            );
        }

        // Rectangle on the -y side
        // Rectangle's X axis is the X axis of the box
        // Rectangle's Y axis is the -Z axis of the box
        // Rectangle's Z axis is the -Y axis of the box
        if let Some(side) = sides.y.minus.as_ref() {
            self.add_rectangle(
                Orientation::new(
                    center - y * half_size.y(),
                    Some(Matrix::new([
                        [x.x(), x.y(), x.z()],
                        [-z.x(), -z.y(), -z.z()],
                        [-y.x(), -y.y(), -y.z()],
                    ])),
                ),
                vector![size.x(), size.z()],
                side.color,
                side.tex_coords,
            );
        }

        // Rectangle on the +y side
        // Rectangle's X axis is the +X axis of the box
        // Rectangle's Y axis is the +Z axis of the box
        // Rectangle's Z axis is the -Y axis of the box
        if let Some(side) = sides.y.plus.as_ref() {
            self.add_rectangle(
                Orientation::new(
                    center + y * half_size.y(),
                    Some(Matrix::new([
                        [x.x(), x.y(), x.z()],
                        [z.x(), z.y(), z.z()],
                        [-y.x(), -y.y(), -y.z()],
                    ])),
                ),
                vector![size.x(), size.z()],
                side.color,
                side.tex_coords,
            );
        }

        // Rectangle on the -z side
        // Rectangle's X axis is the +X axis of the box
        // Rectangle's Y axis is the Y axis of the box
        // Rectangle's Z axis is the -Z axis of the box
        if let Some(side) = sides.z.minus.as_ref() {
            self.add_rectangle(
                Orientation::new(
                    center - z * half_size.z(),
                    Some(Matrix::new([
                        [x.x(), x.y(), x.z()],
                        [y.x(), y.y(), y.z()],
                        [-z.x(), -z.y(), -z.z()],
                    ])),
                ),
                vector![size.x(), size.y()],
                side.color,
                side.tex_coords,
            );
        }

        // Rectangle on the +z side
        // Rectangle's X axis is the +X axis of the box
        // Rectangle's Y axis is the -Y axis of the box
        // Rectangle's Z axis is the +Z axis of the box
        if let Some(side) = sides.z.plus.as_ref() {
            self.add_rectangle(
                Orientation::new(
                    center + z * half_size.z(),
                    Some(Matrix::new([
                        [x.x(), x.y(), x.z()],
                        [-y.x(), -y.y(), -y.z()],
                        [z.x(), z.y(), z.z()],
                    ])),
                ),
                vector![size.x(), size.y()],
                side.color,
                side.tex_coords,
            );
        }
    }

    pub fn add_points(&mut self, points: impl IntoIterator<Item = ProtoVertex>,) {
        if self.primitive_type() != PrimitiveType::Points {
            panic!("Can only add points to point meshes");
        }
        let element_start = self.vertices.len() as u32;
        self.vertices.extend(points);
        self.elements.extend(element_start..self.vertices.len() as u32);
    }
}

/// A prototype vertex for a `ProtoMesh`
#[derive(Debug, Clone)]
pub struct ProtoVertex {
    position: Vector3<f32>,
    normal: Option<Vector3<f32>>,
    color: Option<Color>,
    tex_coord: Option<Vector2<f32>>,
}

impl ProtoVertex {
    /// Create a new `ProtoVertex` with the given position
    pub fn new(position: Vector3<f32>) -> Self {
        Self {
            position,
            normal: None,
            color: None,
            tex_coord: None,
        }
    }

    pub fn with_normal(mut self, normal: Vector3<f32>) -> Self {
        self.normal = Some(normal);
        self
    }

    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }

    pub fn with_tex_coord(mut self, tex_coord: Vector2<f32>) -> Self {
        self.tex_coord = Some(tex_coord);
        self
    }

    pub fn set_position(&mut self, position: Vector3<f32>) {
        self.position = position;
    }

    pub fn set_normal(&mut self, normal: Vector3<f32>) {
        self.normal = Some(normal);
    }

    pub fn set_color(&mut self, color: Color) {
        self.color = Some(color);
    }

    pub fn set_tex_coord(&mut self, tex_coord: Vector2<f32>) {
        self.tex_coord = Some(tex_coord);
    }

    pub fn position(&self) -> Vector3<f32> {
        self.position
    }

    pub fn normal(&self) -> Option<Vector3<f32>> {
        self.normal
    }

    pub fn normal_or_panic(&self) -> Vector3<f32> {
        self.normal
            .unwrap_or_else(|| panic!("ProtoVertex has no normal"))
    }

    pub fn color(&self) -> Option<Color> {
        self.color
    }

    pub fn color_or_panic(&self) -> Color {
        self.color
            .unwrap_or_else(|| panic!("ProtoVertex has no color"))
    }

    pub fn tex_coord(&self) -> Option<Vector2<f32>> {
        self.tex_coord
    }

    pub fn tex_coord_or_panic(&self) -> Vector2<f32> {
        self.tex_coord
            .unwrap_or_else(|| panic!("ProtoVertex has no tex_coord"))
    }
}

pub struct Orientation {
    pub center: Vector3<f32>,
    axes: Option<Matrix3x3<f32>>,
}

impl Orientation {
    pub fn identity() -> Self {
        Self {
            center: vector![0.0, 0.0, 0.0],
            axes: None,
        }
    }

    pub fn new(center: Vector3<f32>, axes: Option<Matrix3x3<f32>>) -> Self {
        Self { center, axes }
    }

    pub fn from_position(position: Vector3<f32>) -> Self {
        Self::new(position, None)
    }

    pub fn from_position_rotation(position: Vector3<f32>, rotation: Quaternion<f32>) -> Self {
        Self {
            center: position,
            axes: Some(rotation.into()),
        }
    }

    pub fn center(&self) -> Vector3<f32> {
        self.center
    }

    pub fn axes(&self) -> Option<&Matrix3x3<f32>> {
        self.axes.as_ref()
    }
}

#[derive(Clone)]
pub struct BoxSides {
    pub x: BoxAxis,
    pub y: BoxAxis,
    pub z: BoxAxis,
}

impl BoxSides {
    pub fn new(x: BoxAxis, y: BoxAxis, z: BoxAxis) -> Self {
        Self { x, y, z }
    }

    pub fn new_uniform(params: &BoxSide) -> Self {
        Self::new(
            BoxAxis::new(Some(params.clone()), Some(params.clone())),
            BoxAxis::new(Some(params.clone()), Some(params.clone())),
            BoxAxis::new(Some(params.clone()), Some(params.clone())),
        )
    }
}

#[derive(Clone)]
pub struct BoxAxis {
    pub minus: Option<BoxSide>,
    pub plus: Option<BoxSide>,
}

impl BoxAxis {
    pub fn new(minus: Option<BoxSide>, plus: Option<BoxSide>) -> Self {
        Self { minus, plus }
    }
}

#[derive(Clone)]
pub struct BoxSide {
    pub color: Color,
    pub tex_coords: (Vector2<f32>, Vector2<f32>),
}
