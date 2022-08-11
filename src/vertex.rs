use std::marker::PhantomData;

use ggmath::prelude::{Vector2, Vector3, Vector4};

use crate::{
    colors::Color,
    program_builder::expr::{self, Scalar},
    proto_mesh::ProtoVertex,
};

/// Trait for vertex structs. Don't implement this yourself unless you know what you're doing.
/// Use `vertex_struct!` instead.
pub trait Vertex: Copy {
    fn attributes() -> Vec<VertexAttribute>;
    fn from_proto_vertex(vertex: &ProtoVertex) -> Self;
}

/// Contains information about a vertex attribute.
#[derive(Debug)]
pub struct VertexAttribute {
    pub name: &'static str,
    pub expression_type: expr::ExpressionType,
    pub ty: gl::types::GLenum,
    pub size: usize,
    pub offset: usize,
    pub location: usize,
}

#[macro_export]
macro_rules! vertex_attributes {
    (
        $($name:ident: $ty:ty),*
        $(,)?
    ) => {
        {
            let mut offset = 0;
            let mut location = 0;
            vec![
                $(
                    #[allow(unused_assignments)]
                    {
                        let attr = $crate::vertex::VertexAttribute {
                            name: stringify!($name),
                            expression_type: $crate::vertex::__VertexAttributeTypes::<$ty>::EXPRESSION_TYPE,
                            ty: $crate::vertex::__VertexAttributeTypes::<$ty>::GL_TYPE,
                            size: $crate::vertex::__VertexAttributeTypes::<$ty>::SIZE,
                            offset,
                            location,
                        };
                        offset += $crate::vertex::__VertexAttributeTypes::<$ty>::BYTE_SIZE;
                        location += attr.expression_type.locations_consumed();
                        attr
                    },
                )*
            ]
        }
    };
}

#[macro_export]
macro_rules! vertex_struct {
    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident {
            $($field:ident: $ty:ty = $from_proto:expr),*
            $(,)?
        }
    ) => {
        $(#[$attr])*
        #[derive(Copy, Clone)]
        #[repr(C)]
        $vis struct $name {
            $(
                $field: $ty,
            )*
        }

        impl $crate::vertex::Vertex for $name {
            fn attributes() -> Vec<$crate::vertex::VertexAttribute> {
                vertex_attributes!(
                    $($field: $ty),*
                )
            }

            fn from_proto_vertex(vertex: &$crate::proto_mesh::ProtoVertex) -> Self {
                Self {
                    $(
                        $field: {
                            let from_proto_fn: fn(&$crate::proto_mesh::ProtoVertex) -> $ty = $from_proto;
                            from_proto_fn(vertex)
                        },
                    )*
                }
            }
        }
    };
}

/// Used for storing information about vertex attribute types.
pub struct __VertexAttributeTypes<T> {
    _phantom_data: PhantomData<T>,
}

impl __VertexAttributeTypes<f32> {
    pub const SIZE: usize = 1;
    pub const BYTE_SIZE: usize = 4;
    pub const GL_TYPE: gl::types::GLenum = gl::FLOAT;
    pub const EXPRESSION_TYPE: expr::ExpressionType = expr::ExpressionType::Scalar(Scalar::F32);
}

impl __VertexAttributeTypes<Vector2<f32>> {
    pub const SIZE: usize = 2;
    pub const BYTE_SIZE: usize = 8;
    pub const GL_TYPE: gl::types::GLenum = gl::FLOAT;
    pub const EXPRESSION_TYPE: expr::ExpressionType = expr::ExpressionType::Vector2(Scalar::F32);
}

impl __VertexAttributeTypes<Vector3<f32>> {
    pub const SIZE: usize = 3;
    pub const BYTE_SIZE: usize = 12;
    pub const GL_TYPE: gl::types::GLenum = gl::FLOAT;
    pub const NAME: &'static str = "vec3";
    pub const EXPRESSION_TYPE: expr::ExpressionType = expr::ExpressionType::Vector3(Scalar::F32);
}

impl __VertexAttributeTypes<Vector4<f32>> {
    pub const SIZE: usize = 4;
    pub const BYTE_SIZE: usize = 16;
    pub const GL_TYPE: gl::types::GLenum = gl::FLOAT;
    pub const NAME: &'static str = "vec4";
    pub const EXPRESSION_TYPE: expr::ExpressionType = expr::ExpressionType::Vector4(Scalar::F32);
}

vertex_struct! {
    /// Basic debug vertex with a position, normal, and color.
    #[derive(Debug)]
    pub struct DebugVertex {
        position: Vector3<f32> = |proto| proto.position(),
        normal: Vector3<f32> = |proto| proto.normal_or_panic(),
        color: Vector4<f32> = |proto| proto.color().unwrap_or(Color::WHITE).into(),
        tex_coord: Vector2<f32> = |proto| proto.tex_coord_or_panic(),
    }
}

impl DebugVertex {
    pub fn new(
        position: Vector3<f32>,
        normal: Vector3<f32>,
        color: Color,
        tex_coord: Vector2<f32>,
    ) -> Self {
        Self {
            position,
            normal,
            color: color.into(),
            tex_coord,
        }
    }
}
