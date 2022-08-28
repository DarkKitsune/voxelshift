use std::fmt::{Debug, Formatter};

use ggmath::{prelude::Vector4, vector};

#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
pub struct Color {
    vector: Vector4<f32>,
}

impl Color {
    pub const WHITE: Color = Color::new(1.0, 1.0, 1.0, 1.0);
    pub const BLACK: Color = Color::new(0.0, 0.0, 0.0, 1.0);
    pub const GRAY: Color = Color::new(0.5, 0.5, 0.5, 1.0);
    pub const DARK_GRAY: Color = Color::new(0.25, 0.25, 0.25, 1.0);
    pub const LIGHT_GRAY: Color = Color::new(0.75, 0.75, 0.75, 1.0);
    pub const RED: Color = Color::new(1.0, 0.0, 0.0, 1.0);
    pub const LIGHT_RED: Color = Color::new(1.0, 0.5, 0.5, 1.0);
    pub const DARK_RED: Color = Color::new(0.5, 0.0, 0.0, 1.0);
    pub const GREEN: Color = Color::new(0.0, 1.0, 0.0, 1.0);
    pub const LIGHT_GREEN: Color = Color::new(0.5, 1.0, 0.5, 1.0);
    pub const DARK_GREEN: Color = Color::new(0.0, 0.5, 0.0, 1.0);
    pub const BLUE: Color = Color::new(0.0, 0.0, 1.0, 1.0);
    pub const LIGHT_BLUE: Color = Color::new(0.5, 0.5, 1.0, 1.0);
    pub const DARK_BLUE: Color = Color::new(0.0, 0.0, 0.5, 1.0);
    pub const YELLOW: Color = Color::new(1.0, 1.0, 0.0, 1.0);
    pub const LIGHT_YELLOW: Color = Color::new(1.0, 1.0, 0.5, 1.0);
    pub const DARK_YELLOW: Color = Color::new(0.5, 0.5, 0.0, 1.0);
    pub const CYAN: Color = Color::new(0.0, 1.0, 1.0, 1.0);
    pub const LIGHT_CYAN: Color = Color::new(0.5, 1.0, 1.0, 1.0);
    pub const DARK_CYAN: Color = Color::new(0.0, 0.5, 0.5, 1.0);
    pub const MAGENTA: Color = Color::new(1.0, 0.0, 1.0, 1.0);
    pub const LIGHT_MAGENTA: Color = Color::new(1.0, 0.5, 1.0, 1.0);
    pub const DARK_MAGENTA: Color = Color::new(0.5, 0.0, 0.5, 1.0);
    pub const TRANSPARENT_BLACK: Color = Color::new(0.0, 0.0, 0.0, 0.0);
    pub const TRANSPARENT_WHITE: Color = Color::new(1.0, 1.0, 1.0, 0.0);

    /// Creates an RGBA color with the given values.
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Color {
            vector: vector!(r, g, b, a),
        }
    }

    /// Gets the red component of the color.
    pub const fn r(&self) -> f32 {
        self.vector.x()
    }

    /// Gets the green component of the color.
    pub const fn g(&self) -> f32 {
        self.vector.y()
    }

    /// Gets the blue component of the color.
    pub const fn b(&self) -> f32 {
        self.vector.z()
    }

    /// Gets the alpha component of the color.
    pub const fn a(&self) -> f32 {
        self.vector.w()
    }

    /// Makes a copy of the color with the given brightness.
    pub const fn with_brightness(self, brightness: f32) -> Self {
        Color {
            vector: vector!(
                self.vector.x() * brightness,
                self.vector.y() * brightness,
                self.vector.z() * brightness,
                self.vector.w()
            ),
        }
    }

    /// Makes a copy of the color with the given alpha.
    pub const fn with_alpha(self, transparency: f32) -> Self {
        Color {
            vector: vector!(
                self.vector.x(),
                self.vector.y(),
                self.vector.z(),
                transparency
            ),
        }
    }

    /// Makes a copy of the color with the given red component.
    pub const fn with_red(self, red: f32) -> Self {
        Color {
            vector: vector!(red, self.vector.y(), self.vector.z(), self.vector.w()),
        }
    }

    /// Makes a copy of the color with the given green component.
    pub const fn with_green(self, green: f32) -> Self {
        Color {
            vector: vector!(self.vector.x(), green, self.vector.z(), self.vector.w()),
        }
    }

    /// Makes a copy of the color with the given blue component.
    pub const fn with_blue(self, blue: f32) -> Self {
        Color {
            vector: vector!(self.vector.x(), self.vector.y(), blue, self.vector.w()),
        }
    }

    /// Makes a premultiplied copy of the color.
    pub const fn premultiplied(self) -> Self {
        Color {
            vector: vector!(
                self.vector.x() * self.vector.w(),
                self.vector.y() * self.vector.w(),
                self.vector.z() * self.vector.w(),
                self.vector.w()
            ),
        }
    }

    /// Makes an unpremultiplied copy of a premultiplied color.
    /// Note that this is a lossy operation, and if the alpha is 0 then the resulting color will be black.
    pub const fn unpremultiplied(self) -> Self {
        if self.vector.w() < 0.00001 {
            // Too low to divide by
            return Color::TRANSPARENT_BLACK;
        }
        Color {
            vector: vector!(
                self.vector.x() / self.vector.w(),
                self.vector.y() / self.vector.w(),
                self.vector.z() / self.vector.w(),
                self.vector.w()
            ),
        }
    }

    /// Convert the color to byte format.
    pub const fn to_bytes(self) -> [u8; 4] {
        [
            (self.vector.x() * 255.0) as u8,
            (self.vector.y() * 255.0) as u8,
            (self.vector.z() * 255.0) as u8,
            (self.vector.w() * 255.0) as u8,
        ]
    }

    /// Create a color from its byte representation.
    pub const fn from_bytes(bytes: [u8; 4]) -> Self {
        Color {
            vector: vector!(
                bytes[0] as f32 / 255.0,
                bytes[1] as f32 / 255.0,
                bytes[2] as f32 / 255.0,
                bytes[3] as f32 / 255.0,
            ),
        }
    }

    /// Break the color into its red, green, blue and alpha components.
    pub const fn to_components(self) -> [f32; 4] {
        [
            self.vector.x(),
            self.vector.y(),
            self.vector.z(),
            self.vector.w(),
        ]
    }

    /// Create a color from its red, green, blue and alpha components.
    pub const fn from_components(components: [f32; 4]) -> Self {
        Color::new(components[0], components[1], components[2], components[3])
    }

    /// Blend two colors
    pub const fn blended_with(self, other: Self, alpha: f32) -> Self {
        let inv_alpha = 1.0 - alpha;
        Color {
            vector: vector!(
                self.vector.x() * inv_alpha + other.vector.x() * alpha,
                self.vector.y() * inv_alpha + other.vector.y() * alpha,
                self.vector.z() * inv_alpha + other.vector.z() * alpha,
                self.vector.w() * inv_alpha + other.vector.w() * alpha,
            ),
        }
    }

    /// Gets whether the color is transparent (alpha less than 1.0).
    pub const fn is_transparent(&self) -> bool {
        self.a() < 1.0
    }
}

impl AsRef<Vector4<f32>> for Color {
    fn as_ref(&self) -> &Vector4<f32> {
        &self.vector
    }
}

impl AsMut<Vector4<f32>> for Color {
    fn as_mut(&mut self) -> &mut Vector4<f32> {
        &mut self.vector
    }
}

impl From<[f32; 4]> for Color {
    fn from(components: [f32; 4]) -> Self {
        Color::from_components(components)
    }
}

impl From<Vector4<f32>> for Color {
    fn from(vector: Vector4<f32>) -> Self {
        Color::new(vector.x(), vector.y(), vector.z(), vector.w())
    }
}

impl Into<[f32; 4]> for Color {
    fn into(self) -> [f32; 4] {
        self.to_components()
    }
}

impl From<[u8; 4]> for Color {
    fn from(bytes: [u8; 4]) -> Self {
        Color::from_bytes(bytes)
    }
}

impl Into<[u8; 4]> for Color {
    fn into(self) -> [u8; 4] {
        self.to_bytes()
    }
}

impl Into<Vector4<f32>> for Color {
    fn into(self) -> Vector4<f32> {
        self.vector
    }
}

impl Debug for Color {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "Color({:?}, {:?}, {:?}, {:?})",
            self.vector.x(),
            self.vector.y(),
            self.vector.z(),
            self.vector.w()
        )
    }
}

#[macro_export]
macro_rules! color {
    ($r:expr, $g:expr, $b:expr, $a:expr$(,)?) => {
        $crate::colors::Color::new($r, $g, $b, $a)
    };
    ($r:expr, $g:expr, $b:expr$(,)?) => {
        $crate::colors::Color::new($r, $g, $b, 1.0)
    };
}
