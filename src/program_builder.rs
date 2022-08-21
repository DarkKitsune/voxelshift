use std::{collections::HashMap, marker::PhantomData, sync::Arc};

use crate::{
    gfx::{Gfx, Program, ShaderType},
    vertex::Vertex,
};

/// A builder for building a shader program.
pub struct ProgramBuilder<V: Vertex> {
    vertex_module: Option<VertexModule<V>>,
    fragment_module: Option<FragmentModule>,
    uniforms: Option<ModuleUniforms>,
}

impl<V: Vertex> ProgramBuilder<V> {
    /// Creates a new program builder.
    pub fn new() -> Self {
        Self {
            vertex_module: None,
            fragment_module: None,
            uniforms: None,
        }
    }

    /// Adds uniform variables to the program.
    pub fn with_uniforms(
        mut self,
        uniforms: impl IntoIterator<Item = (String, expr::ExpressionType)>,
    ) -> Self {
        if self.uniforms.is_none() {
            self.uniforms = Some(ModuleUniforms::new());
        }
        self.uniforms.as_mut().unwrap().add(uniforms);
        self
    }

    /// Adds a camera to the program.
    pub fn with_camera(self) -> Self {
        self.with_uniforms([
            (
                "_camera_position".to_string(),
                expr::ExpressionType::Vector3(expr::Scalar::F32),
            ),
            (
                "_camera_direction".to_string(),
                expr::ExpressionType::Vector3(expr::Scalar::F32),
            ),
            (
                "_camera_up".to_string(),
                expr::ExpressionType::Vector3(expr::Scalar::F32),
            ),
            (
                "_camera_projection".to_string(),
                expr::ExpressionType::Matrix4x4(expr::Scalar::F32),
            ),
        ])
    }

    /// States that the program has no uniform variables.
    pub fn with_no_uniforms(mut self) -> Self {
        self.uniforms = Some(ModuleUniforms::new());
        self
    }

    /// Sets the vertex module.
    pub fn with_vertex_module<F: FnOnce(&ModuleInputs, &mut ModuleOutputs, &mut ModuleUniforms)>(
        mut self,
        main_function: F,
    ) -> Self {
        let mut uniforms = self.uniforms.take().expect("Uniforms must be set before attaching any modules (ProgramBuilder::with_uniforms or ProgramBuilder::with_no_uniforms)");
        self.vertex_module = Some(
            VertexModuleBuilder::<V>::new()
                .with_main_function(&mut uniforms, main_function)
                .build(),
        );
        self.uniforms.replace(uniforms);
        self
    }

    /// Sets the fragment module.
    pub fn with_fragment_module<
        F: FnOnce(&ModuleInputs, &mut ModuleOutputs, &mut ModuleUniforms),
    >(
        mut self,
        main_function: F,
    ) -> Self {
        let mut uniforms = self.uniforms.take().expect("Uniforms must be set before attaching any modules (ProgramBuilder::with_uniforms or ProgramBuilder::with_no_uniforms)");
        self.fragment_module = Some(
            FragmentModuleBuilder::new::<V>(
                &self
                    .vertex_module
                    .as_ref()
                    .expect("Vertex module must be attached before fragment module")
                    .outputs,
            )
            .with_main_function(&mut uniforms, main_function)
            .build(),
        );
        self.uniforms.replace(uniforms);
        self
    }

    /// Builds the program.
    pub fn build(self, gfx: &Gfx) -> Program {
        if let Some(uniforms) = self.uniforms {
            // Check that all the defined uniforms are actually used (only effective in debug mode as it's non-essential).
            #[cfg(debug_assertions)]
            {
                let unused = uniforms.all_unused();
                if unused.len() > 0 {
                    panic!("Unused uniforms: {:?}", unused);
                }
            }
            // Create shader modules
            let vertex_shader = gfx.create_shader(
                ShaderType::Vertex,
                self.vertex_module
                    .as_ref()
                    .expect(
                        "Vertex module must be attached with ProgramBuilder::with_vertex_module",
                    )
                    .source
                    .as_str(),
            );
            let fragment_shader = gfx.create_shader(
                ShaderType::Fragment,
                self.fragment_module.as_ref().expect("Fragment module must be attached with ProgramBuilder::with_fragment_module").source.as_str(),
            );
            // Create program
            gfx.__create_program([&vertex_shader, &fragment_shader], uniforms.unwrap())
        } else {
            panic!("Uniforms must be set before building the program (ProgramBuilder::with_uniforms or ProgramBuilder::with_no_uniforms)");
        }
    }

    pub fn vertex_code(&self) -> &str {
        self.vertex_module
            .as_ref()
            .expect("Vertex module must be attached")
            .source
            .as_str()
    }

    pub fn fragment_code(&self) -> &str {
        self.fragment_module
            .as_ref()
            .expect("Fragment module must be attached")
            .source
            .as_str()
    }
}

/// The vertex module part of a shader program.
#[derive(Debug)]
pub struct VertexModule<V: Vertex> {
    source: String,
    outputs: ModuleOutputs,
    phantom: PhantomData<V>,
}

impl<V: Vertex> VertexModule<V> {
    /// Gets the source code of the vertex module
    pub fn source(&self) -> &str {
        &self.source
    }
}

/// A builder for building the vertex module for a shader program.
pub struct VertexModuleBuilder<V: Vertex> {
    outputs: Option<ModuleOutputs>,
    _phantom_data: PhantomData<V>,
}

impl<V: Vertex> VertexModuleBuilder<V> {
    fn new() -> Self {
        Self {
            outputs: None,
            _phantom_data: PhantomData,
        }
    }

    fn build(self) -> VertexModule<V> {
        // Write version string
        let source = "#version 450\n".to_string();
        // Write input definitions based on V's attributes
        let source = V::attributes().iter().fold(source, |mut acc, attr| {
            acc.push_str("layout (location = ");
            acc.push_str(&attr.location.to_string());
            acc.push_str(") in ");
            acc.push_str(attr.expression_type.to_str());
            acc.push_str(" in_");
            acc.push_str(attr.name);
            acc.push_str(";\n");
            acc
        });
        // Write output definitions
        let outputs = self
            .outputs
            .as_ref()
            .expect("Vertex module must have its outputs set");
        let source = outputs.iter().fold(source, |mut acc, output| {
            if output.next_stage_input.name != "gl_Position" && output.next_stage_input.name != "gl_PointSize" {
                acc.push_str("layout (location = ");
                acc.push_str(&output.next_stage_input.location.to_string());
                acc.push_str(") out ");
                acc.push_str(output.next_stage_input.expression_type.to_str());
                acc.push_str(" out_");
                acc.push_str(output.next_stage_input.name);
                acc.push_str(";\n");
            }
            acc
        });
        // Write uniforms
        let mut uniforms =
            self.outputs
                .as_ref()
                .unwrap()
                .iter()
                .fold(Vec::new(), |mut acc, output| {
                    acc.extend(output.used_uniforms());
                    acc
                });
        uniforms.sort_by(|a, b| a.0.cmp(&b.0));
        uniforms.dedup_by(|a, b| a.0 == b.0);
        let source = uniforms.into_iter().fold(source, |mut acc, uniform| {
            acc.push_str("uniform ");
            acc.push_str(uniform.1.to_str());
            acc.push_str(" ");
            acc.push_str(uniform.0);
            acc.push_str(";\n");
            acc
        });
        // Write main function
        let source = write_main(source, outputs, Module::Vertex);
        // Return built module
        drop(outputs);
        VertexModule {
            outputs: self.outputs.unwrap(),
            source,
            phantom: PhantomData,
        }
    }

    fn with_main_function<F: FnOnce(&ModuleInputs, &mut ModuleOutputs, &mut ModuleUniforms)>(
        mut self,
        uniforms: &mut ModuleUniforms,
        f: F,
    ) -> Self {
        let mut outputs = ModuleOutputs::new(Module::Vertex);
        f(&ModuleInputs::from_vertex::<V>(), &mut outputs, uniforms);
        self.outputs = Some(outputs);
        self
    }
}

/// The fragment module part of a shader program.
#[derive(Debug)]
pub struct FragmentModule {
    source: String,
}

impl FragmentModule {
    /// Gets the source code of the fragment module
    pub fn source(&self) -> &str {
        &self.source
    }
}

/// A builder for building the fragment module for a shader program.
struct FragmentModuleBuilder {
    inputs_from_vertex: ModuleInputs,
    outputs: Option<ModuleOutputs>,
}

impl FragmentModuleBuilder {
    fn new<V: Vertex>(previous_outputs: &ModuleOutputs) -> Self {
        Self {
            inputs_from_vertex: previous_outputs.to_inputs(Module::Fragment),
            outputs: None,
        }
    }

    fn build(self) -> FragmentModule {
        // Write version string
        let source = "#version 450\n".to_string();
        // Write input definitions based on inputs_from_vertex
        let source = self
            .inputs_from_vertex
            .iter()
            .fold(source, |mut acc, input| {
                if input.name != "gl_Position" && input.name != "gl_PointSize" {
                    acc.push_str("layout (location = ");
                    acc.push_str(&input.location.to_string());
                    acc.push_str(") in ");
                    acc.push_str(input.expression_type.to_str());
                    acc.push_str(" in_");
                    acc.push_str(input.name);
                    acc.push_str(";\n");
                }
                acc
            });
        // Write output definitions
        let source = self
            .outputs
            .as_ref()
            .expect("Fragment module must have its outputs set")
            .iter()
            .fold(source, |mut acc, output| {
                acc.push_str("layout (location = ");
                acc.push_str(&output.next_stage_input.location.to_string());
                acc.push_str(") out ");
                acc.push_str(output.next_stage_input.expression_type.to_str());
                if output.next_stage_input.name != "out_frag_color" {
                    acc.push_str(" out_");
                } else {
                    acc.push_str(" ");
                }
                acc.push_str(output.next_stage_input.name);
                acc.push_str(";\n");
                acc
            });
        // Write uniforms
        let mut uniforms =
            self.outputs
                .as_ref()
                .unwrap()
                .iter()
                .fold(Vec::new(), |mut acc, output| {
                    acc.extend(output.used_uniforms());
                    acc
                });
        uniforms.sort_by(|a, b| a.0.cmp(&b.0));
        uniforms.dedup_by(|a, b| a.0 == b.0);
        let source = uniforms.into_iter().fold(source, |mut acc, uniform| {
            acc.push_str("uniform ");
            acc.push_str(uniform.1.to_str());
            acc.push_str(" ");
            acc.push_str(uniform.0);
            acc.push_str(";\n");
            acc
        });
        // Write main function
        let source = write_main(
            source,
            self.outputs
                .as_ref()
                .expect("Fragment module must have its outputs set"),
            Module::Fragment,
        );
        // Return built module
        FragmentModule { source }
    }

    fn with_main_function<F: FnOnce(&ModuleInputs, &mut ModuleOutputs, &mut ModuleUniforms)>(
        mut self,
        uniforms: &mut ModuleUniforms,
        f: F,
    ) -> Self {
        let mut outputs = ModuleOutputs::new(Module::Fragment);
        f(&self.inputs_from_vertex, &mut outputs, uniforms);
        self.outputs = Some(outputs);
        self
    }
}

#[derive(Clone, Debug)]
pub struct ModuleInput {
    location: usize,
    expression_type: expr::ExpressionType,
    name: &'static str,
}

pub struct ModuleInputs {
    inputs: HashMap<String, ModuleInput>,
}

impl ModuleInputs {
    fn from_vertex<V: Vertex>() -> Self {
        let inputs = V::attributes()
            .into_iter()
            .map(|attr| {
                (
                    attr.name.to_string(),
                    ModuleInput {
                        location: attr.location,
                        expression_type: attr.expression_type,
                        name: attr.name,
                    },
                )
            })
            .collect();

        Self { inputs }
    }

    fn iter(&self) -> impl Iterator<Item = &ModuleInput> {
        self.inputs.values()
    }

    pub fn get(&self, name: &'static str) -> Option<expr::Expression> {
        self.inputs
            .get(name)
            .map(|input| expr::Expression::__input(input.name, input.expression_type))
    }
}

#[derive(Clone, Debug)]
pub struct ModuleOutput {
    receiver: Module,
    next_stage_input: ModuleInput,
    value: Arc<expr::Expression>,
}

impl ModuleOutput {
    fn to_input(&self) -> ModuleInput {
        self.next_stage_input.clone()
    }

    fn used_uniforms<'a>(&'a self) -> impl Iterator<Item = (&'a str, expr::ExpressionType)> {
        self.value.used_uniforms()
    }
}

/// The outputs for a module
#[derive(Debug)]
pub struct ModuleOutputs {
    module: Module,
    outputs: HashMap<String, ModuleOutput>,
    next_location: usize,
}

impl ModuleOutputs {
    fn new(module: Module) -> Self {
        Self {
            module,
            outputs: HashMap::new(),
            next_location: 0,
        }
    }

    pub fn to_inputs(&self, stage: Module) -> ModuleInputs {
        let inputs = self
            .outputs
            .iter()
            .filter_map(|(name, output)| {
                if output.receiver == stage {
                    Some((name.clone(), output.to_input()))
                } else {
                    None
                }
            })
            .collect();
        ModuleInputs { inputs }
    }

    pub fn set(
        &mut self,
        receiver: Module,
        name: &'static str,
        value: impl Into<expr::Expression>,
    ) {
        // Verify that the receiver is valid
        if !receiver.comes_after(self.module) && name != "out_frag_color" {
            panic!("Receiver of output {:?} set by the {:?} module must come later in the graphics pipeline", name, self.module);
        }

        // Convert the value into an expression
        let value = value.into();
        let value_type = value.expression_type();

        // Build an output
        let output = ModuleOutput {
            receiver,
            next_stage_input: ModuleInput {
                location: if name == "gl_Position" {
                    0
                } else if name == "gl_PointSize" {
                    1
                } else {
                    self.next_location
                },
                expression_type: value_type,
                name,
            },
            value: Arc::new(value),
        };

        // Don't increment location if it's a built-in output
        if name != "gl_Position" && name != "gl_PointSize" {
            self.next_location += value_type.locations_consumed();
        }

        // Add the output to the map
        self.outputs.insert(name.to_string(), output);
    }

    pub fn set_vertex(&mut self, position: impl Into<expr::Expression>, point_size: impl Into<expr::Expression>) {
        self.set(Module::Fragment, "gl_Position", position);
        self.set(Module::Fragment, "gl_PointSize", point_size);
    }

    pub fn set_fragment_color(&mut self, value: impl Into<expr::Expression>) {
        self.set(Module::Fragment, "out_frag_color", value);
    }

    pub fn get(&self, name: &'static str) -> Option<&ModuleOutput> {
        self.outputs.get(name)
    }

    fn iter(&self) -> impl Iterator<Item = &ModuleOutput> {
        self.outputs.values()
    }

    fn verify(&self, stage: Module) {
        for (name, output) in self.outputs.iter() {
            // Verify expression
            output.value.validate_operands();
            // Verify that built-in outputs are valid
            match stage {
                Module::Vertex => {
                    if name == "gl_Position" {
                        if self.module != Module::Vertex {
                            panic!("Vertex position output can only be set by the vertex module");
                        }
                        if output.value.expression_type()
                            != expr::ExpressionType::Vector4(expr::Scalar::F32)
                        {
                            panic!("Vertex position output must be a vec4");
                        }
                    }
                    else if name == "gl_PointSize" {
                        if self.module != Module::Vertex {
                            panic!("Vertex point size output can only be set by the vertex module");
                        }
                        if output.value.expression_type()
                            != expr::ExpressionType::Scalar(expr::Scalar::F32)
                        {
                            panic!("Vertex position output must be a float");
                        }
                    }
                }
                Module::Fragment => {
                    if name == "out_frag_color" {
                        if self.module != Module::Fragment {
                            panic!("Fragment color output can only be set by the fragment module");
                        }
                        if output.value.expression_type()
                            != expr::ExpressionType::Vector4(expr::Scalar::F32)
                        {
                            panic!("Fragment color output must be a vec4");
                        }
                    }
                }
            }
        }
    }
}

pub struct ModuleUniforms {
    uniforms: HashMap<String, (expr::ExpressionType, bool)>,
}

impl ModuleUniforms {
    /// Create a new empty set of module uniforms
    fn new() -> Self {
        Self {
            uniforms: HashMap::new(),
        }
    }

    /// Add uniforms to the set
    fn add(&mut self, uniforms: impl IntoIterator<Item = (String, expr::ExpressionType)>) {
        self.uniforms.extend(
            uniforms
                .into_iter()
                .map(|(name, expression_type)| (name, (expression_type, false))),
        );
    }

    /// Get all the uniforms in the set that haven't been used
    fn all_unused<'a>(&'a self) -> Vec<(String, expr::ExpressionType)> {
        self.uniforms
            .iter()
            .filter_map(|(name, (expression_type, used))| {
                if !*used && !name.starts_with('_') {
                    Some((name.clone(), *expression_type))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get the uniform with the given name.
    pub fn get(&mut self, name: &'static str) -> expr::Expression {
        self.uniforms
            .get_mut(name)
            .map(|(expression_type, used)| {
                *used = true;
                expr::Expression::__uniform(name, *expression_type)
            })
            .unwrap_or_else(|| panic!("Uniform {} not found", name))
    }

    /// Get the uniform with the given name.
    pub fn try_get(&mut self, name: &'static str) -> Option<expr::Expression> {
        self.uniforms.get_mut(name).map(|(expression_type, used)| {
            *used = true;
            expr::Expression::__uniform(name, *expression_type)
        })
    }

    /// Unwraps the internal hashmap for use in the final `Program`
    fn unwrap(self) -> HashMap<String, expr::ExpressionType> {
        self.uniforms
            .into_iter()
            .map(|(name, (expression_type, _))| (name, expression_type))
            .collect()
    }

    /// Get the camera position, if there is a camera attached to the program
    pub fn camera_position(&mut self) -> expr::Expression {
        self.try_get("_camera_position")
            .expect("Program was not created with a camera")
    }

    /// Get the camera direction, if there is a camera attached to the program
    pub fn camera_direction(&mut self) -> expr::Expression {
        self.try_get("_camera_direction")
            .expect("Program was not created with a camera")
    }

    /// Get the camera up vector, if there is a camera attached to the program
    pub fn camera_up(&mut self) -> expr::Expression {
        self.try_get("_camera_up")
            .expect("Program was not created with a camera")
    }

    /// Get the camera projection matrix, if there is a camera attached to the program
    pub fn camera_projection(&mut self) -> expr::Expression {
        self.try_get("_camera_projection")
            .expect("Program was not created with a camera")
    }

    /// Get the camera view matrix, if there is a camera attached to the program
    pub fn camera_view(&mut self) -> expr::Expression {
        let mut position = self.camera_position();
        let mut z = -self.camera_direction();
        let mut x = self.camera_up().cross(z.clone()).normalized();
        let mut y = z.clone().cross(x.clone());
        expr::Expression::matrix4x4(
            x.clone().x(),
            y.clone().x(),
            z.clone().x(),
            0.0,
            x.clone().y(),
            y.clone().y(),
            z.clone().y(),
            0.0,
            x.clone().z(),
            y.clone().z(),
            z.clone().z(),
            0.0,
            -position.clone().dot(x),
            -position.clone().dot(y),
            -position.dot(z),
            1.0,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Module {
    Vertex,
    Fragment,
}

impl Module {
    fn precedence(self) -> usize {
        match self {
            Module::Vertex => 0,
            Module::Fragment => 1,
        }
    }

    fn comes_after(self, other: Module) -> bool {
        self.precedence() > other.precedence()
    }
}

/// Write the main function for a module.
fn write_main(mut source: String, outputs: &ModuleOutputs, stage: Module) -> String {
    // Check that required outputs are present
    let required_outputs = match stage {
        Module::Vertex => vec!["gl_Position", "gl_PointSize"],
        Module::Fragment => vec!["out_frag_color"],
    };

    let missing_outputs = required_outputs
        .iter()
        .filter(|name| !outputs.outputs.contains_key(**name))
        .collect::<Vec<_>>();

    if missing_outputs.len() != 0 {
        panic!(
            "Missing required outputs in {:?} module: {:?}",
            stage, missing_outputs
        );
    }

    // Verify outputs; If there's an issue in an output this will cause a panic
    outputs.verify(stage);

    // Start main function
    source.push_str("void main() {\n");

    // Write outputs
    let mut source = outputs.iter().fold(source, |mut acc, output| {
        // Write output
        if !required_outputs.contains(&output.next_stage_input.name) {
            acc.push_str("\tout_");
        }
        acc.push_str(output.next_stage_input.name);
        acc.push_str(" = ");
        acc.push_str(&output.value.to_string());
        acc.push_str(";\n");
        acc
    });

    // End main function
    source.push_str("}");

    source
}

/// Expressions for building modules.
pub mod expr {
    use std::{
        any::TypeId,
        fmt::{Debug, Display},
        iter::{empty, once},
        ops::{Add, Div, Mul, Neg, Rem, Sub},
        sync::Arc,
    };

    use ggmath::prelude::{Matrix4x4, Vector2, Vector3, Vector4};

    use crate::gfx::UniformValue;

    /// An expression in a module's code
    #[derive(Debug)]
    pub struct Expression {
        class: Option<ExpressionClass>,
        cached_type: Option<ExpressionType>,
    }

    impl Expression {
        pub fn expression_type(&self) -> ExpressionType {
            self.cached_type
                .clone()
                .unwrap_or_else(|| self.class.as_ref().unwrap().expression_type())
        }

        pub fn validate_operands(&self) {
            self.class.as_ref().unwrap().validate_operands();
        }

        /// Get all the uniforms involved with this expression
        pub fn used_uniforms<'a>(&'a self) -> impl Iterator<Item = (&'a str, ExpressionType)> {
            self.class.as_ref().unwrap().used_uniforms()
        }

        pub fn __input(name: &'static str, expression_type: ExpressionType) -> Self {
            Self {
                class: Some(ExpressionClass::Input(name, expression_type)),
                cached_type: Some(expression_type),
            }
        }

        pub fn __uniform(name: &'static str, expression_type: ExpressionType) -> Self {
            Self {
                class: Some(ExpressionClass::Uniform(name, expression_type)),
                cached_type: Some(expression_type),
            }
        }

        pub fn matrix4x4(
            m00: impl Into<Expression>,
            m01: impl Into<Expression>,
            m02: impl Into<Expression>,
            m03: impl Into<Expression>,
            m10: impl Into<Expression>,
            m11: impl Into<Expression>,
            m12: impl Into<Expression>,
            m13: impl Into<Expression>,
            m20: impl Into<Expression>,
            m21: impl Into<Expression>,
            m22: impl Into<Expression>,
            m23: impl Into<Expression>,
            m30: impl Into<Expression>,
            m31: impl Into<Expression>,
            m32: impl Into<Expression>,
            m33: impl Into<Expression>,
        ) -> Self {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Matrix4x4(Box::new([
                    m00.into(),
                    m01.into(),
                    m02.into(),
                    m03.into(),
                    m10.into(),
                    m11.into(),
                    m12.into(),
                    m13.into(),
                    m20.into(),
                    m21.into(),
                    m22.into(),
                    m23.into(),
                    m30.into(),
                    m31.into(),
                    m32.into(),
                    m33.into(),
                ])))),
                cached_type: None,
            }
        }

        fn constant_f32(value: f32) -> Self {
            Self {
                class: Some(ExpressionClass::Constant(Constant::F32(value))),
                cached_type: Some(ExpressionType::Scalar(Scalar::F32)),
            }
        }

        fn constant_vec2f32(value: Vector2<f32>) -> Self {
            Self {
                class: Some(ExpressionClass::Constant(Constant::Vector2F32(value))),
                cached_type: Some(ExpressionType::Vector2(Scalar::F32)),
            }
        }

        fn constant_vec3f32(value: Vector3<f32>) -> Self {
            Self {
                class: Some(ExpressionClass::Constant(Constant::Vector3F32(value))),
                cached_type: Some(ExpressionType::Vector3(Scalar::F32)),
            }
        }

        fn constant_vec4f32(value: Vector4<f32>) -> Self {
            Self {
                class: Some(ExpressionClass::Constant(Constant::Vector4F32(value))),
                cached_type: Some(ExpressionType::Vector4(Scalar::F32)),
            }
        }

        fn constant_i32(value: i32) -> Self {
            Self {
                class: Some(ExpressionClass::Constant(Constant::I32(value))),
                cached_type: Some(ExpressionType::Scalar(Scalar::I32)),
            }
        }

        fn constant_vec2i32(value: Vector2<i32>) -> Self {
            Self {
                class: Some(ExpressionClass::Constant(Constant::Vector2I32(value))),
                cached_type: Some(ExpressionType::Vector2(Scalar::I32)),
            }
        }

        fn constant_vec3i32(value: Vector3<i32>) -> Self {
            Self {
                class: Some(ExpressionClass::Constant(Constant::Vector3I32(value))),
                cached_type: Some(ExpressionType::Vector3(Scalar::I32)),
            }
        }

        fn constant_vec4i32(value: Vector4<i32>) -> Self {
            Self {
                class: Some(ExpressionClass::Constant(Constant::Vector4I32(value))),
                cached_type: Some(ExpressionType::Vector4(Scalar::I32)),
            }
        }

        fn constant_mat4x4f32(value: Matrix4x4<f32>) -> Self {
            Self {
                class: Some(ExpressionClass::Constant(Constant::Matrix4x4(value))),
                cached_type: Some(ExpressionType::Matrix4x4(Scalar::F32)),
            }
        }

        pub fn add(self, other: impl Into<Expression>) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Add(
                    Box::new(self),
                    Box::new(other.into()),
                ))),
                cached_type: None,
            }
        }

        pub fn sub(self, other: impl Into<Expression>) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Sub(
                    Box::new(self),
                    Box::new(other.into()),
                ))),
                cached_type: None,
            }
        }

        pub fn mul(self, other: impl Into<Expression>) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Mul(
                    Box::new(self),
                    Box::new(other.into()),
                ))),
                cached_type: None,
            }
        }

        pub fn div(self, other: impl Into<Expression>) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Div(
                    Box::new(self),
                    Box::new(other.into()),
                ))),
                cached_type: None,
            }
        }

        pub fn rem(self, other: impl Into<Expression>) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Rem(
                    Box::new(self),
                    Box::new(other.into()),
                ))),
                cached_type: None,
            }
        }

        pub fn neg(self) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Neg(Box::new(self)))),
                cached_type: None,
            }
        }

        pub fn dot(self, other: impl Into<Expression>) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Dot(
                    Box::new(self),
                    Box::new(other.into()),
                ))),
                cached_type: None,
            }
        }

        pub fn cross(self, other: impl Into<Expression>) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Cross(
                    Box::new(self),
                    Box::new(other.into()),
                ))),
                cached_type: None,
            }
        }

        pub fn length(self) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Length(Box::new(self)))),
                cached_type: None,
            }
        }

        pub fn normalized(self) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Normalize(Box::new(
                    self,
                )))),
                cached_type: None,
            }
        }

        pub fn sin(self) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Sin(Box::new(self)))),
                cached_type: None,
            }
        }

        pub fn cos(self) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Cos(Box::new(self)))),
                cached_type: None,
            }
        }

        pub fn tan(self) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Tan(Box::new(self)))),
                cached_type: None,
            }
        }

        pub fn asin(self) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Asin(Box::new(self)))),
                cached_type: None,
            }
        }

        pub fn acos(self) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Acos(Box::new(self)))),
                cached_type: None,
            }
        }

        pub fn atan(self) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Atan(Box::new(self)))),
                cached_type: None,
            }
        }

        pub fn atan2(self, other: impl Into<Expression>) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Atan2(
                    Box::new(self),
                    Box::new(other.into()),
                ))),
                cached_type: None,
            }
        }

        pub fn pow(self, other: impl Into<Expression>) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Pow(
                    Box::new(self),
                    Box::new(other.into()),
                ))),
                cached_type: None,
            }
        }

        pub fn min(self, other: impl Into<Expression>) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Min(
                    Box::new(self),
                    Box::new(other.into()),
                ))),
                cached_type: None,
            }
        }

        pub fn max(self, other: impl Into<Expression>) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Max(
                    Box::new(self),
                    Box::new(other.into()),
                ))),
                cached_type: None,
            }
        }

        pub fn clamp(self, min: impl Into<Expression>, max: impl Into<Expression>) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Clamp(
                    Box::new(self),
                    Box::new(min.into()),
                    Box::new(max.into()),
                ))),
                cached_type: None,
            }
        }

        pub fn abs(self) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Abs(Box::new(self)))),
                cached_type: None,
            }
        }

        pub fn floor(self) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Floor(Box::new(self)))),
                cached_type: None,
            }
        }

        pub fn ceil(self) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Ceil(Box::new(self)))),
                cached_type: None,
            }
        }

        pub fn fract(self) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Fract(Box::new(self)))),
                cached_type: None,
            }
        }

        pub fn mix(self, other: impl Into<Expression>, t: impl Into<Expression>) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Mix(
                    Box::new(self),
                    Box::new(other.into()),
                    Box::new(t.into()),
                ))),
                cached_type: None,
            }
        }

        pub fn concat(self, other: impl Into<Expression>) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Concat(
                    Box::new(self),
                    Box::new(other.into()),
                ))),
                cached_type: None,
            }
        }

        fn convert_to(self, new_type: ExpressionType) -> Expression {
            // Exit early if the type is already the same
            if self.expression_type() == new_type {
                return self;
            }
            // Else convert the expression to the new type
            Self {
                class: Some(ExpressionClass::Operator(Operator::Convert(
                    Box::new(self),
                    new_type,
                ))),
                cached_type: Some(new_type),
            }
        }

        pub fn convert<V: UniformValue + 'static>(self) -> Expression {
            match TypeId::of::<V>() {
                const { TypeId::of::<f32>() } => {
                    self.convert_to(ExpressionType::Scalar(Scalar::F32))
                }
                const { TypeId::of::<Vector2<f32>>() } => {
                    self.convert_to(ExpressionType::Vector2(Scalar::F32))
                }
                const { TypeId::of::<Vector3<f32>>() } => {
                    self.convert_to(ExpressionType::Vector3(Scalar::F32))
                }
                const { TypeId::of::<Vector4<f32>>() } => {
                    self.convert_to(ExpressionType::Vector4(Scalar::F32))
                }
                const { TypeId::of::<i32>() } => {
                    self.convert_to(ExpressionType::Scalar(Scalar::I32))
                }
                const { TypeId::of::<Vector2<i32>>() } => {
                    self.convert_to(ExpressionType::Vector2(Scalar::I32))
                }
                const { TypeId::of::<Vector3<i32>>() } => {
                    self.convert_to(ExpressionType::Vector3(Scalar::I32))
                }
                const { TypeId::of::<Vector4<i32>>() } => {
                    self.convert_to(ExpressionType::Vector4(Scalar::I32))
                }
                _ => panic!("Unsupported type"),
            }
        }

        pub fn clone(&mut self) -> Expression {
            if let ExpressionClass::Clone(class) = self.class.as_ref().unwrap() {
                Self {
                    class: Some(ExpressionClass::Clone(class.clone())),
                    cached_type: self.cached_type.clone(),
                }
            } else {
                let cloned_class = Arc::new(self.class.take().unwrap());
                self.class = Some(ExpressionClass::Clone(cloned_class.clone()));

                Self {
                    class: Some(ExpressionClass::Clone(cloned_class)),
                    cached_type: self.cached_type.clone(),
                }
            }
        }

        pub fn get(self, index: impl Into<Expression>) -> Expression {
            Self {
                class: Some(ExpressionClass::Operator(Operator::Get(
                    Box::new(self),
                    Box::new(index.into()),
                ))),
                cached_type: None,
            }
        }

        pub fn x(self) -> Expression {
            self.get(0)
        }

        pub fn y(self) -> Expression {
            self.get(1)
        }

        pub fn z(self) -> Expression {
            self.get(2)
        }

        pub fn w(self) -> Expression {
            self.get(3)
        }

        pub fn xy(mut self) -> Expression {
            self.clone().get(0).concat(self.get(1))
        }

        pub fn yz(mut self) -> Expression {
            self.clone().get(1).concat(self.get(2))
        }

        pub fn zw(mut self) -> Expression {
            self.clone().get(2).concat(self.get(3))
        }

        pub fn xyz(mut self) -> Expression {
            self.clone()
                .get(0)
                .concat(self.clone().get(1).concat(self.get(2)))
        }

        pub fn yzw(mut self) -> Expression {
            self.clone()
                .get(1)
                .concat(self.clone().get(2).concat(self.get(3)))
        }
    }

    impl Display for Expression {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            fn fmt_class(class: &ExpressionClass, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                match class {
                    ExpressionClass::Input(name, _) => write!(f, "in_{}", name),
                    ExpressionClass::Uniform(name, _) => write!(f, "{}", name),
                    ExpressionClass::Constant(c) => write!(f, "{}", c),
                    ExpressionClass::Operator(op) => write!(f, "{}", op),
                    ExpressionClass::Clone(cloned) => fmt_class(&*cloned, f),
                }
            }
            fmt_class(self.class.as_ref().unwrap(), f)
        }
    }

    impl<T: Into<Expression>> Add<T> for Expression {
        type Output = Expression;

        fn add(self, other: T) -> Expression {
            Expression::add(self, other.into())
        }
    }

    impl<T: Into<Expression>> Sub<T> for Expression {
        type Output = Expression;

        fn sub(self, other: T) -> Expression {
            Expression::sub(self, other.into())
        }
    }

    impl<T: Into<Expression>> Mul<T> for Expression {
        type Output = Expression;

        fn mul(self, other: T) -> Expression {
            Expression::mul(self, other.into())
        }
    }

    impl<T: Into<Expression>> Div<T> for Expression {
        type Output = Expression;

        fn div(self, other: T) -> Expression {
            Expression::div(self, other.into())
        }
    }

    impl<T: Into<Expression>> Rem<T> for Expression {
        type Output = Expression;

        fn rem(self, other: T) -> Expression {
            Expression::rem(self, other.into())
        }
    }

    impl Neg for Expression {
        type Output = Expression;

        fn neg(self) -> Expression {
            Expression::neg(self)
        }
    }

    impl From<f32> for Expression {
        fn from(f: f32) -> Self {
            Self::constant_f32(f)
        }
    }

    impl From<Vector2<f32>> for Expression {
        fn from(v: Vector2<f32>) -> Self {
            Self::constant_vec2f32(v)
        }
    }

    impl From<Vector3<f32>> for Expression {
        fn from(v: Vector3<f32>) -> Self {
            Self::constant_vec3f32(v)
        }
    }

    impl From<Vector4<f32>> for Expression {
        fn from(v: Vector4<f32>) -> Self {
            Self::constant_vec4f32(v)
        }
    }

    impl From<i32> for Expression {
        fn from(i: i32) -> Self {
            Self::constant_i32(i)
        }
    }

    impl From<Vector2<i32>> for Expression {
        fn from(v: Vector2<i32>) -> Self {
            Self::constant_vec2i32(v)
        }
    }

    impl From<Vector3<i32>> for Expression {
        fn from(v: Vector3<i32>) -> Self {
            Self::constant_vec3i32(v)
        }
    }

    impl From<Vector4<i32>> for Expression {
        fn from(v: Vector4<i32>) -> Self {
            Self::constant_vec4i32(v)
        }
    }

    impl From<Matrix4x4<f32>> for Expression {
        fn from(m: Matrix4x4<f32>) -> Self {
            Self::constant_mat4x4f32(m)
        }
    }

    /// An expression's class
    pub enum ExpressionClass {
        /// An input expression
        Input(&'static str, ExpressionType),
        /// A uniform expression
        Uniform(&'static str, ExpressionType),
        /// A constant expression
        Constant(Constant),
        /// Operator expression
        Operator(Operator),
        /// A cloned expression
        Clone(Arc<ExpressionClass>),
    }

    impl ExpressionClass {
        fn expression_type(&self) -> ExpressionType {
            match self {
                Self::Input(_, ty) => *ty,
                Self::Uniform(_, ty) => *ty,
                Self::Constant(constant) => constant.expression_type(),
                Self::Operator(operator) => operator.expression_type(),
                Self::Clone(clone) => clone.expression_type(),
            }
        }

        fn validate_operands(&self) {
            match self {
                Self::Input(_, _) => (),
                Self::Uniform(_, _) => (),
                Self::Constant(_) => (),
                Self::Operator(operator) => operator.validate_operands(),
                Self::Clone(clone) => clone.validate_operands(),
            }
        }

        /// Get all the uniforms involved with this expression
        fn used_uniforms<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a str, ExpressionType)> + 'a> {
            match self {
                Self::Input(_, _) => Box::new(empty()),
                Self::Uniform(name, ty) => Box::new(once((*name, *ty))),
                Self::Constant(_) => Box::new(empty()),
                Self::Operator(operator) => operator.used_uniforms(),
                Self::Clone(clone) => clone.used_uniforms(),
            }
        }
    }

    impl Debug for ExpressionClass {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Input(name, ty) => write!(f, "Input({}: {})", name, ty.to_str()),
                Self::Uniform(name, ty) => write!(f, "Uniform({}: {})", name, ty.to_str()),
                Self::Constant(c) => write!(f, "{:?}", c),
                Self::Operator(o) => write!(f, "{:?}", o),
                Self::Clone(clone) => write!(f, "{:?}", &*clone),
            }
        }
    }

    /// An expression's value type
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum ExpressionType {
        /// A scalar value
        Scalar(Scalar),
        /// A vector of 2 components
        Vector2(Scalar),
        /// A vector of 3 components
        Vector3(Scalar),
        /// A vector of 4 components
        Vector4(Scalar),
        /// A matrix of 4x4 components
        Matrix4x4(Scalar),
    }

    impl ExpressionType {
        /// If the expression's type signature is a vector or matrix type,
        /// returns the scalar type of its components.
        /// If the expression has a scalar type signature,
        /// returns this scalar type.
        fn scalar(self) -> Scalar {
            match self {
                Self::Scalar(s) => s,
                Self::Vector2(s) => s,
                Self::Vector3(s) => s,
                Self::Vector4(s) => s,
                Self::Matrix4x4(s) => s,
            }
        }

        fn width(self) -> Option<usize> {
            Some(match self {
                Self::Vector2(_) => 2,
                Self::Vector3(_) => 3,
                Self::Vector4(_) => 4,
                _ => return None,
            })
        }

        fn row_width(self) -> Option<usize> {
            Some(match self {
                Self::Matrix4x4(_) => 4,
                _ => return None,
            })
        }

        fn is_scalar(self) -> bool {
            match self {
                Self::Scalar(_) => true,
                _ => false,
            }
        }

        fn is_vector(self) -> bool {
            match self {
                Self::Vector2(_) => true,
                Self::Vector3(_) => true,
                Self::Vector4(_) => true,
                _ => false,
            }
        }

        fn is_matrix(self) -> bool {
            match self {
                Self::Matrix4x4(_) => true,
                _ => false,
            }
        }

        pub fn to_str(self) -> &'static str {
            match self {
                Self::Scalar(s) => s.to_str(),
                Self::Vector2(s) => match s {
                    Scalar::F32 => "vec2",
                    Scalar::I32 => "ivec2",
                    Scalar::Bool => panic!("Cannot create a vector of booleans"),
                },
                Self::Vector3(s) => match s {
                    Scalar::F32 => "vec3",
                    Scalar::I32 => "ivec3",
                    Scalar::Bool => panic!("Cannot create a vector of booleans"),
                },
                Self::Vector4(s) => match s {
                    Scalar::F32 => "vec4",
                    Scalar::I32 => "ivec4",
                    Scalar::Bool => panic!("Cannot create a vector of booleans"),
                },
                Self::Matrix4x4(s) => match s {
                    Scalar::F32 => "mat4",
                    Scalar::I32 => "imat4",
                    Scalar::Bool => panic!("Cannot create a vector of booleans"),
                },
            }
        }

        fn plus_width(self, width: usize) -> Option<Self> {
            let own_width = if self.is_scalar() { 1 } else { self.width()? };
            Some(match own_width + width {
                2 => Self::Vector2(self.scalar()),
                3 => Self::Vector3(self.scalar()),
                4 => Self::Vector4(self.scalar()),
                _ => return None,
            })
        }

        /// Get the number of location slots consumed by an input or uniform with this expression type
        pub fn locations_consumed(self) -> usize {
            match self {
                Self::Scalar(s) | Self::Vector2(s) => match s {
                    Scalar::F32 | Scalar::I32 | Scalar::Bool => 1,
                },
                Self::Vector3(s) => match s {
                    Scalar::F32 | Scalar::I32 => 1,
                    Scalar::Bool => panic!("Cannot create a vector of booleans"),
                },
                Self::Vector4(s) => match s {
                    Scalar::F32 | Scalar::I32 => 1,
                    Scalar::Bool => panic!("Cannot create a vector of booleans"),
                },
                Self::Matrix4x4(s) => match s {
                    Scalar::F32 | Scalar::I32 => 4,
                    Scalar::Bool => panic!("Cannot create a vector of booleans"),
                },
            }
        }
    }

    /// An scalar type for building an ExpressionType
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Scalar {
        F32,
        I32,
        Bool,
    }

    impl Scalar {
        fn to_str(self) -> &'static str {
            match self {
                Scalar::F32 => "float",
                Scalar::I32 => "int",
                Scalar::Bool => "bool",
            }
        }
    }

    /// A constant value
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub enum Constant {
        F32(f32),
        Vector2F32(Vector2<f32>),
        Vector3F32(Vector3<f32>),
        Vector4F32(Vector4<f32>),
        I32(i32),
        Vector2I32(Vector2<i32>),
        Vector3I32(Vector3<i32>),
        Vector4I32(Vector4<i32>),
        Matrix4x4(Matrix4x4<f32>),
        Bool(bool),
    }

    impl Constant {
        fn expression_type(&self) -> ExpressionType {
            match self {
                Self::F32(_) => ExpressionType::Scalar(Scalar::F32),
                Self::Vector2F32(_) => ExpressionType::Vector2(Scalar::F32),
                Self::Vector3F32(_) => ExpressionType::Vector3(Scalar::F32),
                Self::Vector4F32(_) => ExpressionType::Vector4(Scalar::F32),
                Self::I32(_) => ExpressionType::Scalar(Scalar::I32),
                Self::Vector2I32(_) => ExpressionType::Vector2(Scalar::I32),
                Self::Vector3I32(_) => ExpressionType::Vector3(Scalar::I32),
                Self::Vector4I32(_) => ExpressionType::Vector4(Scalar::I32),
                Self::Matrix4x4(_) => ExpressionType::Matrix4x4(Scalar::F32),
                Self::Bool(_) => ExpressionType::Scalar(Scalar::Bool),
            }
        }
    }

    impl Display for Constant {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                Self::F32(v) => write!(f, "{:.8}", v),
                Self::Vector2F32(v) => write!(
                    f,
                    "vec2({:.8}, {:.8})",
                    v.x(),
                    v.y(),
                ),
                Self::Vector3F32(v) => write!(
                    f,
                    "vec3({:.8}, {:.8}, {:.8})",
                    v.x(),
                    v.y(),
                    v.z(),
                ),
                Self::Vector4F32(v) => write!(
                    f,
                    "vec4({:.8}, {:.8}, {:.8}, {:.8})",
                    v.x(),
                    v.y(),
                    v.z(),
                    v.w(),
                ),
                Self::I32(v) => write!(f, "{}", v),
                Self::Vector2I32(v) => write!(
                    f,
                    "ivec2({}, {})",
                    v.x(),
                    v.y(),
                ),
                Self::Vector3I32(v) => write!(
                    f,
                    "ivec3({}, {}, {})",
                    v.x(),
                    v.y(),
                    v.z(),
                ),
                Self::Vector4I32(v) => write!(
                    f,
                    "ivec4({}, {}, {}, {})",
                    v.x(),
                    v.y(),
                    v.z(),
                    v.w(),
                ),
                Self::Matrix4x4(v) => write!(
                    f,
                    "mat4({:.8}, {:.8}, {:.8}, {:.8}, {:.8}, {:.8}, {:.8}, {:.8}, {:.8}, {:.8}, {:.8}, {:.8}, {:.8}, {:.8}, {:.8}, {:.8})",
                    v.as_row(0).unwrap().const_column(0),
                    v.as_row(0).unwrap().const_column(1),
                    v.as_row(0).unwrap().const_column(2),
                    v.as_row(0).unwrap().const_column(3),
                    v.as_row(1).unwrap().const_column(0),
                    v.as_row(1).unwrap().const_column(1),
                    v.as_row(1).unwrap().const_column(2),
                    v.as_row(1).unwrap().const_column(3),
                    v.as_row(2).unwrap().const_column(0),
                    v.as_row(2).unwrap().const_column(1),
                    v.as_row(2).unwrap().const_column(2),
                    v.as_row(2).unwrap().const_column(3),
                    v.as_row(3).unwrap().const_column(0),
                    v.as_row(3).unwrap().const_column(1),
                    v.as_row(3).unwrap().const_column(2),
                    v.as_row(3).unwrap().const_column(3),
                ),
                Self::Bool(v) => write!(f, "{}", v),
            }
        }
    }

    /// An operator value
    #[derive(Debug)]
    pub enum Operator {
        Add(Box<Expression>, Box<Expression>),
        Sub(Box<Expression>, Box<Expression>),
        Mul(Box<Expression>, Box<Expression>),
        Div(Box<Expression>, Box<Expression>),
        Rem(Box<Expression>, Box<Expression>),
        Neg(Box<Expression>),
        Dot(Box<Expression>, Box<Expression>),
        Cross(Box<Expression>, Box<Expression>),
        Length(Box<Expression>),
        Normalize(Box<Expression>),
        Min(Box<Expression>, Box<Expression>),
        Max(Box<Expression>, Box<Expression>),
        Clamp(Box<Expression>, Box<Expression>, Box<Expression>),
        Floor(Box<Expression>),
        Ceil(Box<Expression>),
        Round(Box<Expression>),
        Fract(Box<Expression>),
        Sin(Box<Expression>),
        Cos(Box<Expression>),
        Tan(Box<Expression>),
        Asin(Box<Expression>),
        Acos(Box<Expression>),
        Atan(Box<Expression>),
        Atan2(Box<Expression>, Box<Expression>),
        Pow(Box<Expression>, Box<Expression>),
        Exp(Box<Expression>),
        Log(Box<Expression>),
        Sqrt(Box<Expression>),
        InverseSqrt(Box<Expression>),
        Abs(Box<Expression>),
        Sign(Box<Expression>),
        Mix(Box<Expression>, Box<Expression>, Box<Expression>),
        Concat(Box<Expression>, Box<Expression>),
        Convert(Box<Expression>, ExpressionType),
        Get(Box<Expression>, Box<Expression>),
        Matrix4x4(Box<[Expression; 16]>),
    }

    impl Operator {
        fn validate_operands(&self) {
            // Validate types as well as pass validation down the tree
            if let Some((operator_str, operand_types)) = match self {
                Operator::Add(a, b) => {
                    a.validate_operands();
                    b.validate_operands();
                    let a_type = a.expression_type();
                    let b_type = b.expression_type();
                    if a_type != b_type
                        && (a_type.scalar() != b_type.scalar() || !b_type.is_scalar())
                    {
                        Some(("+", vec![a_type, b_type]))
                    } else {
                        None
                    }
                }
                Operator::Sub(a, b) => {
                    a.validate_operands();
                    b.validate_operands();
                    let a_type = a.expression_type();
                    let b_type = b.expression_type();
                    if a_type != b_type
                        && (a_type.scalar() != b_type.scalar() || !b_type.is_scalar())
                    {
                        Some(("-", vec![a_type, b_type]))
                    } else {
                        None
                    }
                }
                Operator::Mul(a, b) => {
                    a.validate_operands();
                    b.validate_operands();
                    let a_type = a.expression_type();
                    let b_type = b.expression_type();
                    if a_type != b_type
                        && (a_type.scalar() != b_type.scalar()
                            || !((a_type.is_matrix() && b_type.width() == a_type.row_width())
                                || (a_type.is_vector() && b_type.is_scalar())))
                    {
                        Some(("*", vec![a_type, b_type]))
                    } else {
                        None
                    }
                }
                Operator::Div(a, b) => {
                    a.validate_operands();
                    b.validate_operands();
                    let a_type = a.expression_type();
                    let b_type = b.expression_type();
                    if a_type != b_type
                        && (a_type.scalar() != b_type.scalar() || !b_type.is_scalar())
                    {
                        Some(("/", vec![a_type, b_type]))
                    } else {
                        None
                    }
                }
                Operator::Rem(a, b) => {
                    a.validate_operands();
                    b.validate_operands();
                    let a_type = a.expression_type();
                    let b_type = b.expression_type();
                    if a_type != b_type
                        && (a_type.scalar() != b_type.scalar() || !b_type.is_scalar())
                    {
                        Some(("%", vec![a_type, b_type]))
                    } else {
                        None
                    }
                }
                Operator::Neg(a) => {
                    a.validate_operands();
                    None
                }
                Operator::Dot(a, b) => {
                    a.validate_operands();
                    b.validate_operands();
                    let a_type = a.expression_type();
                    let b_type = b.expression_type();
                    if !a_type.is_vector() || a_type != b_type {
                        Some(("dot", vec![a_type, b_type]))
                    } else {
                        None
                    }
                }
                Operator::Cross(a, b) => {
                    a.validate_operands();
                    b.validate_operands();
                    let a_type = a.expression_type();
                    let b_type = b.expression_type();
                    if !a_type.is_vector() || a_type != b_type {
                        Some(("cross", vec![a_type, b_type]))
                    } else {
                        None
                    }
                }
                Operator::Length(a) => {
                    a.validate_operands();
                    let a_type = a.expression_type();
                    if !a_type.is_vector() {
                        Some(("length", vec![a_type]))
                    } else {
                        None
                    }
                }
                Operator::Normalize(a) => {
                    a.validate_operands();
                    let a_type = a.expression_type();
                    if !a_type.is_vector() {
                        Some(("normalize", vec![a_type]))
                    } else {
                        None
                    }
                }
                Operator::Min(a, b) => {
                    a.validate_operands();
                    b.validate_operands();
                    let a_type = a.expression_type();
                    let b_type = b.expression_type();
                    if a_type != b_type
                        && (a_type.scalar() != b_type.scalar() || !b_type.is_scalar())
                    {
                        Some(("min", vec![a_type, b_type]))
                    } else {
                        None
                    }
                }
                Operator::Max(a, b) => {
                    a.validate_operands();
                    b.validate_operands();
                    let a_type = a.expression_type();
                    let b_type = b.expression_type();
                    if a_type != b_type
                        && (a_type.scalar() != b_type.scalar() || !b_type.is_scalar())
                    {
                        Some(("max", vec![a_type, b_type]))
                    } else {
                        None
                    }
                }
                Operator::Clamp(a, b, c) => {
                    a.validate_operands();
                    b.validate_operands();
                    c.validate_operands();
                    let a_type = a.expression_type();
                    let b_type = b.expression_type();
                    let c_type = c.expression_type();
                    if a_type != b_type && a_type != c_type {
                        Some(("clamp", vec![a_type, b_type, c_type]))
                    } else {
                        None
                    }
                }
                Operator::Floor(a) => {
                    a.validate_operands();
                    let a_type = a.expression_type();
                    if a_type.scalar() != Scalar::F32 {
                        Some(("floor", vec![a_type]))
                    } else {
                        None
                    }
                }
                Operator::Ceil(a) => {
                    a.validate_operands();
                    let a_type = a.expression_type();
                    if a_type.scalar() != Scalar::F32 {
                        Some(("ceil", vec![a_type]))
                    } else {
                        None
                    }
                }
                Operator::Round(a) => {
                    a.validate_operands();
                    let a_type = a.expression_type();
                    if a_type.scalar() != Scalar::F32 {
                        Some(("round", vec![a_type]))
                    } else {
                        None
                    }
                }
                Operator::Fract(a) => {
                    a.validate_operands();
                    let a_type = a.expression_type();
                    if a_type.scalar() != Scalar::F32 {
                        Some(("frac", vec![a_type]))
                    } else {
                        None
                    }
                }
                Operator::Sin(a) => {
                    a.validate_operands();
                    let a_type = a.expression_type();
                    if a_type.scalar() != Scalar::F32 {
                        Some(("sin", vec![a_type]))
                    } else {
                        None
                    }
                }
                Operator::Cos(a) => {
                    a.validate_operands();
                    let a_type = a.expression_type();
                    if a_type.scalar() != Scalar::F32 {
                        Some(("cos", vec![a_type]))
                    } else {
                        None
                    }
                }
                Operator::Tan(a) => {
                    a.validate_operands();
                    let a_type = a.expression_type();
                    if a_type.scalar() != Scalar::F32 {
                        Some(("tan", vec![a_type]))
                    } else {
                        None
                    }
                }
                Operator::Asin(a) => {
                    a.validate_operands();
                    let a_type = a.expression_type();
                    if a_type.scalar() != Scalar::F32 {
                        Some(("asin", vec![a_type]))
                    } else {
                        None
                    }
                }
                Operator::Acos(a) => {
                    a.validate_operands();
                    let a_type = a.expression_type();
                    if a_type.scalar() != Scalar::F32 {
                        Some(("acos", vec![a_type]))
                    } else {
                        None
                    }
                }
                Operator::Atan(a) => {
                    a.validate_operands();
                    let a_type = a.expression_type();
                    if a_type.scalar() != Scalar::F32 {
                        Some(("atan", vec![a_type]))
                    } else {
                        None
                    }
                }
                Operator::Atan2(a, b) => {
                    a.validate_operands();
                    b.validate_operands();
                    let a_type = a.expression_type();
                    let b_type = b.expression_type();
                    if !a_type.is_scalar() || a_type != b_type {
                        Some(("atan2", vec![a_type, b_type]))
                    } else {
                        None
                    }
                }
                Operator::Pow(a, b) => {
                    a.validate_operands();
                    b.validate_operands();
                    let a_type = a.expression_type();
                    let b_type = b.expression_type();
                    if a_type != ExpressionType::Scalar(Scalar::F32)
                        || b_type != ExpressionType::Scalar(Scalar::F32)
                    {
                        Some(("pow", vec![a_type, b_type]))
                    } else {
                        None
                    }
                }
                Operator::Exp(a) => {
                    a.validate_operands();
                    let a_type = a.expression_type();
                    if a_type != ExpressionType::Scalar(Scalar::F32) {
                        Some(("exp", vec![a_type]))
                    } else {
                        None
                    }
                }
                Operator::Log(a) => {
                    a.validate_operands();
                    let a_type = a.expression_type();
                    if a_type != ExpressionType::Scalar(Scalar::F32) {
                        Some(("log", vec![a_type]))
                    } else {
                        None
                    }
                }
                Operator::Sqrt(a) => {
                    a.validate_operands();
                    let a_type = a.expression_type();
                    if a_type != ExpressionType::Scalar(Scalar::F32) {
                        Some(("sqrt", vec![a_type]))
                    } else {
                        None
                    }
                }
                Operator::InverseSqrt(a) => {
                    a.validate_operands();
                    let a_type = a.expression_type();
                    if a_type != ExpressionType::Scalar(Scalar::F32) {
                        Some(("inversesqrt", vec![a_type]))
                    } else {
                        None
                    }
                }
                Operator::Abs(a) => {
                    a.validate_operands();
                    None
                }
                Operator::Sign(a) => {
                    a.validate_operands();
                    let a_type = a.expression_type();
                    if a_type != ExpressionType::Scalar(Scalar::F32) {
                        Some(("sign", vec![a_type]))
                    } else {
                        None
                    }
                }
                Operator::Mix(a, b, c) => {
                    a.validate_operands();
                    b.validate_operands();
                    c.validate_operands();
                    let a_type = a.expression_type();
                    let b_type = b.expression_type();
                    let c_type = c.expression_type();
                    if a_type != b_type
                        || a_type.scalar() != Scalar::F32
                        || c_type != ExpressionType::Scalar(Scalar::F32)
                    {
                        Some(("mix", vec![a_type, b_type, c_type]))
                    } else {
                        None
                    }
                }
                Operator::Concat(a, b) => {
                    a.validate_operands();
                    b.validate_operands();
                    let a_type = a.expression_type();
                    let b_type = b.expression_type();
                    if a_type.scalar() != b_type.scalar()
                        || (!a_type.is_vector() && !a_type.is_scalar())
                        || (!b_type.is_vector() && !b_type.is_scalar())
                        // The unwrap is safe because we know that the types are scalars or vectors.
                        || a_type.width().unwrap_or_default() + b_type.width().unwrap_or_default() > 4
                    {
                        Some(("concat", vec![a_type, b_type]))
                    } else {
                        None
                    }
                }
                Operator::Convert(a, _) => {
                    a.validate_operands();
                    None
                }
                Operator::Get(a, b) => {
                    a.validate_operands();
                    b.validate_operands();
                    let a_type = a.expression_type();
                    let b_type = b.expression_type();
                    if !(a_type.is_vector() || a_type.is_matrix())
                        || b_type != ExpressionType::Scalar(Scalar::I32)
                    {
                        Some(("get", vec![a_type, b_type]))
                    } else {
                        None
                    }
                }
                Operator::Matrix4x4(a) => {
                    for e in a.iter() {
                        e.validate_operands();
                    }
                    let expected_type = a[0].expression_type();
                    if expected_type != ExpressionType::Scalar(Scalar::F32)
                        && expected_type != ExpressionType::Scalar(Scalar::I32)
                    {
                        Some((
                            "matrix4x4",
                            a.iter().map(Expression::expression_type).collect(),
                        ))
                    } else {
                        None
                    }
                }
            } {
                // Panic if the operator is not valid for its operand types.
                panic!(
                    "Invalid operand type(s) for {} operator: {}",
                    operator_str,
                    operand_types.into_iter().map(ExpressionType::to_str).fold(
                        String::new(),
                        |mut acc, type_str| {
                            if !acc.is_empty() {
                                acc.push_str(", ");
                            }
                            acc.push_str(type_str);
                            acc
                        }
                    )
                );
            }

            // Validate values
            match self {
                Self::Convert(a, b_type) => {
                    let a_type = a.expression_type();
                    match a_type {
                        ExpressionType::Scalar(_) => match *b_type {
                            ExpressionType::Scalar(_) => {}
                            b_type => {
                                panic!("Cannot convert {} to {}", a_type.to_str(), b_type.to_str())
                            }
                        },
                        ExpressionType::Vector2(_) => match *b_type {
                            ExpressionType::Vector2(_) => {}
                            b_type => {
                                panic!("Cannot convert {} to {}", a_type.to_str(), b_type.to_str())
                            }
                        },
                        ExpressionType::Vector3(_) => match *b_type {
                            ExpressionType::Vector3(_) => {}
                            b_type => {
                                panic!("Cannot convert {} to {}", a_type.to_str(), b_type.to_str())
                            }
                        },
                        ExpressionType::Vector4(_) => match *b_type {
                            ExpressionType::Vector4(_) => {}
                            b_type => {
                                panic!("Cannot convert {} to {}", a_type.to_str(), b_type.to_str())
                            }
                        },
                        ExpressionType::Matrix4x4(_) => match *b_type {
                            ExpressionType::Matrix4x4(_) => {}
                            b_type => {
                                panic!("Cannot convert {} to {}", a_type.to_str(), b_type.to_str())
                            }
                        },
                    }
                }
                Self::Get(a, b) => {
                    if let ExpressionClass::Constant(Constant::I32(index)) =
                        b.class.as_ref().unwrap()
                    {
                        // The below unwrap is valid because the previous type check ensures that the type of a has a width.
                        if *index < 0 || *index >= a.expression_type().width().unwrap() as i32 {
                            panic!("Invalid constant index {} for {}", *index, a.to_string());
                        }
                    }
                }
                _ => {}
            }
        }

        fn expression_type(&self) -> ExpressionType {
            match self {
                Operator::Add(a, _) => a.expression_type(),
                Operator::Sub(a, _) => a.expression_type(),
                Operator::Mul(a, b) => {
                    let b_type = b.expression_type();
                    if b_type.is_vector() {
                        b_type
                    } else {
                        a.expression_type()
                    }
                }
                Operator::Div(a, _) => a.expression_type(),
                Operator::Rem(a, _) => a.expression_type(),
                Operator::Neg(a) => a.expression_type(),
                Operator::Dot(_, _) => ExpressionType::Scalar(Scalar::F32),
                Operator::Cross(a, _) => a.expression_type(),
                Operator::Length(_) => ExpressionType::Scalar(Scalar::F32),
                Operator::Normalize(a) => a.expression_type(),
                Operator::Min(a, _) => a.expression_type(),
                Operator::Max(a, _) => a.expression_type(),
                Operator::Clamp(a, _, _) => a.expression_type(),
                Operator::Floor(a) => a.expression_type(),
                Operator::Ceil(a) => a.expression_type(),
                Operator::Round(a) => a.expression_type(),
                Operator::Fract(a) => a.expression_type(),
                Operator::Sin(a) => a.expression_type(),
                Operator::Cos(a) => a.expression_type(),
                Operator::Tan(a) => a.expression_type(),
                Operator::Asin(a) => a.expression_type(),
                Operator::Acos(a) => a.expression_type(),
                Operator::Atan(a) => a.expression_type(),
                Operator::Atan2(a, _) => a.expression_type(),
                Operator::Pow(a, _) => a.expression_type(),
                Operator::Exp(a) => a.expression_type(),
                Operator::Log(a) => a.expression_type(),
                Operator::Sqrt(a) => a.expression_type(),
                Operator::InverseSqrt(a) => a.expression_type(),
                Operator::Abs(a) => a.expression_type(),
                Operator::Sign(_) => ExpressionType::Scalar(Scalar::I32),
                Operator::Mix(a, _, _) => a.expression_type(),
                Operator::Concat(a, b) => {
                    let a_type = a.expression_type();
                    let b_type = b.expression_type();
                    if b_type.is_vector() {
                        a_type.plus_width(b_type.width().unwrap()).unwrap_or(a_type)
                    } else if b_type.is_scalar() {
                        a_type.plus_width(1).unwrap_or(a_type)
                    } else {
                        ExpressionType::Scalar(Scalar::F32)
                    }
                }
                Operator::Convert(_, ty) => *ty,
                Operator::Get(a, _) => ExpressionType::Scalar(a.expression_type().scalar()),
                Operator::Matrix4x4(a) => {
                    ExpressionType::Matrix4x4(a[0].expression_type().scalar())
                }
            }
        }

        /// Get all the uniforms involved with this expression
        fn used_uniforms<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a str, ExpressionType)> + 'a> {
            match self {
                Self::Add(a, b) => Box::new(a.used_uniforms().chain(b.used_uniforms())),
                Self::Sub(a, b) => Box::new(a.used_uniforms().chain(b.used_uniforms())),
                Self::Mul(a, b) => Box::new(a.used_uniforms().chain(b.used_uniforms())),
                Self::Div(a, b) => Box::new(a.used_uniforms().chain(b.used_uniforms())),
                Self::Rem(a, b) => Box::new(a.used_uniforms().chain(b.used_uniforms())),
                Self::Neg(a) => Box::new(a.used_uniforms()),
                Self::Dot(a, b) => Box::new(a.used_uniforms().chain(b.used_uniforms())),
                Self::Cross(a, b) => Box::new(a.used_uniforms().chain(b.used_uniforms())),
                Self::Length(a) => Box::new(a.used_uniforms()),
                Self::Normalize(a) => Box::new(a.used_uniforms()),
                Self::Min(a, b) => Box::new(a.used_uniforms().chain(b.used_uniforms())),
                Self::Max(a, b) => Box::new(a.used_uniforms().chain(b.used_uniforms())),
                Self::Clamp(a, b, c) => Box::new(
                    a.used_uniforms()
                        .chain(b.used_uniforms())
                        .chain(c.used_uniforms()),
                ),
                Self::Floor(a) => Box::new(a.used_uniforms()),
                Self::Ceil(a) => Box::new(a.used_uniforms()),
                Self::Round(a) => Box::new(a.used_uniforms()),
                Self::Fract(a) => Box::new(a.used_uniforms()),
                Self::Sin(a) => Box::new(a.used_uniforms()),
                Self::Cos(a) => Box::new(a.used_uniforms()),
                Self::Tan(a) => Box::new(a.used_uniforms()),
                Self::Asin(a) => Box::new(a.used_uniforms()),
                Self::Acos(a) => Box::new(a.used_uniforms()),
                Self::Atan(a) => Box::new(a.used_uniforms()),
                Self::Atan2(a, b) => Box::new(a.used_uniforms().chain(b.used_uniforms())),
                Self::Pow(a, b) => Box::new(a.used_uniforms().chain(b.used_uniforms())),
                Self::Exp(a) => Box::new(a.used_uniforms()),
                Self::Log(a) => Box::new(a.used_uniforms()),
                Self::Sqrt(a) => Box::new(a.used_uniforms()),
                Self::InverseSqrt(a) => Box::new(a.used_uniforms()),
                Self::Abs(a) => Box::new(a.used_uniforms()),
                Self::Sign(a) => Box::new(a.used_uniforms()),
                Self::Mix(a, b, c) => Box::new(
                    a.used_uniforms()
                        .chain(b.used_uniforms())
                        .chain(c.used_uniforms()),
                ),
                Self::Concat(a, b) => Box::new(a.used_uniforms().chain(b.used_uniforms())),
                Self::Convert(a, _) => Box::new(a.used_uniforms()),
                Self::Get(a, b) => Box::new(a.used_uniforms().chain(b.used_uniforms())),
                Self::Matrix4x4(a) => Box::new(a.iter().flat_map(|x| x.used_uniforms())),
            }
        }
    }

    impl Display for Operator {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                Operator::Add(a, b) => write!(f, "({}) + ({})", a, b),
                Operator::Sub(a, b) => write!(f, "({}) - ({})", a, b),
                Operator::Mul(a, b) => write!(f, "({}) * ({})", a, b),
                Operator::Div(a, b) => write!(f, "({}) / ({})", a, b),
                Operator::Rem(a, b) => write!(f, "({}) % ({})", a, b),
                Operator::Neg(a) => write!(f, "-({})", a),
                Operator::Dot(a, b) => write!(f, "dot({}, {})", a, b),
                Operator::Cross(a, b) => write!(f, "cross({}, {})", a, b),
                Operator::Length(a) => write!(f, "length({})", a),
                Operator::Normalize(a) => write!(f, "normalize({})", a),
                Operator::Min(a, b) => write!(f, "min({}, {})", a, b),
                Operator::Max(a, b) => write!(f, "max({}, {})", a, b),
                Operator::Clamp(a, b, c) => write!(f, "clamp({}, {}, {})", a, b, c),
                Operator::Floor(a) => write!(f, "floor({})", a),
                Operator::Ceil(a) => write!(f, "ceil({})", a),
                Operator::Round(a) => write!(f, "round({})", a),
                Operator::Fract(a) => write!(f, "fract({})", a),
                Operator::Sin(a) => write!(f, "sin({})", a),
                Operator::Cos(a) => write!(f, "cos({})", a),
                Operator::Tan(a) => write!(f, "tan({})", a),
                Operator::Asin(a) => write!(f, "asin({})", a),
                Operator::Acos(a) => write!(f, "acos({})", a),
                Operator::Atan(a) => write!(f, "atan({})", a),
                Operator::Atan2(a, b) => write!(f, "atan2({}, {})", a, b),
                Operator::Pow(a, b) => write!(f, "pow({}, {})", a, b),
                Operator::Exp(a) => write!(f, "exp({})", a),
                Operator::Log(a) => write!(f, "log({})", a),
                Operator::Sqrt(a) => write!(f, "sqrt({})", a),
                Operator::InverseSqrt(a) => write!(f, "inversesqrt({})", a),
                Operator::Abs(a) => write!(f, "abs({})", a),
                Operator::Sign(a) => write!(f, "sign({})", a),
                Operator::Mix(a, b, c) => write!(f, "mix({}, {}, {})", a, b, c),
                Operator::Concat(a, b) => {
                    write!(f, "{}({}, {})", self.expression_type().to_str(), a, b)
                }
                Operator::Convert(a, ty) => write!(f, "{}({})", ty.to_str(), a),
                Operator::Get(a, b) => write!(f, "({})[{}]", a, b),
                Operator::Matrix4x4(a) => {
                    write!(f, "mat4(")?;
                    for (i, x) in a.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", x)?;
                    }
                    write!(f, ")")
                }
            }
        }
    }
}
