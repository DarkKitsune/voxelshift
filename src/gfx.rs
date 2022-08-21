use std::{
    borrow::Cow,
    cell::RefCell,
    collections::HashMap,
    ffi::CString,
    marker::PhantomData,
    mem::size_of,
    rc::Rc,
    sync::{Arc, RwLock},
};

use bitflags::bitflags;
use ggmath::prelude::{Matrix4x4, Vector2, Vector3, Vector4};
use gl::types::{GLchar, GLenum, GLint, GLsizei, GLsizeiptr, GLuint};
use lazy_static::lazy_static;

use crate::{
    app_window::AppWindow,
    colors::Color,
    program_builder::expr::{ExpressionType, Scalar},
    proto_mesh::ProtoMesh,
    vertex::Vertex,
};

lazy_static! {
    /// Keeps track of whether the Gfx has been initialized.
    pub static ref GFX_ALREADY_CREATED: Arc<RwLock<bool>> = Arc::new(RwLock::new(false));
}

/// The app's graphics controller
#[derive(Clone, Debug)]
pub struct Gfx {
    base: Rc<RefCell<GfxBase>>,
}

impl Gfx {
    pub fn new(window: &mut AppWindow) -> Gfx {
        // Guard against creating multiple instances of the graphics system
        if *GFX_ALREADY_CREATED.read().unwrap() {
            panic!("Gfx already created");
        }
        *GFX_ALREADY_CREATED.write().unwrap() = true;

        gl::load_with(|name| window.__get_proc_address(name));

        // Default settings
        unsafe {
            gl::Enable(gl::DEPTH_TEST);
            gl::DepthFunc(gl::LESS);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            gl::Enable(gl::CULL_FACE);
            gl::CullFace(gl::BACK);
            gl::FrontFace(gl::CCW);
        }

        // Create Gfx object
        let gfx = Gfx {
            base: Rc::new(RefCell::new(GfxBase)),
        };

        // Give it to the window
        window.set_gfx(&gfx);

        gfx
    }

    /// Should be called once per frame automatically by AppWindow::update; don't call it manually
    pub fn __update(&self) {
        // Check for GL errors
        #[cfg(debug_assertions)]
        match unsafe { gl::GetError() } {
            gl::NO_ERROR => {}
            gl::INVALID_ENUM => panic!("GL_INVALID_ENUM"),
            gl::INVALID_VALUE => panic!("GL_INVALID_VALUE"),
            gl::INVALID_OPERATION => panic!("GL_INVALID_OPERATION"),
            gl::INVALID_FRAMEBUFFER_OPERATION => panic!("GL_INVALID_FRAMEBUFFER_OPERATION"),
            gl::OUT_OF_MEMORY => panic!("GL_OUT_OF_MEMORY"),
            gl::STACK_UNDERFLOW => panic!("GL_STACK_UNDERFLOW"),
            gl::STACK_OVERFLOW => panic!("GL_STACK_OVERFLOW"),
            error => {
                panic!("OpenGL error: {}", error);
            }
        }
    }

    /// Creates a shader object
    pub fn create_shader(&self, shader_type: ShaderType, source: impl Into<Vec<u8>>) -> Shader {
        Shader::new(
            shader_type,
            CString::new(source).expect("Shader source contained a null byte"),
        )
    }

    /// Creates a program object; use `ProgramBuilder` instead unless you know what you're doing.
    pub fn __create_program<'a>(
        &self,
        shaders: impl AsRef<[&'a Shader]>,
        uniforms: HashMap<String, ExpressionType>,
    ) -> Program {
        Program::new(shaders, uniforms)
    }

    /// Creates a buffer object
    pub fn create_buffer<T: Copy>(
        &self,
        buffer_type: BufferType,
        data: BufferData<'_, T>,
    ) -> Buffer<T> {
        Buffer::new(buffer_type, data)
    }

    /*/// Creates a mesh object
    pub fn create_mesh_from_buffers<V: Vertex>(&self, vertex_buffer: &Buffer<V>, element_buffer: &Buffer<u32>, element_count: usize) -> Mesh<V> {
        Mesh::new(vertex_buffer, element_buffer, element_count)
    }

    /// Creates a mesh object
    pub fn create_mesh<V: Vertex>(&self, vertices: &[V], elements: &[u32]) -> Mesh<V> {
        let vertex_buffer = self.create_buffer(BufferType::VERTEX, BufferData::Data(vertices));
        let element_buffer = self.create_buffer(BufferType::ELEMENT, BufferData::Data(elements));
        Mesh::new(&vertex_buffer, &element_buffer, elements.len())
    }*/

    /// Creates a mesh object
    pub fn create_mesh<V: Vertex>(&self, proto_mesh: &ProtoMesh) -> Mesh<V> {
        if proto_mesh.elements().len() == 0 {
            panic!("Cannot create a mesh from an empty ProtoMesh");
        }
        let vertices: Vec<_> = proto_mesh.convert_vertices::<V>().collect();
        let elements = proto_mesh.elements();
        let vertex_buffer = self.create_buffer(BufferType::VERTEX, BufferData::Data(&vertices));
        let element_buffer = self.create_buffer(BufferType::ELEMENT, BufferData::Data(elements));
        Mesh::new(&vertex_buffer, &element_buffer, elements.len(), proto_mesh.primitive_type())
    }

    /// Sets the target framebuffer for rendering operations
    fn target_framebuffer_for_rendering(&self, framebuffer: &Framebuffer) {
        unsafe {
            gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, framebuffer.handle());
        }
    }

    /// Sets the target framebuffer for reading operations
    fn target_framebuffer_for_reading(&self, framebuffer: &Framebuffer) {
        unsafe {
            gl::BindFramebuffer(gl::READ_FRAMEBUFFER, framebuffer.handle());
        }
    }

    /// Clears the color of a framebuffer
    pub fn clear_color(&self, framebuffer: &Framebuffer, color: Color) {
        // Break the color into its components
        let [r, g, b, a] = color.to_components();

        unsafe {
            // Set the clear color
            gl::ClearColor(r, g, b, a);
            // Set the target for clearing
            self.target_framebuffer_for_rendering(framebuffer);
            // Clear the color buffer
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }
    }

    /// Clears the depth of a framebuffer
    pub fn clear_depth(&self, framebuffer: &Framebuffer, depth: Option<f32>) {
        unsafe {
            // Set the depth clear value
            gl::ClearDepth(depth.map(|d| d as f64).unwrap_or(1.0));
            // Set the target for clearing
            self.target_framebuffer_for_rendering(framebuffer);
            // Clear the depth buffer
            gl::Clear(gl::DEPTH_BUFFER_BIT);
        }
    }

    /// Renders a mesh
    pub fn render_mesh<
        'a,
        V: Vertex,
        U: IntoIterator<Item = (Cow<'a, str>, Box<dyn UniformValue>)>,
    >(
        &self,
        framebuffer: &Framebuffer,
        program: &Program,
        mesh: &Mesh<V>,
        instances: usize,
        camera: Option<&RenderCamera>,
        uniforms: U,
    ) {
        unsafe {
            // Validate the program against the current state
            program.validate();
            // Use the shader program
            program.use_for_render();
            // Bind the vertex array object
            gl::BindVertexArray(mesh.handle());
            // Target the framebuffer for rendering
            self.target_framebuffer_for_rendering(framebuffer);
            // Set the camera
            if let Some(camera) = camera {
                program.set_camera(camera);
            }
            // Set uniforms
            for (name, value) in uniforms {
                // Set the uniform value
                program
                    .set_uniform_value_dyn(&name, value)
                    .unwrap_or_else(|_| {
                        panic!(
                            "Uniform variable \"{}\" is not defined in the program",
                            name
                        )
                    });
            }
            // Draw the elements
            gl::DrawElementsInstanced(
                mesh.primitive_type().to_gl_enum(),
                mesh.element_count() as GLsizei,
                gl::UNSIGNED_INT,
                std::ptr::null(),
                instances as GLsizei,
            );
        }
    }
}

impl !Send for Gfx {}
impl !Sync for Gfx {}

#[derive(Debug)]
struct GfxBase;

/// Macro rules for making graphics object structs
macro_rules! gfx_object_struct {
    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident$(<$($gen:tt$(: $bound:tt)?),+>)? {
            $(
                $(#[$field_attr:meta])*
                $field_name:ident: $field_type:ty
            ),*
            $(,)?
        }

        $(
            $(#[$base_attr:meta])*
            base
        )?

        handle($handle_this:ident) -> u32 {
            $($handle_code:tt)*
        }

        drop($drop_this:ident) {
            $($drop_code:tt)*
        }
    ) => {
        paste::paste! {
            #[derive(Debug, Clone)]
            $(#[$attr])*
            $vis struct $name$(<$($gen$(: $bound)?),+>)? {
                base: std::rc::Rc<std::cell::RefCell<[<$name Base>]$(<$($gen),+>)?>>,
            }

            $($(#[$base_attr])*)?
            struct [<$name Base>]$(<$($gen$(: $bound)?),+>)? {
                $(
                    $(#[$field_attr])*
                    $field_name: $field_type,
                )*
            }

            impl$(<$($gen$(: $bound)?),+>)? Drop for [<$name Base>]$(<$($gen),+>)? {
                fn drop(&mut self) {
                    fn drop_code$(<$($gen$(: $bound)?),+>)?($handle_this: &mut [<$name Base>]$(<$($gen),+>)?) {
                        $($drop_code)*
                    }

                    drop_code(self)
                }
            }

            #[allow(dead_code)]
            impl$(<$($gen$(: $bound)?),+>)? $name$(<$($gen),+>)? {
                fn handle(&self) -> u32 {
                    fn handle_code$(<$($gen$(: $bound)?),+>)?($drop_this: &[<$name Base>]$(<$($gen),+>)?) -> u32 {
                        $($handle_code)*
                    }

                    handle_code(&self.base())
                }

                fn base(&self) -> std::cell::Ref<[<$name Base>]$(<$($gen),+>)?> {
                    self.base.borrow()
                }
            }

            impl$(<$($gen$(: $bound)?),+>)? !Send for [<$name Base>]$(<$($gen),+>)? {}
            impl$(<$($gen$(: $bound)?),+>)? !Sync for [<$name Base>]$(<$($gen),+>)? {}
        }
    };
}

// Framebuffer definition
gfx_object_struct! {
    /// A framebuffer object
    pub struct Framebuffer {
        framebuffer: u32,
        has_color: bool,
        has_depth: bool,
    }

    #[derive(Debug)]
    base

    handle(this) -> u32 {
        this.framebuffer
    }

    drop(this) {
        unsafe {
            if this.framebuffer != 0 { // Don't delete the default framebuffer
                gl::DeleteFramebuffers(1, &this.framebuffer as *const _);
            }
        }
    }
}

impl Framebuffer {
    /// Returns whether the framebuffer has a color buffer
    pub fn has_color(&self) -> bool {
        self.base().has_color
    }

    /// Returns whether the framebuffer has a depth buffer
    pub fn has_depth(&self) -> bool {
        self.base().has_depth
    }

    /// Returns the default framebuffer; typically shouldn't be used unless you know what you're
    /// doing because `AppWindow::present_frame` will provide the framebuffer you need anyway
    pub fn __default() -> Self {
        Self {
            base: std::rc::Rc::new(std::cell::RefCell::new(FramebufferBase {
                framebuffer: 0,
                has_color: true,
                has_depth: true,
            })),
        }
    }
}

// Shader definition
gfx_object_struct! {
    /// A shader object
    pub struct Shader {
        shader: u32,
    }

    #[derive(Debug)]
    base

    handle(this) -> u32 {
        this.shader
    }

    drop(this) {
        unsafe {
            gl::DeleteShader(this.shader);
        }
    }
}

impl Shader {
    /// Creates a new shader from a source string; One should typically use a `ProgramBuilder` instead,
    /// as it performs extra checks and provides a more convenient API, as well as enabling
    /// writing shaders in Rust with working code completion in most IDEs.
    fn new(shader_type: ShaderType, source: CString) -> Self {
        let source_bytes = source.as_bytes_with_nul();
        let sources = [source_bytes.as_ptr()];

        let gl_shader = unsafe {
            // Create and compile the shader
            let gl_shader = gl::CreateShader(shader_type.to_enum());
            gl::ShaderSource(
                gl_shader,
                sources.len() as GLsizei,
                sources.as_ptr() as *const *const GLchar,
                std::ptr::null(),
            );
            gl::CompileShader(gl_shader);

            // Check for errors
            #[cfg(debug_assertions)]
            {
                let mut compile_status = 0;
                gl::GetShaderiv(gl_shader, gl::COMPILE_STATUS, &mut compile_status as *mut _);
                if compile_status == 0 {
                    // Some sort of failure happened
                    // Get the log length
                    let mut info_log_length = 0;
                    gl::GetShaderiv(
                        gl_shader,
                        gl::INFO_LOG_LENGTH,
                        &mut info_log_length as *mut _,
                    );
                    // No log?
                    if info_log_length == 0 {
                        panic!("Shader compilation failed with no error log");
                    }
                    // Get log and panic with it
                    let mut info_log = Vec::with_capacity(info_log_length as usize);
                    gl::GetShaderInfoLog(
                        gl_shader,
                        info_log_length,
                        &mut info_log_length as *mut _,
                        info_log.as_mut_ptr() as *mut _,
                    );
                    info_log.set_len(info_log_length as usize);
                    panic!(
                        "Failed to compile shader: {}\nSource:\n{}",
                        String::from_utf8_lossy(&info_log),
                        String::from_utf8_lossy(source_bytes),
                    );
                }
            }

            gl_shader
        };

        Self {
            base: std::rc::Rc::new(std::cell::RefCell::new(ShaderBase { shader: gl_shader })),
        }
    }
}

/// Represents a type of shader
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderType {
    /// A vertex shader
    Vertex,
    /// A fragment shader
    Fragment,
}

impl ShaderType {
    fn to_enum(self) -> GLenum {
        match self {
            ShaderType::Vertex => gl::VERTEX_SHADER,
            ShaderType::Fragment => gl::FRAGMENT_SHADER,
        }
    }
}

// Program definition
gfx_object_struct! {
    /// A program object
    pub struct Program {
        program: u32,
        _shaders: Vec<Shader>,
        uniforms: HashMap<String, ExpressionType>,
    }

    #[derive(Debug)]
    base

    handle(this) -> u32 {
        this.program
    }

    drop(this) {
        unsafe {
            gl::DeleteProgram(this.program);
        }
    }
}

impl Program {
    fn new<'a>(
        shaders: impl AsRef<[&'a Shader]>,
        uniforms: HashMap<String, ExpressionType>,
    ) -> Self {
        let shaders = shaders.as_ref();
        if shaders.len() == 0 {
            panic!("No shaders provided to Program::new");
        }
        let gl_program = unsafe {
            // Create the program
            let gl_program = gl::CreateProgram();
            // Attach all shaders
            for shader in shaders {
                gl::AttachShader(gl_program, shader.handle());
            }
            // Link the program
            gl::LinkProgram(gl_program);
            // Check for errors
            #[cfg(debug_assertions)]
            {
                let mut link_status = 0;
                gl::GetProgramiv(gl_program, gl::LINK_STATUS, &mut link_status as *mut _);
                if link_status == 0 {
                    // Some sort of failure happened
                    // Get the log length
                    let mut info_log_length = 0;
                    gl::GetProgramiv(
                        gl_program,
                        gl::INFO_LOG_LENGTH,
                        &mut info_log_length as *mut _,
                    );
                    // No log?
                    if info_log_length == 0 {
                        panic!("Program linking failed with no error log");
                    }
                    // Get log and panic with it
                    let mut info_log = Vec::with_capacity(info_log_length as usize);
                    gl::GetProgramInfoLog(
                        gl_program,
                        info_log_length,
                        &mut info_log_length as *mut _,
                        info_log.as_mut_ptr() as *mut _,
                    );
                    info_log.set_len(info_log_length as usize);
                    panic!(
                        "Failed to link program: {}",
                        String::from_utf8_lossy(&info_log)
                    );
                }
            }
            gl_program
        };

        Self {
            base: std::rc::Rc::new(std::cell::RefCell::new(ProgramBase {
                program: gl_program,
                _shaders: shaders.as_ref().into_iter().map(|&s| s.clone()).collect(),
                uniforms,
            })),
        }
    }

    /// Run a validation pass on the program; has no effect in release mode
    fn validate(&self) {
        #[cfg(debug_assertions)]
        unsafe {
            gl::ValidateProgram(self.handle());
            // Check for errors
            let mut validate_status = 0;
            gl::GetProgramiv(
                self.handle(),
                gl::VALIDATE_STATUS,
                &mut validate_status as *mut _,
            );
            if validate_status == 0 {
                // Some sort of failure happened
                // Get the log length
                let mut info_log_length = 0;
                gl::GetProgramiv(
                    self.handle(),
                    gl::INFO_LOG_LENGTH,
                    &mut info_log_length as *mut _,
                );
                // No log?
                if info_log_length == 0 {
                    panic!("Program validation failed with no error log");
                }
                // Get log and panic with it
                let mut info_log = Vec::with_capacity(info_log_length as usize);
                gl::GetProgramInfoLog(
                    self.handle(),
                    info_log_length,
                    &mut info_log_length as *mut _,
                    info_log.as_mut_ptr() as *mut _,
                );
                info_log.set_len(info_log_length as usize);
                panic!(
                    "Failed to validate program: {}",
                    String::from_utf8_lossy(&info_log)
                );
            }
        }
    }

    /// Use the program for render commands
    fn use_for_render(&self) {
        self.validate();
        unsafe {
            gl::UseProgram(self.handle());
        }
    }

    /// Get the location of a uniform variable in the program
    fn get_uniform_location(&self, name: &str, ty: ExpressionType) -> Option<GLint> {
        let name_cstring = CString::new(name).expect("Name contained a null character");
        if let Some(uniform_type) = self.base().uniforms.get(name) {
            if *uniform_type == ty {
                let location = unsafe {
                    gl::GetUniformLocation(self.handle(), name_cstring.as_ptr() as *const _)
                };
                if location == -1 {
                    None
                } else {
                    Some(location)
                }
            } else {
                panic!(
                    "Uniform {} has type {:?}, but was requested as {:?}",
                    name, uniform_type, ty
                );
            }
        } else {
            None
        }
    }

    /// Set the value of a uniform variable in the program
    /// The type of the value must match the type of the uniform variable
    /// If the type of the value is not correct, the program will panic
    /// If the uniform variable does not exist, the program will panic
    fn set_uniform_value<V: UniformValue>(&self, name: &str, value: V) -> Result<(), ()> {
        let location = self
            .get_uniform_location(name, value.expression_type())
            .ok_or(())?;
        unsafe {
            value.set_uniform_value(location);
        }
        Ok(())
    }

    /// Set the value of a uniform variable in the program
    /// The type of the value must match the type of the uniform variable
    /// If the type of the value is not correct, the program will panic
    /// If the uniform variable does not exist, the program will panic
    fn set_uniform_value_dyn(&self, name: &str, value: Box<dyn UniformValue>) -> Result<(), ()> {
        let location = self
            .get_uniform_location(name, value.expression_type())
            .ok_or(())?;
        unsafe {
            value.set_uniform_value(location);
        }
        Ok(())
    }

    /// Set the value of a uniform variable in the program
    fn set_camera(&self, camera: &RenderCamera) {
        let _ = self.set_uniform_value("_camera_position", camera.position.convert_to::<f32>().unwrap());
        let _ = self.set_uniform_value("_camera_direction", camera.direction.convert_to::<f32>().unwrap());
        let _ = self.set_uniform_value("_camera_up", camera.up.convert_to::<f32>().unwrap());
        let _ = self.set_uniform_value("_camera_projection", camera.projection);
    }
}

/// A program template, for easily building a specific program.
pub trait ProgramTemplate {
    fn create_program(&self, gfx: &Gfx) -> Program;
}

// Buffer definition
gfx_object_struct! {
    /// A buffer object
    pub struct Buffer<T: Copy> {
        buffer: u32,
        _phantom_data: PhantomData<T>,
    }

    #[derive(Debug)]
    base

    handle(this) -> u32 {
        this.buffer
    }

    drop(this) {
        unsafe {
            gl::DeleteBuffers(1, &this.buffer);
        }
    }
}

impl<T: Copy> Buffer<T> {
    fn new(buffer_type: BufferType, data: BufferData<'_, T>) -> Self {
        if size_of::<T>() == 0 {
            panic!("Cannot create a buffer containing a type with a size of 0");
        }
        let gl_buffer = unsafe {
            // Create the buffer
            let mut gl_buffer = 0;
            gl::CreateBuffers(1, &mut gl_buffer as *mut _);
            // Fill the buffer with the contents of 'data'
            match data {
                BufferData::Uninitialized(capacity) => {
                    if capacity == 0 {
                        panic!("Cannot create a buffer with a capacity of 0");
                    }
                    gl::NamedBufferStorage(
                        gl_buffer,
                        (capacity * size_of::<T>()) as GLsizeiptr,
                        std::ptr::null(),
                        buffer_type.storage_flags(),
                    );
                }
                BufferData::Data(data) => {
                    if data.len() == 0 {
                        panic!("Cannot create a buffer from an empty slice");
                    }
                    gl::NamedBufferStorage(
                        gl_buffer,
                        (data.len() * size_of::<T>()) as GLsizeiptr,
                        data.as_ptr() as *const _,
                        buffer_type.storage_flags(),
                    );
                }
            }
            gl_buffer
        };

        Self {
            base: std::rc::Rc::new(std::cell::RefCell::new(BufferBase {
                buffer: gl_buffer,
                _phantom_data: PhantomData,
            })),
        }
    }
}

bitflags! {
    /// Buffer type flags
    pub struct BufferType: u32 {
        const VERTEX =  0b00000001;
        const ELEMENT = 0b00000010;
        const STORAGE = 0b00000100;
        const TEXTURE = 0b00001000;
        const UNIFORM = 0b00010000;
        const STAGING = 0b00100000;
    }
}

impl BufferType {
    fn storage_flags(&self) -> GLenum {
        if self.contains(BufferType::STAGING) {
            gl::MAP_WRITE_BIT | gl::CLIENT_STORAGE_BIT
        } else {
            gl::MAP_READ_BIT
        }
    }
}

pub enum BufferData<'a, T: Copy> {
    Uninitialized(usize),
    Data(&'a [T]),
}

// Mesh definition
gfx_object_struct! {
    /// A mesh object
    /// Serves as a wrapper around everything needed to render a 3D mesh
    pub struct Mesh<V: Copy> {
        vertex_array: u32,
        _vertex_buffer: Buffer<V>,
        _element_buffer: Buffer<u32>,
        element_count: usize,
        primitive_type: PrimitiveType,
    }

    #[derive(Debug)]
    base

    handle(this) -> u32 {
        this.vertex_array
    }

    drop(this) {
        unsafe {
            gl::DeleteVertexArrays(1, &this.vertex_array);
        }
    }
}

impl<V: Vertex> Mesh<V> {
    fn new(vertex_buffer: &Buffer<V>, element_buffer: &Buffer<u32>, element_count: usize, primitive_type: PrimitiveType) -> Self {
        #[cfg(debug_assertions)]
        primitive_type.check_element_count(element_count);
        
        if element_count == 0 {
            panic!("Cannot create a mesh with 0 elements");
        }
        let vertex_array = unsafe {
            // Create vertex array
            let mut vertex_array = 0;
            gl::CreateVertexArrays(1, &mut vertex_array);
            // Attach the vertex buffer
            gl::VertexArrayVertexBuffer(
                vertex_array,
                0,
                vertex_buffer.handle(),
                0,
                size_of::<V>() as GLsizei,
            );
            // The current vertex from the vertex buffer is incremented per vertex instead of per instance
            gl::VertexArrayBindingDivisor(vertex_array, 0, 0);
            // Set the attributes for the vertex buffer
            for (i, attrib) in V::attributes().iter().enumerate() {
                gl::EnableVertexArrayAttrib(vertex_array, i as u32);
                gl::VertexArrayAttribBinding(vertex_array, i as u32, 0);
                gl::VertexArrayAttribFormat(
                    vertex_array,
                    i as GLuint,
                    attrib.size as GLint,
                    attrib.ty,
                    gl::FALSE,
                    attrib.offset as GLuint,
                );
            }
            // Attach the element buffer
            gl::VertexArrayElementBuffer(vertex_array, element_buffer.handle());
            vertex_array
        };

        Self {
            base: std::rc::Rc::new(std::cell::RefCell::new(MeshBase {
                vertex_array,
                _vertex_buffer: vertex_buffer.clone(),
                _element_buffer: element_buffer.clone(),
                element_count,
                primitive_type,
            })),
        }
    }

    pub fn element_count(&self) -> usize {
        self.base().element_count
    }

    pub fn primitive_type(&self) -> PrimitiveType {
        self.base().primitive_type
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum PrimitiveType {
    Triangles,
    Lines,
    Points,
}

impl PrimitiveType {
    pub fn check_element_count(self, element_count: usize) {
        match self {
            PrimitiveType::Triangles => {
                if element_count % 3 != 0 {
                    panic!("Cannot create a triangle mesh with {} elements", element_count);
                }
            }
            PrimitiveType::Lines => {
                if element_count % 2 != 0 {
                    panic!("Cannot create a point mesh with {} elements", element_count);
                }
            }
            PrimitiveType::Points => {}
        }
    }

    pub fn to_gl_enum(self) -> GLenum {
        match self {
            PrimitiveType::Triangles => gl::TRIANGLES,
            PrimitiveType::Lines => gl::LINES,
            PrimitiveType::Points => gl::POINTS,
        }
    }
}

pub trait UniformValue {
    unsafe fn set_uniform_value(&self, location: GLint);
    fn expression_type(&self) -> ExpressionType;
}

impl UniformValue for f32 {
    unsafe fn set_uniform_value(&self, location: GLint) {
        gl::Uniform1f(location, *self);
    }

    fn expression_type(&self) -> ExpressionType {
        ExpressionType::Scalar(Scalar::F32)
    }
}

impl UniformValue for Vector2<f32> {
    unsafe fn set_uniform_value(&self, location: GLint) {
        gl::Uniform2f(location, self.x(), self.y());
    }

    fn expression_type(&self) -> ExpressionType {
        ExpressionType::Vector2(Scalar::F32)
    }
}

impl UniformValue for Vector3<f32> {
    unsafe fn set_uniform_value(&self, location: GLint) {
        gl::Uniform3f(location, self.x(), self.y(), self.z());
    }

    fn expression_type(&self) -> ExpressionType {
        ExpressionType::Vector3(Scalar::F32)
    }
}

impl UniformValue for Vector4<f32> {
    unsafe fn set_uniform_value(&self, location: GLint) {
        gl::Uniform4f(location, self.x(), self.y(), self.z(), self.w());
    }

    fn expression_type(&self) -> ExpressionType {
        ExpressionType::Vector4(Scalar::F32)
    }
}

impl UniformValue for i32 {
    unsafe fn set_uniform_value(&self, location: GLint) {
        gl::Uniform1i(location, *self);
    }

    fn expression_type(&self) -> ExpressionType {
        ExpressionType::Scalar(Scalar::I32)
    }
}

impl UniformValue for Vector2<i32> {
    unsafe fn set_uniform_value(&self, location: GLint) {
        gl::Uniform2i(location, self.x(), self.y());
    }

    fn expression_type(&self) -> ExpressionType {
        ExpressionType::Vector2(Scalar::I32)
    }
}

impl UniformValue for Vector3<i32> {
    unsafe fn set_uniform_value(&self, location: GLint) {
        gl::Uniform3i(location, self.x(), self.y(), self.z());
    }

    fn expression_type(&self) -> ExpressionType {
        ExpressionType::Vector3(Scalar::I32)
    }
}

impl UniformValue for Vector4<i32> {
    unsafe fn set_uniform_value(&self, location: GLint) {
        gl::Uniform4i(location, self.x(), self.y(), self.z(), self.w());
    }

    fn expression_type(&self) -> ExpressionType {
        ExpressionType::Vector4(Scalar::I32)
    }
}

impl UniformValue for Matrix4x4<f32> {
    unsafe fn set_uniform_value(&self, location: GLint) {
        gl::UniformMatrix4fv(location, 1, gl::FALSE, self.as_ptr());
    }

    fn expression_type(&self) -> ExpressionType {
        ExpressionType::Matrix4x4(Scalar::F32)
    }
}

/// Represents a camera for rendering
pub struct RenderCamera {
    position: Vector3<f64>,
    direction: Vector3<f64>,
    up: Vector3<f64>,
    projection: Matrix4x4<f32>,
}

impl RenderCamera {
    pub fn new(
        position: Vector3<f64>,
        direction: Vector3<f64>,
        up: Vector3<f64>,
        projection: Matrix4x4<f32>,
    ) -> Self {
        let direction = direction.normalized();
        let up = up.normalized();

        #[cfg(debug_assertions)]
        {
            if direction.dot(&up).abs() > 0.99999 {
                panic!("Forward and up must not be parallel or too close to parallel");
            }
        }

        Self {
            position,
            direction,
            up,
            projection,
        }
    }
}

#[macro_export]
macro_rules! program_uniforms {
    (
        $($name:ident: $type:ty),*
        $(,)?
    ) => {
        [
            $(
                (
                    stringify!($name).to_string(),
                    $crate::vertex::__VertexAttributeTypes::<$type>::EXPRESSION_TYPE,
                ),
            )*
        ]
    };
}

#[macro_export]
macro_rules! render_uniforms {
    (
        $($name:ident: $value:expr),*
        $(,)?
    ) => {
        [
            $(
                (
                    std::borrow::Cow::Borrowed(stringify!($name)),
                    std::convert::identity::<Box<dyn $crate::gfx::UniformValue>>(Box::new($value)),
                ),
            )*
        ]
    };
}
