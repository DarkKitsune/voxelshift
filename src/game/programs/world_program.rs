use crate::game::{voxel::VoxelVertex, *};

#[derive(Clone, Copy)]
pub struct WorldProgram;

impl ProgramTemplate for WorldProgram {
    fn create_program(&self, gfx: &Gfx) -> Program {
        let builder = ProgramBuilder::<VoxelVertex>::new()
            .with_uniforms(program_uniforms! {
                mesh_position: Vector3<f32>,
                mesh_scale: Vector3<f32>,
                screen_largest_dimension: f32,
                voxel_meters: f32,
            })
            .with_camera()
            .with_vertex_module(vertex_main)
            .with_fragment_module(fragment_main);
        println!(
            "\n############################\nVertex code:\n\n{}\n\n############################\n",
            builder.vertex_code()
        );
        builder.build(gfx)
    }
}

// ============================================================================
//     Vertex shader
// ============================================================================
fn vertex_main(inputs: &ModuleInputs, outputs: &mut ModuleOutputs, uniforms: &mut ModuleUniforms) {
    // Get the vertex attributes
    let position = inputs.get("position").expect("No input named position");
    let color = inputs.get("color").expect("No input named color");
    let normal = inputs.get("normal").expect("No input named normal");

    // Get the camera view and projection matrices
    let view = uniforms.camera_view();
    let projection = uniforms.camera_projection();

    // Calculate the world space position of the vertex
    let world_space_position =
        position * uniforms.get("mesh_scale") + uniforms.get("mesh_position");

    // Calculate the screen space position of the vertex
    let mut screen_space_position = projection * view * world_space_position.concat(1.0);

    outputs.set_vertex(
        // The final vertex position is the screen space position
        screen_space_position.clone(),
        // Point size is screen size * voxel size in meters * 0.7 * vertex W component
        uniforms.get("screen_largest_dimension") * uniforms.get("voxel_meters") * 0.7
            / screen_space_position.w(),
    );

    // Pass outputs to the fragment shader
    outputs.set(Module::Fragment, "color", color);
    outputs.set(Module::Fragment, "normal", normal);
}

// ============================================================================
//     Fragment shader
// ============================================================================
fn fragment_main(
    inputs: &ModuleInputs,
    outputs: &mut ModuleOutputs,
    _uniforms: &mut ModuleUniforms,
) {
    // Get inputs from the vertex shader
    let color = inputs.get("color").expect("No input named color");
    let normal = inputs.get("normal").expect("No input named normal");

    // Calculate the lighting
    let lighting = normal.dot(vector!(1.0, 1.0, 0.0).normalized()) * 0.5 + 0.5;

    // The color of the fragment should be vertex color * lighting
    let fragment_color = (color.xyz() * lighting).concat(1.0);

    // Output the final color of the fragment
    outputs.set_fragment_color(fragment_color);
}
