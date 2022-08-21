use crate::{game::*, vertex::WorldVertex};

#[derive(Clone, Copy)]
pub struct WorldProgram;

impl ProgramTemplate for WorldProgram {
    fn create_program(&self, gfx: &Gfx) -> Program {
        let builder = ProgramBuilder::<WorldVertex>::new()
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

    // Get the camera view and projection matrices
    let view = uniforms.camera_view();
    let projection = uniforms.camera_projection();

    // Calculate the world space position of the vertex
    let world_space_position = position * uniforms.get("mesh_scale") + uniforms.get("mesh_position");

    // Calculate the screen space position of the vertex
    let mut screen_space_position = projection * view * world_space_position.concat(1.0);

    // The final vertex position is the screen space position
    outputs.set_vertex(screen_space_position.clone(), uniforms.get("screen_largest_dimension") * uniforms.get("voxel_meters") * 0.5 / screen_space_position.w());

    // Pass the color of the vertex to the fragment shader
    outputs.set(Module::Fragment, "color", color);
}

// ============================================================================
//     Fragment shader
// ============================================================================
fn fragment_main(
    inputs: &ModuleInputs,
    outputs: &mut ModuleOutputs,
    _uniforms: &mut ModuleUniforms,
) {
    // Get the color of the vertex
    let color = inputs.get("color").expect("No input named color");

    // The color of the fragment should be the color of the vertex
    let fragment_color = color;

    // Output the final color of the fragment
    outputs.set_fragment_color(fragment_color);
}
