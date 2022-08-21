#![allow(incomplete_features)]
#![feature(negative_impls)]
#![feature(const_fn_floating_point_arithmetic)]
#![feature(inline_const_pat)]
#![feature(const_type_id)]
#![feature(step_trait)]

use ggmath::vector;

pub mod app;
pub mod app_window;
pub mod colors;
pub mod components;
pub mod game;
pub mod gfx;
pub mod program_builder;
pub mod proto_mesh;
pub mod scene;
pub mod vertex;

// Use the module of choice as game_scene below to run the game with different scenes.
use game::game_scene;

#[tokio::main]
async fn main() {
    use crate::app::*;

    let scene = game_scene::build_scene();

    App::new()
        .with_title("Voxelshift")
        .with_size(vector!(1800, 1000))
        .with_scene(scene)
        .run();
}

#[cfg(test)]
pub mod tests {
    #[test]
    fn it_works() {
        println!("{}", env!("CARGO_PKG_VERSION"))
    }
}
