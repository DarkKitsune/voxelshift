use std::time::Duration;

use ggmath::{vector, vector_alias::Vector2};
use tokio::time::Instant;

use crate::{
    app_window::{AppWindow, AppWindowUpdateEvent},
    gfx::Gfx,
    scene::Scene,
};

pub struct App {
    title: String,
    size: Vector2<u32>,
    initial_scene: Option<Scene>,
}

impl App {
    /// Create a new app configuration with default settings.
    pub fn new() -> Self {
        Self {
            title: "Untitled App".to_string(),
            size: vector!(1280, 720),
            initial_scene: None,
        }
    }

    /// Set the title of the app.
    pub fn with_title(mut self, title: &str) -> Self {
        self.title = title.to_string();
        self
    }

    /// Set the size of the app (client/window size).
    pub fn with_size(mut self, size: Vector2<u32>) -> Self {
        self.size = size;
        self
    }

    /// Set the initial scene of the app.
    pub fn with_scene(mut self, scene: Scene) -> Self {
        self.initial_scene = Some(scene);
        self
    }

    /// Run the app and block until it's exited.
    pub fn run(self) {
        let Self {
            title,
            size,
            initial_scene,
        } = self;

        // No scene?
        let mut scene =
            initial_scene.expect("No scene specified when creating the App (use App::with_scene)");

        // Create GLFW context
        let glfw = glfw::init(glfw::FAIL_ON_ERRORS).expect("Failed to initialize GLFW");

        // Create window
        let (mut window, _window_sender) = AppWindow::builder()
            .with_title(title)
            .with_size(size)
            .build(glfw);

        // Create graphics controller
        let gfx = Gfx::new(&mut window);

        // Initialize the scene
        scene.__init(&gfx);

        let mut last_frame = None;

        // Main loop
        while !window.should_exit() {
            // Update window
            let window_update_events = window.update();

            for update_event in window_update_events {
                match update_event {
                    AppWindowUpdateEvent::KeyAction {
                        key,
                        action,
                        modifiers,
                    } => {
                        scene.__key_action(key, action, modifiers);
                    }
                    AppWindowUpdateEvent::Closed => {}
                    AppWindowUpdateEvent::MouseMove { position } => {
                        scene.__mouse_move(position);
                    }
                }
            }

            let this_frame = Instant::now();
            let frame_delta = if let Some(last_frame) = last_frame {
                this_frame.duration_since(last_frame)
            } else {
                Duration::ZERO
            };

            if !window.should_exit() {
                // Update scene
                scene.__update(frame_delta);

                // Render
                window.present_frame(|framebuffer| {
                    // Clear the framebuffer's depth buffer
                    gfx.clear_depth(framebuffer, None);

                    // Render the scene
                    scene.__render(&gfx, framebuffer, window.size(), frame_delta);
                });
            }

            last_frame = Some(this_frame);
        }

        // Cleanup
        drop(gfx);

        // Close the window
        window.close();
    }
}
