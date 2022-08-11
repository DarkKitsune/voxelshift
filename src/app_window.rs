use std::{
    ffi::c_void,
    sync::{Arc, RwLock},
    thread::ThreadId,
};

use ggmath::prelude::*;
use glfw::{
    ClientApiHint, Context, Glfw, OpenGlProfileHint, SwapInterval, Window, WindowEvent, WindowHint, Key, Modifiers, Action, CursorMode,
};
use tokio::sync::mpsc::{channel, Receiver, Sender};

use crate::gfx::{Framebuffer, Gfx};

pub struct AppWindowBuilder {
    title: String,
    size: Vector2<u32>,
}

impl AppWindowBuilder {
    // Set the window title
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    // Set the window size
    pub fn with_size(mut self, size: Vector2<u32>) -> Self {
        self.size = size.into();
        self
    }

    /// Creates a new app window.
    pub fn build(self, mut glfw: Glfw) -> (AppWindow, Sender<AppWindowEvent>) {
        // Set window hint
        glfw.window_hint(WindowHint::Resizable(false));
        glfw.window_hint(WindowHint::ClientApi(ClientApiHint::OpenGl));
        glfw.window_hint(WindowHint::ContextVersion(4, 5));
        glfw.window_hint(WindowHint::OpenGlProfile(OpenGlProfileHint::Core));
        glfw.window_hint(WindowHint::OpenGlForwardCompat(true));

        // Create window
        let (mut window, glfw_window_receiver) = glfw
            .create_window(
                self.size.x(),
                self.size.y(),
                &self.title,
                glfw::WindowMode::Windowed,
            )
            .expect("Failed to create GLFW window");
        window.make_current();
        window.set_close_polling(true);
        window.set_key_polling(true);
        window.set_cursor_mode(CursorMode::Disabled);
        window.set_raw_mouse_motion(true);
        window.set_cursor_pos_polling(true);
        glfw.set_swap_interval(SwapInterval::Sync(1));

        let (app_window_sender, app_window_receiver) = channel(16);

        (
            AppWindow {
                parent_thread: std::thread::current().id(),
                size: self.size,
                gfx: None,
                base: Arc::new(RwLock::new(AppWindowBase {
                    should_exit: false,
                    glfw,
                    window,
                    glfw_window_receiver: Some(glfw_window_receiver),
                    app_window_receiver,
                })),
            },
            app_window_sender,
        )
    }
}

impl Default for AppWindowBuilder {
    fn default() -> Self {
        AppWindowBuilder {
            title: "Untitled App".to_string(),
            size: vector!(1280, 720),
        }
    }
}

/// The app's main window.
#[derive(Debug, Clone)]
pub struct AppWindow {
    parent_thread: ThreadId,
    size: Vector2<u32>,
    gfx: Option<Gfx>,
    base: Arc<RwLock<AppWindowBase>>,
}

impl AppWindow {
    pub fn builder() -> AppWindowBuilder {
        AppWindowBuilder::default()
    }

    pub fn update(&self) -> Vec<AppWindowUpdateEvent> {
        if self.parent_thread != std::thread::current().id() {
            panic!("AppWindow::update() should be called from the same thread the window was created in");
        }
        let mut base = self.base.write().unwrap();
        if base.should_exit {
            panic!("AppWindow exited, cannot update further");
        } else {
            // Update graphics controller
            self.gfx
                .as_ref()
                .expect("Window does not have \"gfx\" set")
                .__update();
            // Handle GLFW events
            let mut update_events = Vec::new();
            base.glfw.poll_events();
            let glfw_window_receiver = base.glfw_window_receiver.take().unwrap();
            for (_, event) in glfw::flush_messages(&glfw_window_receiver) {
                match event {
                    glfw::WindowEvent::Close => {
                        base.should_exit = true;
                        println!("Window close was requested");
                        update_events.push(AppWindowUpdateEvent::Closed);
                    },
                    glfw::WindowEvent::Key(key, _, action, modifiers) => update_events.push(
                        AppWindowUpdateEvent::KeyAction {
                            key,
                            action,
                            modifiers,
                        },
                    ),
                    glfw::WindowEvent::CursorPos(x, y) => {
                        update_events.push(AppWindowUpdateEvent::MouseMove { position: vector!(x as f32, y as f32) });
                    }
                    _ => {}
                }
            }
            base.glfw_window_receiver.replace(glfw_window_receiver);
            // Handle AppWindow events
            while let Ok(event) = base.app_window_receiver.try_recv() {
                match event {}
            }
            // Return update events
            update_events
        }
    }

    pub fn present_frame<R, F: FnOnce(&Framebuffer) -> R>(&self, f: F) -> R {
        if self.parent_thread != std::thread::current().id() {
            panic!("AppWindow::present_frame() should be called from the same thread the window was created in");
        }
        let mut base = self.base.write().unwrap();
        if base.should_exit {
            panic!("AppWindow exited, cannot present frame");
        }
        let ret = f(&Framebuffer::__default());
        base.window.swap_buffers();
        ret
    }

    pub fn set_gfx(&mut self, gfx: &Gfx) {
        self.gfx.replace(gfx.clone());
    }

    pub fn should_exit(&self) -> bool {
        let base = self.base.read().unwrap();
        base.should_exit
    }

    pub fn close(self) {
        if self.parent_thread != std::thread::current().id() {
            panic!("AppWindow::close() should be called from the same thread the window was created in");
        }
        Arc::try_unwrap(self.base)
            .expect("Can't close the window when clones of the same window still exist")
            .into_inner()
            .unwrap()
            .window
            .close();
    }

    pub fn __get_proc_address(&self, name: &str) -> *const c_void {
        let mut base = self.base.write().unwrap();
        base.window.get_proc_address(name)
    }

    pub fn size(&self) -> Vector2<u32> {
        self.size
    }
}

unsafe impl Send for AppWindow {}
unsafe impl Sync for AppWindow {}

#[derive(Debug)]
pub struct AppWindowBase {
    should_exit: bool,
    glfw: Glfw,
    window: Window,
    glfw_window_receiver: Option<std::sync::mpsc::Receiver<(f64, WindowEvent)>>,
    app_window_receiver: Receiver<AppWindowEvent>,
}

/// We need to use an event channel (look at mpsc or one shot in tokio crate) to communicate between threads so this type can be send + sync
pub enum AppWindowEvent {}

/// An event that occured during AppWindow update
pub enum AppWindowUpdateEvent {
    /// The window was closed
    Closed,
    /// A key was pressed or released
    KeyAction {
        /// The key that was pressed
        key: Key,
        /// The action that was performed
        action: Action,
        /// The modifiers that were pressed
        modifiers: Modifiers,
    },
    /// The mouse was moved
    MouseMove {
        /// The change in position of the mouse
        position: Vector2<f32>,
    },
}