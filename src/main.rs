extern crate sdl2;
extern crate rand;

mod network;
mod neat;

use network::*;
use neat::*;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::keyboard::KeyboardState;
use sdl2::keyboard::Scancode;
use sdl2::mouse::{MouseState};
use sdl2::pixels::Color;
use rand::Rng;

fn main() {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let window_size = (1280, 1280);

    let window = video_subsystem.window(format!("NEAT").as_str(), window_size.0, window_size.1)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas()
        .accelerated()
        .build()
        .unwrap();

    let mut event_pump = sdl_context.event_pump().unwrap();

    let mut neat = NEAT::new(50, 3, 1);

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} |
                Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                   break 'running; 
                },
                _ => {}
            }
        }

        neat.train();

        let top = &neat.pop[0];
        println!("{}", top.fitness);

        canvas.set_draw_color(Color::from((0, 0, 0)));
        canvas.clear();
        top.draw(&mut canvas, 50.0, 50.0, 1280.0, 500.0);
        canvas.present();
    }

}
