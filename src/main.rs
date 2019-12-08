extern crate sdl2;
extern crate rand;

mod network;
mod neat;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::keyboard::KeyboardState;
use sdl2::keyboard::Scancode;
use sdl2::mouse::{MouseState};
use sdl2::pixels::Color;
use network::*;
use neat::*;
use rand::Rng;



fn print_eval_info(mut n: Network) {
    println!("----------");
    println!("0 0 -> {}", n.evaluate(vec!(0.0, 0.0, 1.0))[0]);
    println!("1 0 -> {}", n.evaluate(vec!(1.0, 0.0, 1.0))[0]);
    println!("0 1 -> {}", n.evaluate(vec!(0.0, 1.0, 1.0))[0]);
    println!("1 1 -> {}", n.evaluate(vec!(1.0, 1.0, 1.0))[0]);
    println!("Fitness: {}", n.fitness);
}

fn main() {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let window_size = (1280, 720);

    let window = video_subsystem.window(format!("NEAT").as_str(), window_size.0, window_size.1)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas()
        .accelerated()
        .build()
        .unwrap();

    let mut event_pump = sdl_context.event_pump().unwrap();

    let mut neat = NEAT::new(3, 1);
    let iterations = 10000;

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

        if neat.generation < iterations {
            neat.train();
        }

        let mut top = neat.pop[0].clone();
        print_eval_info(top.clone());

        canvas.set_draw_color(Color::from((0, 0, 0)));
        canvas.clear();
        top.draw(&mut canvas, 100.0, 100.0);
        canvas.present();
    }

}
