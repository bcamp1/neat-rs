use crate::network::*;
use rand::Rng;

// Helper Functions
pub fn random() -> f32 {
    let f: f32 = rand::thread_rng().gen();
    return f;
}

pub fn weighted_bool(true_chance: f32) -> bool {
    if random() < true_chance {
        return true;
    }
    false
}

pub fn random_bool() -> bool {
    weighted_bool(0.5)
}

pub fn evaluate_xor(mut n: Network) -> f32 {
    let mut error = 0f32;

    let actual: Vec<f32> = vec!(
        n.evaluate(vec!(0.0, 0.0, 1.0))[0],
        n.evaluate(vec!(1.0, 0.0, 1.0))[0],
        n.evaluate(vec!(0.0, 1.0, 1.0))[0],
        n.evaluate(vec!(1.0, 1.0, 1.0))[0],
    );

    let expected: Vec<f32> = vec!(
        0.0,
        1.0,
        1.0,
        0.0,
    );

    for i in 0..expected.len() {
        error += (1.0/4.0) * (expected[i] - actual[i]).powf(2.0);
    }
    return 1.0 - error;
}

pub struct NEAT {
    pub generation: u32,
    pub pop: Vec<Network>,
    pub pop_size: u32,

    // Mutations
    pub change_weights_chance: f32,
    pub perturb_weights_chance: f32,
    pub perturb_amount: f32,
    pub add_node_chance: f32,
    pub add_connection_chance: f32,

    // Measuring Coefficients
    pub c1: f32,
    pub c2: f32,
    pub c3: f32,
    pub distance_threshold: f32,

    pub preserve_champion_threshhold: u32,
}

impl NEAT {
    pub fn new(num_inputs: u32, num_outputs: u32) -> Self {
        let pop_size = 150;
        let mut pop: Vec<Network> = Vec::new();
        for _ in 0..pop_size {
            let mut n = Network::new(num_inputs, num_outputs);
            n.mutate_add_connection();
            n.mutate_add_connection();
            pop.push(n.clone());
        }

        NEAT {
            generation: 0,
            pop,
            pop_size,

            // Mutations
            change_weights_chance: 0.8,
            perturb_weights_chance: 0.9,
            perturb_amount: 0.1,
            add_node_chance: 0.03,
            add_connection_chance: 0.05,

            // Measuring Coefficients
            c1: 1.0,
            c2: 1.0,
            c3: 0.4,
            distance_threshold: 3.0,

            preserve_champion_threshhold: 5,
        }
    }

    

    pub fn train(&mut self) {
        // Evaluate all networks
        for i in 0..self.pop.len() {
            self.pop[i].fitness = evaluate_xor(self.pop[i].clone());
        } 

        // Sort by Fitness
        self.pop.sort_by(|a, b| a.partial_cmp(b).unwrap());
      
        // Copy top 50
        let top_networks = 50;
        for i in 0..top_networks {
            self.pop[(self.pop_size - top_networks + i) as usize] = self.pop[i as usize].clone();            
        }

        // Mutations
        for network in &mut self.pop {
            // Change weights
            if weighted_bool(self.change_weights_chance) {
                if weighted_bool(self.perturb_weights_chance) {
                    for i in 0..network.connections.len() {
                        let amount = random() * self.perturb_amount - (0.5 * self.perturb_amount);
                        network.connections[i].weight += amount;
                        if network.connections[i].weight < -1.0 {
                            network.connections[i].weight = -1.0;
                        }

                        if network.connections[i].weight > 1.0 {
                            network.connections[i].weight = 1.0;
                        }
                    }
                } else {
                    // Randomize weights
                    for i in 0..network.connections.len() {
                        network.connections[i].weight = random() * 2.0 - 1.0;
                    }
                }
            }

            // Add node/connection
            if weighted_bool(self.add_node_chance) {
                network.mutate_add_node();
            }

            if weighted_bool(self.add_connection_chance) {
                network.mutate_add_connection();
            }
        }

        self.generation += 1;

    }
}