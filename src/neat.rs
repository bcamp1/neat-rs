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

#[derive(Debug, Clone)]
pub struct NEAT {
    // Global Inno Number
    pub global_inno_number: u32,

    // Generation
    pub generation: u32,
    pub pop_size: u32,

    // Population
    pub pop: Vec<Network>,
    pub species_list: Vec<Vec<usize>>,

    pub past_pop: Vec<Network>,
    pub past_species_list: Vec<Vec<usize>>,

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
    pub fn new(pop_size: u32, num_inputs: u32, num_outputs: u32) -> Self {
        let mut pop: Vec<Network> = Vec::new();


        // Initialize population
        for _ in 0..pop_size {
            let mut inno_number: u32 = 0;
            let mut n = Network::new(num_inputs, num_outputs);
            //n.add_link(&mut inno_number, 0, num_inputs as usize + 1, 0.0);
            //n.mutate_add_connection();
            //n.connections[0].weight = 0.0;
            pop.push(n.clone());
        }

        NEAT {
            global_inno_number: 1,

            generation: 0,
            pop_size,

            pop,
            past_pop: Vec::new(),

            species_list: vec!((0..pop_size as usize).collect()),
            past_species_list: Vec::new(),

            // Mutations
            change_weights_chance: 0.8,
            perturb_weights_chance: 0.9,
            perturb_amount: 0.5,
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

    pub fn get_distance(a: &Network, b: &Network) -> f32 {
        return 0.0;
    }

    pub fn train(&mut self) {
        // Update Past Population
        self.past_pop = self.pop.clone();
        self.past_species_list = self.species_list.clone();

        // Evaluate all networks
        for i in 0..self.pop.len() {
            self.pop[i].fitness = evaluate_xor(self.pop[i].clone());
        } 

        // Sort by Fitness
        self.pop.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // ----- Speciate -----
        self.species_list = Vec::new();
        
        // Select species reps
        let mut species_reps: Vec<usize> = Vec::new();
        for i in 0..self.past_species_list.len() {
            let rep_index: usize = rand::thread_rng().gen_range(0, self.past_species_list[i].len());
            species_reps.push(self.past_species_list[i][rep_index])
        }

        let elite = 20;
        for i in 0..elite {
            self.pop[self.pop_size as usize - elite + i] = self.pop[i].clone();
        }

        // Mutations
        for network in &mut self.pop {
            // Change weights
            if weighted_bool(self.change_weights_chance) {
                if weighted_bool(self.perturb_weights_chance) {
                    for i in 0..network.links.len() {
                        let amount = random() * self.perturb_amount - (0.5 * self.perturb_amount);
                        network.links[i].weight += amount;
                    }
                } else {
                    // Randomize weights
                    for i in 0..network.links.len() {
                        network.links[i].weight = random() * 10.0 - 5.0;
                    }
                }
            }

            // Add node/connection
            if weighted_bool(self.add_node_chance) {
                network.add_random_node(&mut self.global_inno_number);
            }

            if weighted_bool(self.add_connection_chance) {
                network.add_random_link(&mut self.global_inno_number).expect("No nodes available");
            }
        }

        self.generation += 1;

    }
}