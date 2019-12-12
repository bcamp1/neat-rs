use std::fmt;
use std::fmt::{Formatter, Display};
use std::result::Result;
use rand::Rng;

use sdl2::render::WindowCanvas;
use sdl2::gfx::primitives::DrawRenderer;

use std::cmp::*;

#[derive(Debug, Clone, Copy)]
pub enum NodeType {
    Input,
    Output,
    Hidden,
}

impl PartialEq for NodeType {
    fn eq(&self, other: &Self) -> bool {
        match self {
            NodeType::Input => {
                match other {
                    NodeType::Input => true,
                    _ => false,
                }
            },

            NodeType::Output => {
                match other {
                    NodeType::Output => true,
                    _ => false,
                }
            },

            NodeType::Hidden => {
                match other {
                    NodeType::Hidden => true,
                    _ => false,
                }
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Node {
    pub node_type: NodeType,
    pub level: u32,
    value: f32,
    graphics_x: f32,
    graphics_y: f32,
}

impl Node {
    fn new(node_type: NodeType) -> Node {
        let level = match node_type {
            NodeType::Hidden => 1,
            _ => 0,
        };
        
        Node {
            node_type,
            level,
            value: 0.0,
            graphics_x: 0.0,
            graphics_y: 0.0,   
        }
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let node_str = match self.node_type {
            NodeType::Input => "Input",
            NodeType::Hidden => "Hidden",
            NodeType::Output => "Output",
        };

        write!(f, "----- Node -----\nType: {}\nLevel: {}\n----------------", node_str, self.level)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Link {
    pub inno_number: u32,
    pub in_index: usize,
    pub out_index: usize,
    pub weight: f32,
    pub enabled: bool,
}

impl Link {
    pub fn new(inno_number: u32, in_index: usize, out_index: usize, weight: f32) -> Link {
        Link {
            inno_number,
            in_index,
            out_index,
            weight,
            enabled: true,
        }
    }
}

impl Display for Link {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let disabled_str = if self.enabled {
            ""
        } else {
            "DISABLED\n"
        };

        write!(f, "----- Link -----\nInno {}\n{} --> {}\nWeight: {}\n{}----------------", self.inno_number, self.in_index, self.out_index, self.weight, disabled_str)
    }
}

#[derive(Debug, Clone)]
pub struct Network {
    pub nodes: Vec<Node>,
    pub links: Vec<Link>,
    pub input_count: u32,
    pub output_count: u32,
    pub fitness: f32,
}

impl PartialEq for Network {
    fn eq(&self, other: &Self) -> bool {
        self.fitness == other.fitness
    }
}

impl PartialOrd for Network {
    fn partial_cmp(&self, other: &Network) -> Option<Ordering> {
        if self.fitness > other.fitness {
            Some(Ordering::Less)
        } else if self.fitness < other.fitness {
            Some(Ordering::Greater)
        } else {
            Some(Ordering::Equal)
        }
    }
}

impl Display for Network {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut output = std::string::String::new();
        output += "---------- Network ----------\n";
        for node in &self.nodes {
            output += format!("{}\n", node).as_str();
        }

        for link in &self.links {
            output += format!("{}\n", link).as_str();
        }

        output += "-----------------------------\n";

        write!(f, "{}", output)
    }
}

impl Network {
    pub fn new(inputs: u32, outputs: u32) -> Network {
        let mut nodes: Vec<Node> = Vec::new();
        for _ in 0..inputs {
            nodes.push(Node::new(NodeType::Input));
        }
        for _ in 0..outputs {
            nodes.push(Node::new(NodeType::Output));
        }

        Network {
            nodes,
            links: vec!(),
            input_count: inputs,
            output_count: outputs,
            fitness: 0.0,
        }
    }

    pub fn sigmoid(x: f32) -> f32 {
        (2.0 / (1.0 + (-x).exp())) - 1.0
    }

    pub fn add_link(&mut self, global_inno_number: &mut u32, node_index_1: usize, node_index_2: usize, weight: f32) -> Result<(), &'static str> {
        let node1 = self.nodes[node_index_1].clone();
        let node2 = self.nodes[node_index_2].clone();

        if node1.node_type == node2.node_type && node1.level == node2.level {
            return Err("Node types were equal and z indexes were equal");
        }
        
        let mut in_index: usize = 0;
        let mut out_index: usize = 0;

        match node1.node_type {
            NodeType::Input => {
                in_index = node_index_1;
                out_index = node_index_2;
            },
            NodeType::Output => {
                in_index = node_index_2;
                out_index = node_index_1;
            },
            NodeType::Hidden => {
                if node1.level > node2.level {
                    in_index = node_index_2;
                    out_index = node_index_1; 
                } else if node2.level > node1.level {
                    in_index = node_index_1;
                    out_index = node_index_2; 
                } else {
                    return Err("Node levels are the same");
                }
            },
        }

        self.links.push(Link::new(global_inno_number.clone(), in_index, out_index, weight));
        *global_inno_number += 1;
        Ok(())
    }

    pub fn add_node(&mut self, global_inno_number: &mut u32, link_index: usize) -> Result<(), &'static str> {
        let link = self.links[link_index].clone();
        let in_node = self.nodes[link.in_index].clone();
        let out_node = self.nodes[link.out_index].clone();

        let level = in_node.level + 1;
        if out_node.level == level {
            self.nodes[link.out_index].level += 1;
        }

        let mut new_node = Node::new(NodeType::Hidden);
        new_node.level = level;
        self.nodes.push(new_node);
        let new_index = self.nodes.len() - 1;
        
        self.links[link_index].enabled = false;
        
        self.links.push(Link::new(*global_inno_number, link.in_index, new_index, 1.0));
        self.links.push(Link::new(*global_inno_number + 1, new_index, link.out_index, link.weight));

        *global_inno_number += 2;
        Ok(())
    }

    pub fn add_random_link(&mut self, global_inno_number: &mut u32) -> Result<(), &'static str> {
        let node1_index = rand::thread_rng().gen_range(0, self.nodes.len());
        let mut node2_selection: Vec<usize> = Vec::new();

        for i in 0..self.nodes.len() {
            let node1 = self.nodes[node1_index].clone();
            let node2 = self.nodes[i].clone();
            if node1_index != i {
                if (node1.node_type != node2.node_type) || (node1.level != node2.level) {
                    let mut link_exists = false;
                    for link in &self.links {
                        if (link.in_index == node1_index && link.out_index == i) || (link.in_index == i && link.out_index == node1_index) {
                            link_exists = true;
                        }
                    }
                    if true {
                        node2_selection.push(i);
                    }
                }
            }                        
        }

        if node2_selection.len() == 0 {
            return Err("No nodes available");
        }

        let node2_index = node2_selection[rand::thread_rng().gen_range(0, node2_selection.len())];
        let r: f32 = rand::thread_rng().gen();
        let weight: f32 = (r * 10.0) - 5.0;

        self.add_link(global_inno_number, node1_index, node2_index, weight)
    }

    pub fn add_random_node(&mut self, global_inno_number: &mut u32) -> Result<(), &'static str> {
        let mut available_links: Vec<usize> = Vec::new();
        for i in 0..self.links.len() {
            if self.links[i].enabled {
                available_links.push(i);
            }
        }

        if available_links.len() == 0 {
            return Err("No links available");
        }

        let link_index = available_links[rand::thread_rng().gen_range(0, available_links.len())];

        self.add_node(global_inno_number, link_index)
    }

    pub fn filter_node_indexes(&self, node_type: NodeType, level: u32) -> Vec<usize> {
        let mut filtered_nodes: Vec<usize> = Vec::new();
        for i in 0..self.nodes.len() {
            let node = self.nodes[i];
            if node.node_type == node_type && node.level == level {
                filtered_nodes.push(i);
            }
        }
        filtered_nodes
    }

    pub fn input_links(&self, node_index: usize) -> Vec<usize> {
        let mut inputs: Vec<usize> = Vec::new();
        for i in 0..self.links.len() {
            if self.links[i].out_index == node_index {
                inputs.push(i);
            }
        }
        inputs
    }

    pub fn evaluate(&self, input_values: Vec<f32>) -> Vec<f32> {
        if self.links.len() == 0 {
            return vec![0.0; self.output_count as usize];
        }

        let mut node_values: Vec<f32> = vec![0.0; self.nodes.len()];

        // Set inputs
        for i in 0..self.input_count as usize {
            node_values[i] = input_values[i];
        }

        // Compute hidden nodes
        let mut current_level = 0u32;
        let mut hidden_nodes_collection: Vec<usize> = vec!(0);

        while hidden_nodes_collection.len() != 0 {
            current_level += 1;
            hidden_nodes_collection = Vec::new();

            // Collect hidden nodes with correct level
            for i in ((self.input_count + self.output_count) as usize)..self.nodes.len() {
                let node = &self.nodes[i];
                if node.level == current_level {
                    hidden_nodes_collection.push(i);
                }
            }

            // Evaluate group of hidden nodes
            for hidden_node_index in hidden_nodes_collection.clone() {
                let link_indexes = self.input_links(hidden_node_index);

                for link_index in link_indexes {
                    let link = &self.links[link_index];
                    if link.enabled {
                        node_values[hidden_node_index] += node_values[link.in_index] * link.weight;
                    }
                }

                // Squash Node Value
                node_values[hidden_node_index] = Network::sigmoid(node_values[hidden_node_index]);
            }
        }

        // Compute output nodes
        for i in self.input_count as usize..((self.input_count + self.output_count) as usize) {
            let link_indexes = self.input_links(i);

            for link_index in link_indexes {
                let link = self.links[link_index].clone();
                if link.enabled {
                    node_values[i] += node_values[link.in_index] * link.weight;
                }
            }

            // Squash Node Value
            node_values[i] = Network::sigmoid(node_values[i]);
        }

        return node_values[(self.input_count as usize)..((self.input_count + self.output_count) as usize)].to_vec();
    }

    pub fn draw(&self, canvas: &mut WindowCanvas, x: f32, y: f32, width: f32, height: f32) {
        let mut node_layer_counts: Vec<u32> = Vec::new();
        node_layer_counts.push(self.input_count);
        node_layer_counts.push(self.output_count);

        let mut hidden_level = 0;
        let mut done_hidden = false;

        // Set node_layer_counts
        while !done_hidden {
            hidden_level += 1;
            let filtered_node_indexes = self.filter_node_indexes(NodeType::Hidden, hidden_level);
            if filtered_node_indexes.len() == 0 {
                done_hidden = true;
            } else {
                node_layer_counts.push(filtered_node_indexes.len() as u32);
            }
        }


        // Find maximum value of node_layer_counts
        let mut most_layer_nodes = 0;
        for layer_nodes in node_layer_counts.clone() {
            if layer_nodes > most_layer_nodes {
                most_layer_nodes = layer_nodes;
            }
        }

        // Set Layer Spacings
        let x_spacing: f32 = width / node_layer_counts.len() as f32;
        let y_spacing: f32 = height / most_layer_nodes as f32;

        // Set Y Offsets to Vertically Center
        let mut y_offsets: Vec<f32> = Vec::new();
        for i in 0..node_layer_counts.len() {
            let current_layer_count = node_layer_counts[i];
            y_offsets.push(((most_layer_nodes - current_layer_count) as f32 / 2.0) * y_spacing);
        }

        // Set Node Positions
        let mut node_positions: Vec<(f32, f32)> = Vec::new();
        for i in 0..node_layer_counts.len() {
            let layer_count = node_layer_counts[i];
            let y_offset = y_offsets[i];
            let x_index = match i {
                0 => 0,
                1 => node_layer_counts.len() - 1,
                _ => i - 1,
            };
            for j in 0..layer_count {
                node_positions.push((x + x_spacing * x_index as f32, y  + y_offset + y_spacing * j as f32));
            }
        }

        // Draw Links
        let line_thickness: u8 = 2;
        for link in &self.links {
            if link.enabled {
                let color: (u8, u8, u8, u8) = if link.weight > 0.0 {
                    (0, 0, (link.weight * 255.0) as u8, 255)
                } else {
                    ((-link.weight * 255.0) as u8, 0, 0, 255)
                };

                let in_pos = node_positions[link.in_index];
                let out_pos = node_positions[link.out_index];
            
                canvas.aa_line(in_pos.0 as i16, in_pos.1 as i16, out_pos.0 as i16, out_pos.1 as i16, color)
                    .expect("Failed to draw link")
            }
        }

        // Draw Nodes
        let circle_radius: i16 = 10;
        for i in 0..self.nodes.len() {
            let node = &self.nodes[i];
            let pos = node_positions[i];
            let color: (u8, u8, u8, u8) = match node.node_type {
                NodeType::Input => (0, 255, 0, 255),
                NodeType::Output => (255, 100, 0, 255),
                NodeType::Hidden => (255, 255, 255, 255),
            };

            canvas.filled_circle(pos.0 as i16, pos.1 as i16, circle_radius, color)
                .expect("Failed to draw node");
        }
    }
}