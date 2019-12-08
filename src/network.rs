use sdl2::render::WindowCanvas;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use sdl2::gfx::primitives::DrawRenderer;
use std::fmt;
use std::string::String;
use rand::Rng;
use std::cmp::*;


pub fn sigmoid(x: f32) -> f32 {
    let ex = std::f32::consts::E.powf(x);
    ((ex) / (ex + 1.0)) * 2.0 - 1.0
}

fn get_node_type_name(node_type: NodeType) -> &'static str {
    match node_type {
        NodeType::Input => "INPUT",
        NodeType::Hidden => "HIDDEN",
        NodeType::Output => "OUTPUT",
    }
}

#[derive(Debug, Clone, Copy)]
pub enum NodeType {
    Input,
    Hidden,
    Output,
}

impl PartialEq for NodeType {
    fn eq(&self, other: &Self) -> bool {
        match self {
            NodeType::Input => {
                match other {
                    NodeType::Input => true,
                    NodeType::Output => false,
                    NodeType::Hidden => false,
                }
            },

            NodeType::Output => {
                match other {
                    NodeType::Input => false,
                    NodeType::Output => true,
                    NodeType::Hidden => false,
                }
            },

            NodeType::Hidden => {
                match other {
                    NodeType::Input => false,
                    NodeType::Output => false,
                    NodeType::Hidden => true,
                }
            },
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Node {
    pub id: u32,
    pub node_type: NodeType,
    pub z_index: u32,
    pub value: f32,
    pub graphics_x: f32,
    pub graphics_y: f32,
}

impl Node {
    pub fn new(id: u32, node_type: NodeType) -> Node {
        let z_index = match node_type {
            NodeType::Hidden => 1,
            _ => 0,
        };
        
        Node {
            id: id,
            node_type: node_type,
            z_index: z_index,
            value: 0.0,
            graphics_x: 0.0,
            graphics_y: 0.0,
        }
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.node_type == NodeType::Hidden {
            write!(f, "-----NODE-----\n#{}\n{}\nZ: {}\n--------------", self.id, get_node_type_name(self.node_type), self.z_index)
        } else {
            write!(f, "-----NODE-----\n#{}\n{}\n--------------", self.id, get_node_type_name(self.node_type))
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Connection {
    pub in_node: u32,
    pub out_node: u32,
    pub weight: f32,
    pub enabled: bool,
    pub id: u32,
}

impl fmt::Display for Connection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.enabled {
            write!(f, "-----CONN-----\n#{}\n{} --> {}\nW: {}\n--------------", self.id, self.in_node, self.out_node, self.weight)
        } else {
            write!(f, "-----CONN-----\n#{}\n{} --> {}\nW: {}\nDISABLED\n--------------", self.id, self.in_node, self.out_node, self.weight)
        }
    }
}

#[derive(Debug, Clone)]
pub struct Network {
    pub nodes: Vec<Node>,
    pub connections: Vec<Connection>,
    pub fitness: f32,
    pub generation: u32,
    pub kid: u32,
    pub greatest_id: u32,
    pub greatest_z_index: u32,
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

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut display = String::new();
        display += "------------------------\nNETWORK\n";
        display += format!("FITNESS: {}\nGEN: {}\nKID: {}\n", self.fitness, self.generation, self.kid).as_str();
        for node in &self.nodes {
            display += format!("{}\n", node).as_str();
        }

        for connection in &self.connections {
            display += format!("{}\n", connection).as_str();
        }

        display += "------------------------\n";

        write!(f, "{}", display)
    }
}

impl Network {
    fn new_id(&mut self) -> u32 {
        self.greatest_id += 1;
        return self.greatest_id; 
    }

    pub fn new(num_inputs: u32, num_outputs: u32) -> Network {
        let mut nodes: Vec<Node> = Vec::new();
        let mut greatest_id: u32 = 0;
        // Input Nodes
        for _ in 0..num_inputs {
            let id = greatest_id + 1;
            greatest_id += 1;
            nodes.push(Node::new((id) as u32, NodeType::Input));
        }

        // Output Nodes
        for _ in num_inputs..(num_inputs + num_outputs) {
            let id = greatest_id + 1;
            greatest_id += 1;
            nodes.push(Node::new((id) as u32, NodeType::Output));
        }

        Network {
            nodes: nodes,
            connections: Vec::new(),
            fitness: 0.0,
            generation: 0,
            kid: 0,
            greatest_id: greatest_id,
            greatest_z_index: 0,
        }
    }

    pub fn filter_nodes(&self, node_type: NodeType) -> Vec<Node> {
        let mut filtered_nodes: Vec<Node> = Vec::new();
        for i in 0..self.nodes.len() {
            if self.nodes[i].node_type == node_type {
                filtered_nodes.push(self.nodes[i]);
            }
        }
        return filtered_nodes;
    }

    pub fn filter_hidden_nodes(&self, z_index: u32) -> Vec<Node> {
        let mut filtered_nodes: Vec<Node> = Vec::new();
        for i in 0..self.nodes.len() {
            if self.nodes[i].z_index == z_index {
                filtered_nodes.push(self.nodes[i]);
            }
        }
        return filtered_nodes;
    }

    pub fn get_node_index(&self, node_id: u32) -> usize {
        for i in 0..self.nodes.len() {
            if self.nodes[i].id == node_id {
                return i as usize;
            }
        }
        return 0;
    }

    pub fn get_connection_index(&self, connection_id: u32) -> usize {
        for i in 0..self.connections.len() {
            if self.connections[i].id == connection_id {
                return i as usize;
            }
        }
        return 0;
    }

    pub fn new_connection(&mut self, node_id_1: u32, node_id_2: u32, weight: f32) {
        let index1 = self.get_node_index(node_id_1);
        let index2 = self.get_node_index(node_id_2);

        let node1 = &self.nodes[index1];
        let node2 = &self.nodes[index2];

        if node1.node_type == node2.node_type && node1.z_index == node2.z_index {
            //TODO: Throw Error
            return;
        }

        let mut in_node: u32 = 0;
        let mut out_node: u32 = 0;

        if node1.node_type == NodeType::Input {
            in_node = node1.id;
            out_node = node2.id;
        } else if node1.node_type == NodeType::Output {
            in_node = node2.id;
            out_node = node1.id;
        } else if node1.z_index > node2.z_index {
            in_node = node2.id;
            out_node = node1.id;
        } else if node1.z_index < node2.z_index {
            in_node = node1.id;
            out_node = node2.id;
        } else {
            //TODO: Throw Error
        }

        let connection = Connection {
            in_node: in_node,
            out_node: out_node,
            weight: weight,
            enabled: true,
            id: self.new_id(),
        };

        self.connections.push(connection);
    }

    pub fn new_node(&mut self, connection_id: u32) {
        let index = self.get_connection_index(connection_id);
        let connection = self.connections[index];

        let in_index = self.get_node_index(connection.in_node);
        let out_index = self.get_node_index(connection.out_node);

        let z_index = self.nodes[in_index].z_index + 1;
        if z_index > self.greatest_z_index {
            self.greatest_z_index = z_index;
        }

        if self.nodes[out_index].z_index == z_index {
            self.nodes[out_index].z_index += 1;
            if self.nodes[out_index].z_index > self.greatest_z_index {
                self.greatest_z_index = self.nodes[out_index].z_index;
            }
        }

        let new_node_id = self.new_id();

        let new_node = Node {
            id: new_node_id,
            node_type: NodeType::Hidden,
            z_index: z_index,
            value: 0.0,
            graphics_x: 0.0,
            graphics_y: 0.0,
        };

        // Push new node
        self.nodes.push(new_node);

        // Change/Add connections
        self.connections[index].enabled = false;
        let connection_a = Connection {
            in_node: connection.in_node,
            out_node: new_node_id,
            weight: 1.0,
            enabled: true,
            id: self.new_id(),
        };

        let connection_b = Connection {
            in_node: new_node_id,
            out_node: connection.out_node,
            weight: connection.weight,
            enabled: true,
            id: self.new_id(),
        };

        // Push new connections
        self.connections.push(connection_a);
        self.connections.push(connection_b);
    }

    pub fn mutate_add_node(&mut self) {
        if self.connections.len() > 0 {
            let mut selected = false;
            while !selected {
                let index = rand::thread_rng().gen_range(0, self.connections.len());
                let id = self.connections[index].id;
                if self.connections[index].enabled {
                    selected = true;
                    self.new_node(id);
                } else {
                    selected = false;
                }
            }
        } else {
        }
    }

    pub fn mutate_add_connection(&mut self) {
        // Select first node
        let index1 = rand::thread_rng().gen_range(0, self.nodes.len());
        let node1 = self.nodes[index1].clone();
        
        let mut selected = false;
        let mut node2 = self.nodes[0].clone();
        while !selected {
            let index2 = rand::thread_rng().gen_range(0, self.nodes.len());
            node2 = self.nodes[index2].clone();

            if node1.id == node2.id || (node1.node_type != NodeType::Hidden && node1.node_type == node2.node_type) {
                selected = false;
            } else {
                selected = true;
                // Select weight
                let r: f32 = rand::thread_rng().gen();
                let weight = (r * 2.0) - 1.0;
                self.new_connection(node1.id, node2.id, weight);
            }
        }
    }

    pub fn input_connections(&self, node: Node) -> Vec<Connection> {
        let mut filtered_connections = Vec::new() as Vec<Connection>;

        if node.node_type == NodeType::Input || self.connections.len() == 0 {
            return filtered_connections;
        }

        for i in 0..self.connections.len() {
            if self.connections[i].out_node == node.id {
                filtered_connections.push(self.connections[i]);
            }
        }

        return filtered_connections;
    }

    pub fn evaluate(&mut self, input_values: Vec<f32>) -> Vec<f32> {
        for i in 0..input_values.len() {
            self.nodes[i].value = input_values[i];
        }

        //  Evaluate the hidden layer
        let mut current_z_index = 1;
        let mut current_nodes: Vec<Node> = vec![Node::new(0, NodeType::Input)];

        while current_nodes.len() != 0 {
            current_nodes = self.filter_hidden_nodes(current_z_index);

            if current_nodes.len() != 0 {
                current_z_index += 1;
            }

            for i in 0..current_nodes.len() {
                let mut node = current_nodes[i].clone();
                // Reset values
                node.value = 0.0;
                let input_connections = self.input_connections(node);
                for connection in input_connections {
                    if connection.enabled {
                        let previous_node = self.nodes[self.get_node_index(connection.in_node)].clone();
                        node.value += connection.weight * previous_node.value;
                    }
                }
                // Squash the node value and write it to self.nodes
                node.value = sigmoid(node.value);
                let node_index = self.get_node_index(node.id);
                self.nodes[node_index].value = node.value;
            }
        }

        // Evaluate the Output Layer
        let mut output_values: Vec<f32> = Vec::new();
        let output_nodes = self.filter_nodes(NodeType::Output).clone();
        for i in 0..output_nodes.len() {
            let mut node = output_nodes[i].clone();
            node.value = 0.0;

            let input_connections = self.input_connections(node);
            for connection in input_connections {
                if connection.enabled {
                    let previous_node = self.nodes[self.get_node_index(connection.in_node)].clone();
                    node.value += connection.weight * previous_node.value;
                }
            }
            // Squash the node value and write it to self.nodes
            node.value = sigmoid(node.value);
            let node_index = self.get_node_index(node.id);
            self.nodes[node_index].value = node.value;
            output_values.push((node.value + 1.0) / 2.0);
        }
        return output_values;
    }


    pub fn draw(&mut self, canvas: &mut WindowCanvas, x: f32, y: f32) {
        let node_size: f32 = 10.0;
        let network_width: f32 = 500.0;
        let node_y_spacing: f32 = 80.0;
        let greatest_z = self.greatest_z_index;
        let x_spacing = network_width / (greatest_z as f32 + 1.0);
        let input_nodes = self.filter_nodes(NodeType::Input).clone();
        let output_nodes = self.filter_nodes(NodeType::Output).clone();

        // Get Positions of Nodes
        // Input Layers
        for i in 0..input_nodes.len() {
            let index = self.get_node_index(input_nodes[i].id);
            let node = &mut self.nodes[index];
            node.graphics_x = x;
            node.graphics_y = y + node_y_spacing * (i as f32);
        }

        // Hidden Layer
        for j in 1..(self.greatest_z_index + 1) {
            let nodes = self.filter_hidden_nodes(j);
            for i in 0..nodes.len() {
                let index = self.get_node_index(nodes[i].id);
                let node = &mut self.nodes[index];
                node.graphics_x = x + x_spacing * node.z_index as f32;
                node.graphics_y = y + node_y_spacing * (i as f32);
            }
        }

        // Output Layers
        for i in 0..output_nodes.len() {
            let index = self.get_node_index(output_nodes[i].id);
            let node = &mut self.nodes[index];
            node.graphics_x = x + network_width;
            node.graphics_y = y + node_y_spacing * (i as f32);
        }

        // Draw Connections
        for connection in &self.connections {
            let in_node = &self.nodes[self.get_node_index(connection.in_node)];
            let out_node = &self.nodes[self.get_node_index(connection.out_node)];
            let color: (u8, u8, u8, u8) = if connection.weight > 0.0 {
                (0, 0, (255.0*connection.weight) as u8, 255)
            } else {
                ((255.0*connection.weight) as u8, 0, 0, 255)
            };
            canvas.thick_line(in_node.graphics_x as i16, in_node.graphics_y as i16, out_node.graphics_x as i16, out_node.graphics_y as i16, 2, color)
                .expect("Failed to draw connection");
        }

        for node in &self.nodes{
            let color: (u8, u8, u8, u8) = if node.node_type == NodeType::Input {
                (0, 255, 0, 255)
            } else if node.node_type == NodeType::Output {
                (255, 100, 0, 255)
            } else {
                (255, 255, 255, 255)
            };
            canvas.filled_circle(node.graphics_x as i16, node.graphics_y as i16, node_size as i16, color)
                .expect("Failed to draw node");
        }
    }
}