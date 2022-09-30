use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::vecops::{add, iadd, sub, mul, imul, div};
use crate::{DType,ANode,NodeIdx};

#[derive(Debug)]
pub struct Graph {
    gradients: HashMap<NodeIdx, Vec<DType>>,
    freelist: HashMap<usize, Vec<Vec<DType>>>
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            gradients: HashMap::new(),
            freelist: HashMap::new()
        }
    }

    fn get_or_create_grad(&mut self, node: &ANode) -> Vec<DType> {
        let n_idx = node.get_id();
        if self.gradients.contains_key(&n_idx)  {
            self.gradients.remove(&n_idx).unwrap()
        } else {
            vec![0.; node.value().len()]
        }
    }

    fn get_temp_space(&mut self, size: usize) -> Vec<DType> {
        if let Some(list) = self.freelist.get_mut(&size) {
            if let Some(mut v) = list.pop() {
                v.fill(0f32);
                return v
            }
        }
        vec![0.; size]
    }

    fn ret_temp_space(&mut self, v: Vec<DType>) {
        let e = self.freelist.entry(v.len()).or_insert_with(|| Vec::new());
        e.push(v);
    }

    fn add_grad(&mut self, node: &ANode, grad: Vec<DType>) {
        self.gradients.insert(node.get_id(), grad);
    }

    pub fn backward(&mut self, end_node: &ANode) {
        // dz/dz of course is 1
        let mut z_grad = self.get_or_create_grad(end_node);
        z_grad.fill(1f32);
        self.add_grad(end_node, z_grad);
        self.recurse(end_node);
    }

    fn recurse(&mut self, node: &ANode) {
        if !node.is_leaf() {
            let node_grad = self.get_or_create_grad(node);
            if let Some(children) = node.get_children() {
                // Grab gradients
                let mut grads: Vec<_> = children.iter()
                    .map(|c| self.get_or_create_grad(c))
                    .collect();

                let mut temp_grads: Vec<_> = grads.iter()
                    .map(|g| self.get_temp_space(g.len()))
                    .collect();

                node.compute_grad(&node_grad, &mut temp_grads);

                // Update grads
                grads.iter_mut().zip(temp_grads.into_iter()).for_each(|(g, tg)| {
                    iadd(g, &tg);
                    self.ret_temp_space(tg);
                });

                // Re-add gradients
                children.iter().zip(grads.into_iter()).for_each(|(c, g)| {
                    self.add_grad(c, g);
                });

                // Run children
                for child in children.iter() {
                    self.recurse(child);
                }
            }
        }
    }

    pub fn get_grad(&self, node: &ANode) -> Option<&Vec<DType>> {
        self.gradients.get(&node.get_id())
    }

}


