use std::rc::Rc;

use hashbrown::HashMap;
use hashbrown::hash_map::Entry;
use crate::{DType,ANode,NodeIdx,Node};
use crate::vecops::iadd;
use crate::pool::{allocate_vec,MPVec};

#[derive(Debug)]
pub struct Graph {
    gradients: HashMap<NodeIdx, MPVec>,
    nan_check: bool
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            gradients: HashMap::new(),
            nan_check: false
        }
    }

    #[inline]
    pub fn debug_nan(&mut self, check: bool)  {
        self.nan_check = check;
    }
    
    #[inline]
    pub fn get_grad(&self, node: &ANode) -> Option<&Vec<DType>> {
        self.gradients.get(&node.get_id()).map(|v| v.as_ref())
    }

    #[inline]
    pub fn zero_grads(&mut self) {
        self.gradients.clear();
    }

    #[inline]
    pub fn clear_memory(&mut self) {
        self.gradients.clear();
    }

    fn get_or_create_grad(&mut self, node: &ANode) -> MPVec {
        let n_idx = node.get_id();
        if self.gradients.contains_key(&n_idx)  {
            self.gradients.remove(&n_idx).unwrap()
        } else {
            self.get_temp_space(node.value().len())
        }
    }

    #[inline]
    fn get_temp_space(&mut self, size: usize) -> MPVec {
        allocate_vec(size)
    }

    #[inline]
    fn add_grad(&mut self, node: &ANode, grad: MPVec) {
        self.gradients.insert(node.get_id(), grad);
    }

    #[inline]
    fn add_or_update_grad(&mut self, node: &ANode, grad: MPVec) {
        match self.gradients.entry(node.get_id()) {
            Entry::Occupied(mut entry) => {
                iadd(entry.get_mut(), grad.as_slice());
            },
            Entry::Vacant(mut entry) => {
                entry.insert(grad);
            }
        }
    }

    
    pub fn backward(&mut self, end_node: &ANode) {
        let out = Run::new(end_node);
        // dz/dz of course is 1
        let mut z_grad = self.get_or_create_grad(&out);
        z_grad.fill(1f32);
        
        // Allocate once
        let mut temp_grads = Vec::new();
        self.add_grad(&out, z_grad);
        self.recurse(&out, &mut temp_grads);
    }

     fn recurse(&mut self, node: &ANode, temp_grads: &mut Vec<MPVec>) {
        if !node.is_leaf() {
            let node_grad = self.get_or_create_grad(node);
            if let Some(children) = node.get_children() {
                temp_grads.clear();
                // Grab gradients

                temp_grads.extend(children.iter()
                    .map(|c| self.get_temp_space(c.value().len())));

                node.compute_grad(&node_grad, temp_grads);

                if self.nan_check {
                    for (i, grad) in temp_grads.iter().enumerate() {
                        for gi in grad.iter() {
                            if gi.is_nan() {
                                eprintln!("Nan detected with id {:?}, child {}", node.get_id(), i);
                                panic!()
                            }
                        }
                    }
                }

                // Update grads

                // Re-add gradients
                children.iter().zip(temp_grads.drain(..)).for_each(|(c, g)| {
                    self.add_or_update_grad(c, g);
                });

                if node.requires_grad() {
                    self.add_grad(node, node_grad);
                }

                // Run children
                for child in children.iter() {
                    self.recurse(child, temp_grads);
                }

            } else {
                self.add_grad(node, node_grad);
            }
        }
    }


}

pub(crate) struct Run(NodeIdx, Vec<ANode>);

impl Run {
    pub(crate) fn new(x: &ANode) -> ANode {
        let idx = NodeIdx::new();
        ANode::new(Rc::new(Run(idx, vec![x.clone()])))
    }
}

impl Node for Run {
    fn get_id(&self) -> NodeIdx { self.0.clone() }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(&self.1)
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.1[0].value()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], results: &mut [MPVec]) {
        let mut out = &mut results[0];
        out.fill(1f32);
    }
}

