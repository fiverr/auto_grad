use std::rc::Rc;

use hashbrown::HashMap;
use crate::{DType,ANode,NodeIdx,Node};
use crate::vecops::iadd;
use crate::pool::allocate_vec;

#[derive(Debug)]
pub struct Graph {
    gradients: HashMap<NodeIdx, Vec<DType>>,
    freelist: HashMap<usize, Vec<Vec<DType>>>,
    nan_check: bool
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            gradients: HashMap::new(),
            freelist: HashMap::new(),
            nan_check: false
        }
    }

    pub fn debug_nan(&mut self, check: bool)  {
        self.nan_check = check;
    }

    pub fn backward(&mut self, end_node: &ANode) {
        let out = Run::new(end_node);
        // dz/dz of course is 1
        let mut z_grad = self.get_or_create_grad(&out);
        z_grad.fill(1f32);
        self.add_grad(&out, z_grad);
        self.recurse(&out);
    }

    pub fn get_grad(&self, node: &ANode) -> Option<&Vec<DType>> {
        self.gradients.get(&node.get_id())
    }

    pub fn zero_grads(&mut self) {
        let mut hm = HashMap::new();
        std::mem::swap(&mut hm, &mut self.gradients);
        hm.into_values()
            .for_each(|v| self.ret_temp_space(v));
    }

    pub fn clear_memory(&mut self) {
        self.gradients.clear();
        self.freelist.clear();
    }

    fn get_or_create_grad(&mut self, node: &ANode) -> Vec<DType> {
        let n_idx = node.get_id();
        if self.gradients.contains_key(&n_idx)  {
            self.gradients.remove(&n_idx).unwrap()
        } else {
            self.get_temp_space(node.value().len())
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
                grads.iter_mut().zip(temp_grads.into_iter()).for_each(|(g, tg)| {
                    iadd(g, &tg);
                    self.ret_temp_space(tg);
                });

                // Re-add gradients
                children.iter().zip(grads.into_iter()).for_each(|(c, g)| {
                    self.add_grad(c, g);
                });

                if node.requires_grad() {
                    self.add_grad(node, node_grad);
                } else {
                    self.ret_temp_space(node_grad);
                }

                // Run children
                for child in children.iter() {
                    self.recurse(child);
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

    fn compute_grad(&self, grad: &[DType], results: &mut [Vec<DType>]) {
        let mut out = &mut results[0];
        out.fill(1f32);
    }
}

