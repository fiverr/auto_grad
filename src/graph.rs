use std::rc::Rc;
use std::ops::Add;

use std::cell::UnsafeCell;
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

    pub fn stats(&self, node: &ANode) -> GraphStats {
        let stats = GraphStats::new(1, node.value().len());
        if let Some(children) = node.get_children() {
            children.iter()
                .map(|cn| self.stats(cn))
                .fold(stats, |acc, x| acc + x)
        } else {
            stats
        }
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
    fn add_or_update_grad(&mut self, node: &ANode, grad: &mut [f32]) {
        match self.gradients.entry(node.get_id()) {
            Entry::Occupied(mut entry) => {
                iadd(entry.get_mut(), grad);
            },
            Entry::Vacant(mut entry) => {
                let mut v = allocate_vec(0);
                v.extend_from_slice(grad);

                entry.insert(v);
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
        let mut space = UnsafeCell::new(Vec::new());
        self.recurse(&out, &mut temp_grads, &mut space);
    }

    fn get_mut_slices<'a,'b>(
        &self,
        nodes: &[ANode],
        space: &UnsafeCell<Vec<DType>>, 
        buff: &mut Vec<&'a mut [DType]>
    ) {
        buff.clear();
        let size = nodes.iter().map(|n| n.value().len()).sum::<usize>();
        unsafe {
            let mut s = &mut *space.get();
            while s.len() < size + 1 {
                s.push(0.);
            }
            (&mut s[..size]).fill(0.);
        }

        let mut offset = 0;
        for node in nodes {
            let len = node.value().len();
            unsafe {
                buff.push(&mut (*space.get())[offset..offset+len]);
            }
            offset += len;
        }
    }

    fn recurse(&mut self, node: &ANode, temp_grads: &mut Vec<&mut [DType]>, space: &UnsafeCell<Vec<DType>>) {
        if !node.is_leaf() {
            let node_grad = self.get_or_create_grad(node);
            if let Some(children) = node.get_children() {
                self.get_mut_slices(children, space, temp_grads);

                node.compute_grad(&node_grad, temp_grads.as_mut_slice());

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
                    self.recurse(child, temp_grads, space);
                }

            } else {
                if node.requires_grad() {
                    self.add_grad(node, node_grad);
                }
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

    fn compute_grad(&self, grad: &[DType], results: &mut [&mut [DType]]) {
        let mut out = &mut results[0];
        out.fill(1f32);
    }
}

#[derive(Clone,Copy,Debug)]
pub struct GraphStats {
    ops: usize,
    memory: usize
}

impl GraphStats {
    fn new(ops: usize, memory: usize) -> Self {
        GraphStats {ops, memory}
    }

    fn zero() -> Self {
        GraphStats::new(0, 0)
    }
}

impl Add for GraphStats {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            ops: self.ops + other.ops,
            memory: self.memory + other.memory,
        }
    }
}


#[cfg(test)]
mod graph_tests {
    use super::*;
    use crate::*;

    #[test]
    fn test_add() {
        let x = Variable::new(vec![0., 1.]);
        let y = Variable::new(vec![2., 3.]);
        let res = x + y;
        let graph = Graph::new();
        let stats = graph.stats(&res);
        assert_eq!(stats.ops, 3);
        assert_eq!(stats.memory, 6);
    }
}
