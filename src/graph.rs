use std::collections::HashMap;
use std::ops::{Add, Sub, Mul, Div,Deref};
use std::sync::atomic::{AtomicUsize,ATOMIC_USIZE_INIT, Ordering};
use std::sync::{Arc, Mutex};

use crate::vecops::{add, iadd, sub, mul, imul, div};

type DType = f32;

static GLOBAL_HANDLE_COUNT: AtomicUsize = ATOMIC_USIZE_INIT;

#[derive(Clone,Copy,Eq,Hash,PartialEq,Ord,PartialOrd,Debug)]
struct NodeIdx(usize);

impl NodeIdx {
    fn new() -> Self {
        NodeIdx(GLOBAL_HANDLE_COUNT.fetch_add(1, Ordering::SeqCst))
    }
}

#[derive(Clone)]
struct Computation {
    value: Vec<DType>
}

impl Computation {
    fn new(value: Vec<DType>) -> Self {
        Computation { value: value }
    }
}

trait Node {
    fn get_id(&self) -> NodeIdx;

    fn is_leaf(&self) -> bool;

    fn get_children(&self) -> Option<&[ANode]>;

    fn value(&self) -> &[DType];

    fn requires_grad(&self) -> bool;

    fn compute_grad(&self, grad: &[DType], results: &mut [Vec<DType>]) { }

}

#[derive(Debug)]
struct Graph {
    gradients: HashMap<NodeIdx, Vec<DType>>,
    freelist: HashMap<usize, Vec<Vec<DType>>>
}

impl Graph {
    fn new() -> Self {
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

    fn backward(&mut self, end_node: &ANode) {
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

    fn get_grad(&self, node: &ANode) -> Option<&Vec<DType>> {
        self.gradients.get(&node.get_id())
    }

}

#[derive(Clone)]
struct ANode(Arc<dyn Node>);

impl ANode {
    fn new(n: Arc<dyn Node>) -> Self {
        ANode(n)
    }
}

impl Deref for ANode {
    type Target = Arc<dyn Node>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}


#[derive(Clone)]
struct Variable(NodeIdx, Computation);

impl Variable {
    fn new(value: Vec<DType>) -> ANode {
        let v = Variable(NodeIdx::new(), Computation::new(value));
        ANode::new(Arc::new(v))
    }
}

impl Node for Variable {
    fn get_id(&self) -> NodeIdx { self.0.clone() }

    fn is_leaf(&self) -> bool { true }

    fn value(&self) -> &[DType] {
        &self.1.value
    }

    fn get_children(&self) -> Option<&[ANode]> { None }

    fn requires_grad(&self) -> bool { true }

    fn compute_grad(&self, grad: &[DType], results: &mut [Vec<DType>]) {
        // Pass
    }
}

#[derive(Clone)]
struct Constant(NodeIdx, Computation);

impl Constant {
    fn new(value: Vec<DType>) -> ANode {
        let c = Constant(NodeIdx::new(), Computation::new(value));
        ANode::new(Arc::new(c))
    }
    fn scalar(value: DType) -> ANode {
        Constant::new(vec![value])
    }
}

impl Node for Constant {
    fn get_id(&self) -> NodeIdx { self.0.clone() }

    fn get_children(&self) -> Option<&[ANode]> { None }

    fn is_leaf(&self) -> bool { true }

    fn value(&self) -> &[DType] {
        &self.1.value
    }

    fn requires_grad(&self) -> bool { false }
}

struct AddN(NodeIdx, Vec<ANode>, Computation);

impl AddN {
    fn new(left: ANode, right: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = AddN::compute(&left, &right);
        let node = AddN(idx, vec![left, right], Computation::new(value));
        ANode::new(Arc::new(node))
    }

    fn compute(left: &ANode, right: &ANode) -> Vec<DType> {
        let lv = left.value();
        let rv = right.value();
        let mut out = vec![0.; lv.len()];
        add(&lv, &rv, &mut out);
        out
    }
}

impl Node for AddN {
    fn get_id(&self) -> NodeIdx { self.0.clone() }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.value
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], results: &mut [Vec<DType>]) {
        // f(x,y) = x - y
        // df(x,y)/dx = 1
        // df(x,y)/dy = 1
        results[0].clone_from_slice(grad);
        results[1].clone_from_slice(grad);
    }

}

struct Subtract(NodeIdx, Vec<ANode>, Computation);

impl Subtract {
    fn new(left: ANode, right: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Subtract::compute(&left, &right);
        let node = Subtract(idx, vec![left, right], Computation::new(value));
        ANode::new(Arc::new(node))
    }

    fn compute(left: &ANode, right: &ANode) -> Vec<DType> {
        let lv = left.value();
        let rv = right.value();
        let mut out = vec![0.; lv.len()];
        sub(&lv, &rv, &mut out);
        out
    }
}

impl Node for Subtract {
    fn get_id(&self) -> NodeIdx { self.0.clone() }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.value
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], results: &mut [Vec<DType>]) {
        // f(x,y) = x - y
        // df(x,y)/dx = 1
        // df(x,y)/dy = -1
        results[0].fill(1f32);
        imul(&mut results[0], grad);

        results[1].fill(-1f32);
        imul(&mut results[1], grad);
    }

}

struct Multiply(NodeIdx, Vec<ANode>, Computation);

impl Multiply {
    fn new(left: ANode, right: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Multiply::compute(&left, &right);
        let node = Multiply(idx, vec![left, right], Computation::new(value));
        ANode::new(Arc::new(node))
    }

    fn compute(left: &ANode, right: &ANode) -> Vec<DType> {
        let lv = left.value();
        let rv = right.value();
        let mut out = vec![0.; lv.len()];
        mul(lv, rv, &mut out);
        out
    }
}

impl Node for Multiply {
    fn get_id(&self) -> NodeIdx { self.0.clone() }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.value
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], results: &mut [Vec<DType>]) {
        // f(x,y) = x * y
        // df(x,y)/dx = y
        // df(x,y)/dy = x
        let x = self.1[0].value();
        let y = self.1[1].value();
        results[0].clone_from_slice(y);
        imul(&mut results[0], &grad);

        results[1].clone_from_slice(x);
        imul(&mut results[1], &grad);
    }

}

struct Divide(NodeIdx, Vec<ANode>, Computation);

impl Divide {
    fn new(left: ANode, right: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Divide::compute(&left, &right);
        let node = Divide(idx, vec![left, right], Computation::new(value));
        ANode::new(Arc::new(node))
    }

    fn compute(left: &ANode, right: &ANode) -> Vec<DType> {
        let lv = left.value();
        let rv = right.value();
        let mut out = vec![0.; lv.len()];
        div(lv, rv, &mut out);
        out
    }
}

impl Node for Divide {
    fn get_id(&self) -> NodeIdx { self.0.clone() }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.value
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], results: &mut [Vec<DType>]) {
        // f(x,y) = x / y
        let x = self.1[0].value();
        let y = self.1[1].value();
        // df(x,y)/dx = 1 / y
        let mut out = &mut results[0];
        out.iter_mut().zip(y.iter())
            .for_each(|(oi, yi)| *oi = 1f32 / *yi);

        imul(&mut out, grad);
        
        // df(x,y)/dy = x / y ^ 2
        let mut out = &mut results[1];
        out.iter_mut().zip(x.iter().zip(y.iter())).for_each(|(oi, (xi, yi))| {
            *oi = *xi / yi.powf(2f32);
        });
        imul(&mut out, grad);
    }

}

struct Power(NodeIdx, Vec<ANode>, Computation);

impl Power {
    fn new(base: ANode, exp: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Power::compute(&base, &exp);
        let node = Power(idx, vec![base, exp], Computation::new(value));
        ANode::new(Arc::new(node))
    }

    fn compute(left: &ANode, right: &ANode) -> Vec<DType> {
        let lv = left.value();
        let rv = right.value();
        lv.iter().zip(rv.iter()).map(|(lvi, rvi)| lvi.powf(*rvi)).collect()
    }
}

impl Node for Power {
    fn get_id(&self) -> NodeIdx { self.0.clone() }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.value
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], results: &mut [Vec<DType>]) {
        // f(x,y) = x ^ y
        // df(x,y)/dx = y * x ^ (y - 1)
        // df(x,y)/dy = ln(y) * x ^ y
        let x = self.1[0].value();
        let y = self.1[1].value();
        
        let mut out = &mut results[0];
        out.iter_mut().zip(x.iter().zip(y.iter()))
            .for_each(|(oi, (xi, yi))| *oi = *yi * xi.powf(*yi - 1f32));
        imul(&mut out, grad);
        
        let mut out = &mut results[1];

        out.iter_mut().zip(x.iter().zip(y.iter()))
            .for_each(|(oi, (xi, yi))| *oi = (yi).ln() * xi.powf(*yi));
        imul(&mut out, grad);
    }

}

trait Pow<Rhs=Self> {
    type Output;
    fn powf(self, rhs: Rhs) -> Self::Output;
}

impl Add for &ANode {
    type Output = ANode;
    fn add(self, rhs: &ANode) -> Self::Output {
        AddN::new(self.clone(), rhs.clone())
    }
}

impl Add<f32> for &ANode {
    type Output = ANode;
    fn add(self, rhs: f32) -> Self::Output {
        AddN::new(self.clone(), Constant::scalar(rhs))
    }
}


impl Sub for &ANode {
    type Output = ANode;
    fn sub(self, rhs: &ANode) -> Self::Output {
        Subtract::new(self.clone(), rhs.clone())
    }
}

impl Sub<f32> for &ANode {
    type Output = ANode;
    fn sub(self, rhs: f32) -> Self::Output {
        Subtract::new(self.clone(), Constant::scalar(rhs))
    }
}

impl Mul for &ANode {
    type Output = ANode;
    fn mul(self, rhs: &ANode) -> Self::Output {
        Multiply::new(self.clone(), rhs.clone())
    }
}

impl Mul<f32> for &ANode {
    type Output = ANode;
    fn mul(self, rhs: f32) -> Self::Output {
        Multiply::new(self.clone(), Constant::scalar(rhs))
    }
}


impl Div for &ANode {
    type Output = ANode;
    fn div(self, rhs: &ANode) -> Self::Output {
        Divide::new(self.clone(), rhs.clone())
    }
}

impl Div<f32> for &ANode {
    type Output = ANode;
    fn div(self, rhs: f32) -> Self::Output {
        Divide::new(self.clone(), Constant::scalar(rhs))
    }
}

impl Pow for &ANode {
    type Output = ANode;
    fn powf(self, rhs: &ANode) -> Self::Output {
        Power::new(self.clone(), rhs.clone())
    }
}

impl Pow<f32> for &ANode {
    type Output = ANode;
    fn powf(self, rhs: f32) -> Self::Output {
        Power::new(self.clone(), Constant::scalar(rhs))
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let x = Variable::new(vec![0., 1.]);
        let y = Variable::new(vec![2., 3.]);
        let res = AddN::new(x, y);
        assert_eq!(res.value(), &[2., 4.]);
    }

    #[test]
    fn test_sub() {
        let x = Variable::new(vec![0., 1.]);
        let y = Variable::new(vec![2., 3.]);
        let res = Subtract::new(x, y);
        assert_eq!(res.value(), &[-2., -2.]);
    }

    #[test]
    fn test_mul() {
        let x = Variable::new(vec![0., 1.]);
        let y = Variable::new(vec![2., 3.]);
        let res = Multiply::new(x, y);
        assert_eq!(res.value(), &[0., 3.]);
    }

    #[test]
    fn test_div() {
        let x = Variable::new(vec![0., 1.]);
        let y = Variable::new(vec![2., 3.]);
        let res = Divide::new(x, y);
        assert_eq!(res.value(), &[0., 1./3.]);
    }

    #[test]
    fn test_pow() {
        let x = Variable::new(vec![0., 1., 2.]);
        let y = Variable::new(vec![2., 3., 3.]);
        let res = Power::new(x, y);
        assert_eq!(res.value(), &[0., 1., 8.]);
    }

    #[test]
    fn test_backward_pass_simple1() {
        // 2x
        // df/dx = 2
        let x = Variable::new(vec![0f32]);
        let x2 = Multiply::new(x.clone(), Constant::scalar(2f32));

        let mut graph = Graph::new();
        graph.backward(&x2);
        println!("X: {:?}", x.get_id());
        println!("Graph: {:?}", graph);
        let x_grad = graph.get_grad(&x);
        assert_eq!(Some(&vec![2f32]), x_grad);
    }

    #[test]
    fn test_backward_pass_simple2() {
        // 2 + x
        // df/dx = 1
        let x = Variable::new(vec![0f32]);
        let x2 = AddN::new(x.clone(), Constant::scalar(2f32));

        let mut graph = Graph::new();
        graph.backward(&x2);
        println!("X: {:?}", x.get_id());
        println!("Graph: {:?}", graph);
        let x_grad = graph.get_grad(&x);
        assert_eq!(Some(&vec![1f32]), x_grad);
    }

    #[test]
    fn test_backward_pass_simple3() {
        // x - y
        // df/dx = 1
        let x = Variable::new(vec![1f32]);
        let y = Variable::new(vec![2f32]);
        let x2 = Subtract::new(x.clone(), y.clone());

        let mut graph = Graph::new();
        graph.backward(&x2);
        println!("Graph: {:?}", graph);
        let x_grad = graph.get_grad(&x);
        let y_grad = graph.get_grad(&y);
        println!("X Grad: {:?}", x_grad);
        println!("Y Grad: {:?}", y_grad);
        assert_eq!(Some(&vec![1f32]), x_grad);
        assert_eq!(Some(&vec![-1f32]), y_grad);
    }

    #[test]
    fn test_backward_pass_simple4() {
        // x ^ 2
        // df/dx = 2x
        let x = Variable::new(vec![1f32]);
        let x2 = Power::new(x.clone(), Constant::scalar(2f32));

        let mut graph = Graph::new();
        graph.backward(&x2);

        let x_grad = graph.get_grad(&x);
        println!("X Grad: {:?}", x_grad);
        assert_eq!(Some(&vec![2f32]), x_grad);
    }

    #[test]
    fn test_backward_pass_simple5() {
        // x ^ 2 + 3x
        // df/dx = 2x + 3
        let x = Variable::new(vec![1f32]);
        let x2 = Power::new(x.clone(), Constant::scalar(2f32));
        let x3 = Multiply::new(x.clone(), Constant::scalar(3f32));
        let x4 = AddN::new(x2, x3);

        assert_eq!(x4.value(), vec![4f32]);

        let mut graph = Graph::new();
        graph.backward(&x4);

        let x_grad = graph.get_grad(&x);
        println!("X Grad: {:?}", x_grad);
        assert_eq!(Some(&vec![5f32]), x_grad);
    }


    #[test]
    fn test_backward_pass_simple() {
        // 2x + 3
        // df/dx = 2
        let x = Variable::new(vec![0f32]);
        let x2 = Multiply::new(x.clone(), Constant::scalar(2f32));
        let x2_3 = AddN::new(x2, Constant::scalar(3f32));

        let mut graph = Graph::new();
        graph.backward(&x2_3);
        println!("X: {:?}", x.get_id());
        println!("Graph: {:?}", graph);
        let x_grad = graph.get_grad(&x);
        assert_eq!(Some(&vec![2f32]), x_grad);
    }

    #[test]
    fn test_backward_pass_complicated() {
        // (x+2) ^ 2 
        // x^2 + 4x + 4
        // 2x + 4
        let x      = Variable::new(vec![0f32]);
        let x2     = AddN::new(x.clone(), Constant::scalar(2f32));
        let x2_2   = Power::new(x2.clone(), Constant::scalar(2f32));

        assert_eq!(x2_2.value(), vec![4f32]);

        let mut graph = Graph::new();
        graph.backward(&x2_2);

        println!("X: {:?}", x.get_id());
        println!("Graph: {:?}", graph);

        let x2_grad = graph.get_grad(&x2);
        println!("x2 grad:{:?}", x2_grad);
        let x_grad = graph.get_grad(&x);
        assert_eq!(Some(&vec![4f32]), x_grad);
    }

    #[test]
    fn test_composition() {
        // (x+2) ^ 2 
        let x      = Variable::new(vec![0f32]);
        let res = (&x + 2f32).powf(2f32);
        assert_eq!(res.value(), vec![4f32]);

        let mut graph = Graph::new();
        graph.backward(&res);

        let x_grad = graph.get_grad(&x);
        assert_eq!(Some(&vec![4f32]), x_grad);
    }


}
