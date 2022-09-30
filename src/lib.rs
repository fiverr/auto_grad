mod graph;
mod vecops;
mod ops;

pub use graph::Graph;
pub use ops::{Variable,Constant};

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::ops::{Add,Sub,Mul,Div,Deref};

use crate::ops::*;

static GLOBAL_HANDLE_COUNT: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone,Copy,Eq,Hash,PartialEq,Ord,PartialOrd,Debug)]
pub struct NodeIdx(usize);

type DType = f32;

impl NodeIdx {
    fn new() -> Self {
        NodeIdx(GLOBAL_HANDLE_COUNT.fetch_add(1, Ordering::SeqCst))
    }
}


pub trait Node {
    fn get_id(&self) -> NodeIdx;

    fn is_leaf(&self) -> bool;

    fn get_children(&self) -> Option<&[ANode]>;

    fn value(&self) -> &[DType];

    fn requires_grad(&self) -> bool;

    fn compute_grad(&self, _grad: &[DType], _results: &mut [Vec<DType>]) { }

}

#[derive(Clone)]
pub struct ANode(Arc<dyn Node>);

impl ANode {
    fn new(n: Arc<dyn Node>) -> Self {
        ANode(n)
    }

    pub fn dot(&self, other: &ANode) -> ANode {
        SumVec::new(self * other)
    }
}


impl Deref for ANode {
    type Target = Arc<dyn Node>;

    fn deref(&self) -> &Self::Target {
        &self.0
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


