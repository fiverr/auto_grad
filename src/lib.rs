//#![feature(trace_macros)]

//trace_macros!(true);

mod graph;
mod vecops;
mod ops;
mod pool;

pub use graph::Graph;
pub use ops::{Variable,Constant};
pub use pool::{clear_pool, use_shared_pool, MPVec};

use std::sync::atomic::{AtomicUsize, Ordering};
use std::rc::Rc;
use std::ops::{Add,Sub,Mul,Div,Deref,Neg};

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

    #[inline]
    fn value(&self) -> &[DType];

    fn requires_grad(&self) -> bool;

    fn compute_grad(&self, _grad: &[DType], _results: &mut [MPVec]) { }

}

#[derive(Clone)]
pub struct ANode(Rc<dyn Node>);

impl ANode {
    fn new(n: Rc<dyn Node>) -> Self {
        ANode(n)
    }

    pub fn dot(&self, other: &ANode) -> ANode {
        SumVec::new(self * other)
    }

    pub fn ln(&self) -> ANode {
        Ln::new(self.clone())
    }

    pub fn cos(&self) -> ANode {
        Cos::new(self.clone())
    }

    pub fn sin(&self) -> ANode {
        Cos::new(self.clone())
    }

    pub fn tanh(&self) -> ANode {
        Tanh::new(self.clone())
    }

    pub fn exp(&self) -> ANode {
        Exp::new(self.clone())
    }

    pub fn sum(&self) -> ANode {
        SumVec::new(self.clone())
    }

    pub fn slice(&self, start: usize, len: usize) -> ANode {
        Slice::new(self.clone(), start, len)
    }
}

trait FromConstant {
    fn convert(self) -> ANode; 
}

impl FromConstant for f32 {
    fn convert(self) -> ANode {
        Constant::scalar(self)
    }
}

impl FromConstant for Vec<f32> {
    fn convert(self) -> ANode {
        Constant::new(self)
    }
}


impl Deref for ANode {
    type Target = Rc<dyn Node>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

macro_rules! forward_ref_binop {
    (impl $imp:ident, $method:ident for $t:ty, $u:ty) => {
        impl<'a> $imp<$u> for &'a $t {
            type Output = <$t as $imp<$u>>::Output;

            #[inline]
            fn $method(self, other: $u) -> <$t as $imp<$u>>::Output {
                $imp::$method(self.clone(), other)
            }
        }

        impl $imp<&$u> for $t {
            type Output = <$t as $imp<$u>>::Output;

            #[inline]
            fn $method(self, other: &$u) -> <$t as $imp<$u>>::Output {
                $imp::$method(self, other.clone())
            }
        }

        impl $imp<&$u> for &$t {
            type Output = <$t as $imp<$u>>::Output;

            #[inline]
            fn $method(self, other: &$u) -> <$t as $imp<$u>>::Output {
                $imp::$method(self.clone(), other.clone())
            }
        }


    };
}

macro_rules! convert_binops {
    (impl $imp:ident, $method:ident for $t:ty, $u:ty) => {
        impl <C: FromConstant> $imp<C> for $t {
            type Output = <$t as $imp<$u>>::Output;
            //type Output = ANode;
            fn $method(self, other: C) -> <$t as $imp<$u>>::Output {
                $imp::$method(self, other.convert())
            }
        }

        impl <C: FromConstant> $imp<C> for &$t {
            type Output = <$t as $imp<$u>>::Output;
            //type Output = ANode;
            fn $method(self, other: C) -> <$t as $imp<$u>>::Output {
                $imp::$method(self.clone(), other.convert())
            }
        }
    };
}

impl Add for ANode {
    type Output = ANode;
    fn add(self, rhs: ANode) -> Self::Output {
        AddN::new(self, rhs)
    }
}

impl Add<ANode> for f32 {
    type Output = ANode;
    fn add(self, rhs: ANode) -> Self::Output {
        rhs + self.convert()
    }
}

impl Add<ANode> for Vec<f32> {
    type Output = ANode;
    fn add(self, rhs: ANode) -> Self::Output {
        rhs + self.convert()
    }
}

convert_binops! { impl Add, add for ANode, ANode }
forward_ref_binop! { impl Add, add for ANode, ANode }
forward_ref_binop! { impl Add, add for f32, ANode }
forward_ref_binop! { impl Add, add for Vec<f32>, ANode }

impl Sub for ANode {
    type Output = ANode;
    fn sub(self, rhs: ANode) -> Self::Output {
        Subtract::new(self, rhs)
    }
}

impl Sub<ANode> for f32 {
    type Output = ANode;
    fn sub(self, rhs: ANode) -> Self::Output {
        self.convert() - rhs
    }
}

impl Sub<ANode> for Vec<f32> {
    type Output = ANode;
    fn sub(self, rhs: ANode) -> Self::Output {
        self.convert() - rhs
    }
}

convert_binops! { impl Sub, sub for ANode, ANode }
forward_ref_binop! { impl Sub, sub for ANode, ANode }
forward_ref_binop! { impl Sub, sub for f32, ANode }
forward_ref_binop! { impl Sub, sub for Vec<f32>, ANode }

impl Mul for ANode {
    type Output = ANode;
    fn mul(self, rhs: ANode) -> Self::Output {
        Multiply::new(self, rhs)
    }
}

impl Mul<ANode> for f32 {
    type Output = ANode;
    fn mul(self, rhs: ANode) -> Self::Output {
        self.convert() * rhs
    }
}

impl Mul<ANode> for Vec<f32> {
    type Output = ANode;
    fn mul(self, rhs: ANode) -> Self::Output {
        self.convert() * rhs
    }
}

convert_binops! {    impl Mul, mul for ANode, ANode }
forward_ref_binop! { impl Mul, mul for ANode, ANode }
forward_ref_binop! { impl Mul, mul for f32, ANode }
forward_ref_binop! { impl Mul, mul for Vec<f32>, ANode }

impl Div for ANode {
    type Output = ANode;
    fn div(self, rhs: ANode) -> Self::Output {
        Divide::new(self, rhs)
    }
}

impl Div<ANode> for f32 {
    type Output = ANode;
    fn div(self, rhs: ANode) -> Self::Output {
        self.convert() / rhs
    }
}

impl Div<ANode> for Vec<f32> {
    type Output = ANode;
    fn div(self, rhs: ANode) -> Self::Output {
        self.convert() / rhs
    }
}

convert_binops!    { impl Div, div for ANode, ANode }
forward_ref_binop! { impl Div, div for ANode, ANode }
forward_ref_binop! { impl Div, div for f32, ANode }
forward_ref_binop! { impl Div, div for Vec<f32>, ANode }

impl Neg for ANode {
    type Output = ANode;
    fn neg(self) -> Self::Output {
        Negate::new(self)
    }
}

impl Neg for &ANode {
    type Output = ANode;
    fn neg(self) -> Self::Output {
        -self.clone()
    }
}

pub trait Pow<Rhs=Self> {
    type Output;
    fn pow(self, rhs: Rhs) -> Self::Output;
}

impl Pow for ANode {
    type Output = ANode;
    fn pow(self, rhs: ANode) -> Self::Output {
        Power::new(self, rhs)
    }
}

impl Pow<ANode> for f32 {
    type Output = ANode;
    fn pow(self, rhs: ANode) -> Self::Output {
        self.convert().pow(rhs)
    }
}

impl Pow<ANode> for Vec<f32> {
    type Output = ANode;
    fn pow(self, rhs: ANode) -> Self::Output {
        self.convert().pow(rhs)
    }
}

convert_binops!    { impl Pow, pow for ANode, ANode }
forward_ref_binop! { impl Pow, pow for ANode, ANode }
forward_ref_binop! { impl Pow, pow for f32, ANode }
forward_ref_binop! { impl Pow, pow for Vec<f32>, ANode }

pub trait BulkOps {
    fn sum_all(self) -> ANode;
    fn concat(self) -> ANode;
}

impl BulkOps for Vec<ANode> {
    fn sum_all(self) -> ANode {
        BulkSum::new(self.into_iter())
    }

    fn concat(self) -> ANode {
        Concat::new(self)
    }

}

impl BulkOps for Vec<&ANode> {
    fn sum_all(self) -> ANode {
        BulkSum::new(self.into_iter().cloned())
    }

    fn concat(self) -> ANode {
        let n = self.into_iter().map(|n| n.clone()).collect();
        Concat::new(n)
    }

}

pub trait MaximumOps<Rhs=Self> {
    type Output;
    fn maximum(self, rhs: Rhs) -> Self::Output;

}

impl MaximumOps for ANode {
    type Output = ANode;
    fn maximum(self, rhs: ANode) -> Self::Output {
        Maximum::new(self, rhs)
    }
}

convert_binops!    { impl MaximumOps, maximum for ANode, ANode }
forward_ref_binop! { impl MaximumOps, maximum for ANode, ANode }

pub trait MinimumOps<Rhs=Self> {
    type Output;
    fn minimum(self, rhs: Rhs) -> Self::Output;

}

impl MinimumOps for ANode {
    type Output = ANode;
    fn minimum(self, rhs: ANode) -> Self::Output {
        Minimum::new(self, rhs)
    }
}

convert_binops!    { impl MinimumOps, minimum for ANode, ANode }
forward_ref_binop! { impl MinimumOps, minimum for ANode, ANode }

