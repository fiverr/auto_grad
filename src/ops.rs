use std::sync::Arc;

use crate::*;
use crate::vecops::{add, iadd, sub, isub, mul, imul, div};
use crate::pool::{MPVec,allocate_vec};

enum Data {
    Owned(Vec<DType>),
    Shared(Arc<Vec<DType>>),
    Pooled(MPVec)
}

struct Computation {
    value: Data
}

impl Computation {
    fn new(value: Vec<DType>) -> Self {
        Computation { value: Data::Owned(value) }
    }

    fn shared(value: Arc<Vec<DType>>) -> Self {
        Computation { value: Data::Shared(value) }
    }

    fn pooled(value: MPVec) -> Self {
        Computation { value: Data::Pooled(value) }
    }

    fn get(&self) -> &[DType] {
        match &self.value {
            Data::Owned(v) => &v,
            Data::Shared(v) => &v,
            Data::Pooled(v) => v.as_ref().as_slice()
        }
    }
}

pub struct Variable(NodeIdx, Computation);

impl Variable {
    pub fn new(value: Vec<DType>) -> ANode {
        let v = Variable(NodeIdx::new(), Computation::new(value));
        ANode::new(Arc::new(v))
    }

    pub fn scalar(value: DType) -> ANode {
        Variable::new(vec![value])
    }
    
    pub fn shared(value: Arc<Vec<DType>>) -> ANode {
        let v = Variable(NodeIdx::new(), Computation::shared(value));
        ANode::new(Arc::new(v))
    }

}

impl Node for Variable {
    fn get_id(&self) -> NodeIdx { self.0.clone() }

    fn is_leaf(&self) -> bool { true }

    fn value(&self) -> &[DType] {
        &self.1.get()
    }

    fn get_children(&self) -> Option<&[ANode]> { None }

    fn requires_grad(&self) -> bool { true }

    fn compute_grad(&self, _grad: &[DType], _child_grads: &mut [Vec<DType>]) {
        // Pass
    }
}

pub struct Constant(NodeIdx, Computation);

impl Constant {
    pub fn new(value: Vec<DType>) -> ANode {
        let c = Constant(NodeIdx::new(), Computation::new(value));
        ANode::new(Arc::new(c))
    }

    pub fn scalar(value: DType) -> ANode {
        let mut v = allocate_vec(1);
        v.as_mut()[0] = value;
        let c = Constant(NodeIdx::new(), Computation::pooled(v));
        ANode::new(Arc::new(c))
    }

}

impl Node for Constant {
    fn get_id(&self) -> NodeIdx { self.0.clone() }

    fn get_children(&self) -> Option<&[ANode]> { None }

    fn is_leaf(&self) -> bool { true }

    fn value(&self) -> &[DType] {
        &self.1.get()
    }

    fn requires_grad(&self) -> bool { false }
}

struct Broadcast<'a> {
    vec: &'a [DType],
    remaining: usize,
    len: usize
}

impl <'a> Broadcast<'a> {
    fn new<'b>(vec: &'a [DType], other: &'b [DType]) -> Self {
        Broadcast::sized(vec, other.len())
    }

    fn sized(vec: &'a [DType], other: usize) -> Self {
        if vec.len() == 1 || vec.len() == other {
            Broadcast { vec, remaining: other, len: other }
        } else if other == 1 {
            Broadcast { vec, remaining: vec.len(), len: vec.len() }
        } else {
            panic!("Cannot broadcast values!");
        }
    }

    fn from_pair<'b>(left: &'a [DType], right: &'b [DType]) -> (Self, Broadcast<'b>) {
        (Broadcast::new(left, right), Broadcast::new(right, left))
    }
}

impl <'a> Iterator for Broadcast<'a> {
    type Item = &'a DType;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            None
        } else {
            let v_len = self.vec.len();
            if v_len > 1 {
                let idx = v_len - self.remaining;
                self.remaining -= 1;
                Some(&self.vec[idx])
            } else {
                self.remaining -= 1;
                Some(&self.vec[0])
            }
        }
    }
}

struct Updater<'a> {
    data: &'a mut [DType],
    cur_idx: usize,
    max_size: usize
}

impl <'a> Updater<'a> {
    fn new(data: &'a mut [DType], max_size: usize) -> Self {
        let v_len = data.len();
        if v_len == max_size || v_len == 1 {
            Updater { data, cur_idx: 0, max_size }
        } else {
            panic!("Cannot broadcast values!");
        }
    }

    fn add(&mut self, v: DType) {
        if self.data.len() == 1 {
            self.data[0] += v;
        } else {
            self.data[self.cur_idx] += v;
            self.cur_idx += 1;
        }
    }
}

pub(crate) struct AddN(NodeIdx, Vec<ANode>, Computation);

impl AddN {
    pub(crate) fn new(left: ANode, right: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = AddN::compute(&left, &right);
        let node = AddN(idx, vec![left, right], Computation::pooled(value));
        ANode::new(Arc::new(node))
    }

    fn compute(left: &ANode, right: &ANode) -> MPVec {
        let (lv, rv) = Broadcast::from_pair(left.value(), right.value());
        let mut out = allocate_vec(lv.len);
        out.iter_mut().zip(lv.zip(rv)).for_each(|(oi, (lvi, rvi))| {
            *oi = lvi + rvi
        });
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
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [Vec<DType>]) {
        // f(x,y) = x - y
        // df(x,y)/dx = 1
        // df(x,y)/dy = 1
        for out in child_grads.iter_mut() {
            let it  = Broadcast::sized(grad, out.len());
            let mut agg = Updater::new(out, grad.len());
            it.for_each(|gi| agg.add(*gi));
        }
    }

}

pub(crate) struct Subtract(NodeIdx, Vec<ANode>, Computation);

impl Subtract {
    pub(crate) fn new(left: ANode, right: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Subtract::compute(&left, &right);
        let node = Subtract(idx, vec![left, right], Computation::pooled(value));
        ANode::new(Arc::new(node))
    }

    fn compute(left: &ANode, right: &ANode) -> MPVec {
        let (lv, rv) = Broadcast::from_pair(left.value(), right.value());
        let mut out = allocate_vec(lv.len);
        out.iter_mut().zip(lv.zip(rv)).for_each(|(oi, (lvi, rvi))| {
            *oi = lvi - rvi
        });
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
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [Vec<DType>]) {
        // f(x,y) = x - y
        // df(x,y)/dx = 1
        // df(x,y)/dy = -1
        let mut out = Updater::new(&mut child_grads[0], grad.len());
        grad.iter().for_each(|gi| out.add(*gi));

        let mut out = Updater::new(&mut child_grads[1], grad.len());
        grad.iter().for_each(|gi| out.add(-*gi));
    }

}

pub(crate) struct Multiply(NodeIdx, Vec<ANode>, Computation);

impl Multiply {
    pub(crate) fn new(left: ANode, right: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Multiply::compute(&left, &right);
        let node = Multiply(idx, vec![left, right], Computation::pooled(value));
        ANode::new(Arc::new(node))
    }

    fn compute(left: &ANode, right: &ANode) -> MPVec {
        let (lv, rv) = Broadcast::from_pair(left.value(), right.value());
        let mut out = allocate_vec(lv.len);
        out.iter_mut().zip(lv.zip(rv)).for_each(|(oi, (lvi, rvi))| {
            *oi = lvi * rvi
        });
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
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [Vec<DType>]) {
        // f(x,y) = x * y
        // df(x,y)/dx = y
        // df(x,y)/dy = x
        let x = self.1[0].value();
        let y = self.1[1].value();

        // x.len() = 1, x_grad.len() = 1, y.len() == 3, y_grad.len() = 3, grad.len() = 3
        let mut ly  = Broadcast::sized(y, child_grads[0].len());
        let mut out = Updater::new(&mut child_grads[0], grad.len());
        grad.iter().zip(ly).for_each(|(gi, yi)| out.add(*gi * *yi));

        let mut lx  = Broadcast::sized(x, child_grads[1].len());
        let mut out = Updater::new(&mut child_grads[1], grad.len());
        grad.iter().zip(lx).for_each(|(gi, xi)| out.add(*gi * *xi));

    }

}

pub(crate) struct Divide(NodeIdx, Vec<ANode>, Computation);

impl Divide {
    pub(crate) fn new(left: ANode, right: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Divide::compute(&left, &right);
        let node = Divide(idx, vec![left, right], Computation::pooled(value));
        ANode::new(Arc::new(node))
    }

    fn compute(left: &ANode, right: &ANode) -> MPVec {
        let (lv, rv) = Broadcast::from_pair(left.value(), right.value());
        let mut out = allocate_vec(lv.len);
        out.iter_mut().zip(lv.zip(rv)).for_each(|(oi, (lvi, rvi))| {
            *oi = lvi / rvi
        });
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
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [Vec<DType>]) {
        // f(x,y) = x / y
        let x = self.1[0].value();
        let y = self.1[1].value();
        
        // df(x,y)/dx = 1 / y
        let mut ly  = Broadcast::sized(y, child_grads[0].len());
        let mut out = Updater::new(&mut child_grads[0], grad.len());
        grad.iter().zip(ly).for_each(|(gi, yi)| out.add(*gi / *yi));

        // df(x,y)/dy = -x / y ^ 2
        let (lx, ly) = Broadcast::from_pair(x, y);
        let mut out = Updater::new(&mut child_grads[1], lx.len);
        grad.iter().zip(lx.zip(ly)).for_each(|(gi, (xi, yi))| out.add(*gi * -*xi / yi.powf(2f32)));
    }

}

pub(crate) struct Power(NodeIdx, Vec<ANode>, Computation);

impl Power {
    pub(crate) fn new(base: ANode, exp: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Power::compute(&base, &exp);
        let node = Power(idx, vec![base, exp], Computation::pooled(value));
        ANode::new(Arc::new(node))
    }

    fn compute(left: &ANode, right: &ANode) -> MPVec {
        let (lv, rv) = Broadcast::from_pair(left.value(), right.value());
        let mut out = allocate_vec(lv.len);
        out.iter_mut().zip(lv.zip(rv)).for_each(|(oi, (lvi, rvi))| {
            *oi = lvi.powf(*rvi)
        });
        out
    }
}

impl Node for Power {
    fn get_id(&self) -> NodeIdx { self.0.clone() }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [Vec<DType>]) {
        // f(x,y) = x ^ y
        // df(x,y)/dx = y * x ^ (y - 1)
        // df(x,y)/dy = ln(y) * x ^ y
        let x = self.1[0].value();
        let y = self.1[1].value();

        // df(x,y)/dx = y * x ^ (y - 1)
        let (lx, ly) = Broadcast::from_pair(x, y);
        let mut out = Updater::new(&mut child_grads[0], lx.len);
        grad.iter().zip(lx.zip(ly)).for_each(|(gi, (xi, yi))| {
            out.add(*gi * *yi * xi.powf(*yi - 1f32));
        });
        
        // df(x,y)/dy = ln(y) * x ^ y
        let (lx, ly) = Broadcast::from_pair(x, y);
        let mut out = Updater::new(&mut child_grads[1], lx.len);
        grad.iter().zip(lx.zip(ly)).for_each(|(gi, (xi, yi))| {
            out.add(*gi * yi.ln() * xi.powf(*yi));
        });
    }

}

pub(crate) struct SumVec(NodeIdx, Vec<ANode>, Computation);

impl SumVec {
    pub(crate) fn new(vec: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = SumVec::compute(&vec);
        let node = SumVec(idx, vec![vec], Computation::pooled(value));
        ANode::new(Arc::new(node))
    }

    fn compute(left: &ANode) -> MPVec {
        let lv = left.value();
        let mut out = allocate_vec(1);
        out[0] = lv.iter().sum::<f32>();
        out
    }
}

impl Node for SumVec {
    fn get_id(&self) -> NodeIdx { self.0.clone() }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [Vec<DType>]) {
        // f(x) = x.sum()
        // df(x)/dx_1 = 1;
        for out in child_grads.iter_mut() {
            out.fill(grad[0]);
        }
    }
}

pub(crate) struct Cos(NodeIdx, Vec<ANode>, Computation);

impl Cos {
    pub(crate) fn new(vec: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Cos::compute(&vec);
        let node = Cos(idx, vec![vec], Computation::pooled(value));
        ANode::new(Arc::new(node))
    }

    fn compute(left: &ANode) -> MPVec {
        let lv = left.value();
        let mut out = allocate_vec(lv.len());
        out.iter_mut().zip(lv.iter()).for_each(|(oi, lvi)| *oi = lvi.cos());
        out
    }
}

impl Node for Cos {
    fn get_id(&self) -> NodeIdx { self.0.clone() }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [Vec<DType>]) {
        let x = self.1[0].value();
        let out = &mut child_grads[0];
        out.iter_mut().zip(grad.iter().zip(x.iter())).for_each(|(oi, (gi, xi))| {
            *oi = *gi * -xi.sin()
        });
    }
}

pub(crate) struct Sin(NodeIdx, Vec<ANode>, Computation);

impl Sin {
    pub(crate) fn new(vec: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Sin::compute(&vec);
        let node = Sin(idx, vec![vec], Computation::pooled(value));
        ANode::new(Arc::new(node))
    }

    fn compute(left: &ANode) -> MPVec {
        let lv = left.value();
        let mut out = allocate_vec(lv.len());
        out.iter_mut().zip(lv.iter()).for_each(|(oi, lvi)| *oi = lvi.sin());
        out
    }

}

impl Node for Sin {
    fn get_id(&self) -> NodeIdx { self.0.clone() }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [Vec<DType>]) {
        let x = self.1[0].value();
        let out = &mut child_grads[0];
        out.iter_mut().zip(grad.iter().zip(x.iter())).for_each(|(oi, (gi, xi))| {
            *oi = *gi * xi.cos()
        });
    }
}

pub(crate) struct Ln(NodeIdx, Vec<ANode>, Computation);

impl Ln {
    pub(crate) fn new(vec: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Ln::compute(&vec);
        let node = Ln(idx, vec![vec], Computation::pooled(value));
        ANode::new(Arc::new(node))
    }

    fn compute(left: &ANode) -> MPVec {
        let lv = left.value();
        let mut out = allocate_vec(lv.len());
        out.iter_mut().zip(lv.iter()).for_each(|(oi, lvi)| *oi = lvi.ln());
        out
    }
}

impl Node for Ln {
    fn get_id(&self) -> NodeIdx { self.0.clone() }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [Vec<DType>]) {
        let x = self.1[0].value();
        let out = &mut child_grads[0];
        out.iter_mut().zip(grad.iter().zip(x.iter())).for_each(|(oi, (gi, xi))| {
            *oi = *gi / *xi
        });
    }
}

pub(crate) struct Exp(NodeIdx, Vec<ANode>, Computation);

impl Exp {
    pub(crate) fn new(vec: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Exp::compute(&vec);
        let node = Exp(idx, vec![vec], Computation::pooled(value));
        ANode::new(Arc::new(node))
    }

    fn compute(left: &ANode) -> MPVec {
        let lv = left.value();
        let mut out = allocate_vec(lv.len());
        out.iter_mut().zip(lv.iter()).for_each(|(oi, lvi)| *oi = lvi.exp());
        out
    }

}

impl Node for Exp {
    fn get_id(&self) -> NodeIdx { self.0.clone() }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [Vec<DType>]) {
        let x = self.value();
        let mut out = &mut child_grads[0];
        out.clone_from_slice(x);
        imul(&mut out, grad);
    }
}

pub(crate) struct Negate(NodeIdx, Vec<ANode>, Computation);

impl Negate {
    pub(crate) fn new(vec: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Negate::compute(&vec);
        let node = Negate(idx, vec![vec], Computation::pooled(value));
        ANode::new(Arc::new(node))
    }

    fn compute(left: &ANode) -> MPVec {
        let lv = left.value();
        let mut out = allocate_vec(lv.len());
        out.iter_mut().zip(lv.iter()).for_each(|(oi, lvi)| *oi = -lvi);
        out
    }

}

impl Node for Negate {
    fn get_id(&self) -> NodeIdx { self.0.clone() }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [Vec<DType>]) {
        child_grads[0].iter_mut().zip(grad.iter()).for_each(|(oi, gi)| {
            *oi = -*gi;
        });
    }
}

pub(crate) struct BulkSum(NodeIdx, Vec<ANode>, Computation);

impl BulkSum {
    pub(crate) fn new(vecs: impl Iterator<Item=ANode>) -> ANode {
        let idx = NodeIdx::new();
        let children: Vec<_> = vecs.collect();
        let value = BulkSum::compute(&children);
        let node  = BulkSum(idx, children, Computation::pooled(value));
        ANode::new(Arc::new(node))
    }

    fn compute(xs: &[ANode]) -> MPVec {
        let mut agg = allocate_vec(xs[0].value().len());
        for x in xs {
            iadd(&mut agg, x.value());
        }
        agg
    }
}

impl Node for BulkSum {
    fn get_id(&self) -> NodeIdx { self.0.clone() }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [Vec<DType>]) {
        // Just the gradient for each, easy peasy
        let x = self.value();
        for out in child_grads.iter_mut() {
            out.clone_from_slice(grad);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    #[test]
    fn test_add() {
        let x = Variable::new(vec![0., 1.]);
        let y = Variable::new(vec![2., 3.]);
        let res = AddN::new(x, y);
        assert_eq!(res.value(), &[2., 4.]);
    }

    #[test]
    fn test_add_scalar() {
        let x = Variable::new(vec![0., 1.]);
        let y = Variable::new(vec![2.]);
        let res = &x + &y;
        assert_eq!(res.value(), &[2., 3.]);

        let mut graph = Graph::new();
        graph.backward(&res);

        let res = graph.get_grad(&x).unwrap();
        assert_eq!(res, &[1., 1.]);
        let res = graph.get_grad(&y).unwrap();
        assert_eq!(res, &[2.]);
    }

    #[test]
    fn test_sub() {
        let x = Variable::new(vec![0., 1.]);
        let y = Variable::new(vec![2., 3.]);
        let res = Subtract::new(x, y);
        assert_eq!(res.value(), &[-2., -2.]);
    }

    #[test]
    fn test_sub_scalar() {
        let x = Variable::new(vec![0., 1.]);
        let y = Variable::scalar(2f32);
        let res = &x - &y;
        assert_eq!(res.value(), &[-2., -1.]);

        let mut graph = Graph::new();
        graph.backward(&res);

        let x_grad = graph.get_grad(&x).unwrap();
        let y_grad = graph.get_grad(&y).unwrap();
        assert_eq!(x_grad, &[1., 1.]);
        assert_eq!(y_grad, &[-2.]);

    }

    #[test]
    fn test_mul() {
        let x = Variable::new(vec![0., 1.]);
        let y = Variable::new(vec![2., 3.]);
        let res = Multiply::new(x, y);
        assert_eq!(res.value(), &[0., 3.]);
    }

    #[test]
    fn test_mul_scalar() {
        let x = Variable::new(vec![1., 2.]);
        let y = Variable::scalar(3f32);
        let res = &x * &y;
        assert_eq!(res.value(), &[3., 6.]);

        let mut graph = Graph::new();
        graph.backward(&res);

        let x_grad = graph.get_grad(&x).unwrap();
        let y_grad = graph.get_grad(&y).unwrap();
        assert_eq!(x_grad, &[3., 3.]);
        assert_eq!(y_grad, &[3.]);
    }

    #[test]
    fn test_div() {
        let x = Variable::new(vec![0., 1.]);
        let y = Variable::new(vec![2., 3.]);
        let res = Divide::new(x, y);
        assert_eq!(res.value(), &[0., 1./3.]);
    }

    #[test]
    fn test_div_scalar() {
        let x = Variable::new(vec![1., 2.]);
        let y = Variable::scalar(3f32);
        let res = &x / &y;
        assert_eq!(res.value(), &[1./3., 2./3.]);

        let mut graph = Graph::new();
        graph.backward(&res);

        let x_grad = graph.get_grad(&x).unwrap();
        let y_grad = graph.get_grad(&y).unwrap();
        assert_eq!(x_grad, &[1./3., 1./3.]);
        assert_eq!(y_grad, &[-1./3.]);
    }

    #[test]
    fn test_pow() {
        let x = Variable::new(vec![0., 1., 2.]);
        let y = Variable::new(vec![2., 3., 3.]);
        let res = Power::new(x, y);
        assert_eq!(res.value(), &[0., 1., 8.]);
    }

    #[test]
    fn test_pow_scalar() {
        let x = Variable::new(vec![1., 2.]);
        let y = Variable::scalar(3f32);
        let res = (&x).pow(&y);
        assert_eq!(res.value(), &[1., 8.]);

        let mut graph = Graph::new();
        graph.backward(&res);

        let x_grad = graph.get_grad(&x).unwrap();
        let y_grad = graph.get_grad(&y).unwrap();
        assert_eq!(x_grad, &[3., 12.]);
        
        // df(x,y)/dy = ln(y) * x ^ y
        let e_y_grad = 3f32.ln() * (1f32.powf(3.) + 2f32.powf(3.));
        assert_eq!(y_grad, &[e_y_grad]);
    }

    #[test]
    fn test_exp() {
        let x = Variable::new(vec![0., 1., 2.]);
        let out = (&x).exp();
        let mut graph = Graph::new();
        graph.backward(&out);
        let grad = graph.get_grad(&x);
        assert_eq!(out.value(), &[1., 1f32.exp(), 2f32.exp()]);
    }

    #[test]
    fn test_sum() {
        let x = Variable::new(vec![0., 1., 2.]);
        let out = x.sum();
        assert_eq!(out.value(), vec![3f32]);
        let mut graph = Graph::new();

        graph.backward(&out);

        let grad = graph.get_grad(&x).unwrap();
        assert_eq!(grad, &[1f32, 1f32, 1f32]);
    }

    #[test]
    fn test_neg_exp() {
        let x = Variable::new(vec![0., 1., 2.]);
        let nx = -&x;
        let enx = nx.exp();
        let out = enx;
        let mut graph = Graph::new();
        graph.backward(&out);

        let grad = graph.get_grad(&x).unwrap();
        assert_eq!(grad, &[-1., -(-1f32).exp(), -(-2f32).exp()]);
    }

    #[test]
    fn test_backward_pass_simple1() {
        // 2x
        // df/dx = 2
        let x = Variable::new(vec![0f32]);
        let x2 = Multiply::new(x.clone(), Constant::scalar(2f32));

        let mut graph = Graph::new();
        graph.backward(&x2);
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
        let x_grad = graph.get_grad(&x);
        let y_grad = graph.get_grad(&y);

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
        assert_eq!(Some(&vec![5f32]), x_grad);
    }


    #[test]
    fn test_backward_pass_simple6() {
        // 2x + 3
        // df/dx = 2
        let x = Variable::new(vec![0f32]);
        let x2 = Multiply::new(x.clone(), Constant::scalar(2f32));
        let x2_3 = AddN::new(x2, Constant::scalar(3f32));

        let mut graph = Graph::new();
        graph.backward(&x2_3);
        let x_grad = graph.get_grad(&x);
        assert_eq!(Some(&vec![2f32]), x_grad);
    }

    #[test]
    fn test_backward_pass_simple7() {
        // dot(x, y)
        let x = Variable::new(vec![1f32, 2f32, 3f32]);
        let y = Variable::new(vec![0f32, 2f32, 4f32]);
        let x2 = Multiply::new(x.clone(), y.clone());
        let ret = SumVec::new(x2);

        assert_eq!(ret.value(), vec![16f32]);
        let mut graph = Graph::new();
        graph.backward(&ret);
        let x_grad = graph.get_grad(&x);
        assert_eq!(Some(&vec![0f32, 2f32, 4f32]), x_grad);
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

        let x2_grad = graph.get_grad(&x2);
        let x_grad = graph.get_grad(&x);
        assert_eq!(Some(&vec![4f32]), x_grad);
    }

    #[test]
    fn test_composition() {
        // (x+2) ^ 2 
        let x      = Variable::new(vec![0f32]);
        let res = (&x + 2f32).pow(2f32);
        assert_eq!(res.value(), vec![4f32]);

        let mut graph = Graph::new();
        graph.backward(&res);

        let x_grad = graph.get_grad(&x);
        assert_eq!(Some(&vec![4f32]), x_grad);
    }

    #[test]
    fn test_sigmoid_denom() {
        // e ^ -x
        let x      = Variable::new(vec![1f32]);
        let res = &(-&x).exp();
        assert_eq!(res.value(), vec![(-1f32).exp()]);

        let mut graph = Graph::new();
        graph.backward(&res);

        let x_grad = graph.get_grad(&x);
        let x_0 = res.value()[0];
        let expected = -(-1f32).exp();
        assert_eq!(Some(&vec![expected]), x_grad);
    }

    fn sigmoid(x: &ANode) -> ANode {
        1f32 / ((-x).exp() + 1f32)
    }

    #[test]
    fn test_logistic() {
        // 1 / (1 + e ^ -x)
        let x = Variable::new(vec![0f32]);
        let res = sigmoid(&x);
        assert_eq!(res.value(), vec![0.5]);

        let mut graph = Graph::new();
        graph.backward(&res);

        let x_grad = graph.get_grad(&x);
        let sigma_trick = res.value()[0] * (1f32 - res.value()[0]);
        assert_eq!(Some(&vec![sigma_trick]), x_grad);
    }

    #[test]
    fn test_simple_sgd() {
        let y = Constant::new(vec![3f32,-4f32]);
        let mut v = vec![0f32, 0f32]; 
        let mut graph = Graph::new();
        let alpha = 3e-1;
        for _ in 0..20 {
            let x = Variable::new(v.clone());
            let c = Constant::scalar(2f32);
            let y1 = &x - &y;
            let y2 = (&y1).pow(&c);
            let err = (&y2).sum();
            println!("Error: {}", err.value()[0]);
            graph.zero_grads();
            graph.backward(&err);
            let x_grad = graph.get_grad(&x).unwrap();
            
            // SGD!
            v.iter_mut().zip(x_grad.iter()).for_each(|(vi, gi)| {
                *vi -= alpha * *gi;
            });
            println!("X: {:?}", v);
        }

        assert!((v[0] - y.value()[0]).abs() < 1e-5);
        assert!((v[1] - y.value()[1]).abs() < 1e-5);
    }

    #[test]
    fn test_updateable() {
        let mut v = Arc::new(vec![0f32, 0f32]);
        let mut graph = Graph::new();
        let grad = {
            let x = Variable::shared(v.clone());
            let res = (&x + 3f32).pow(2f32) + 3f32;
            graph.backward(&res);
            graph.get_grad(&x)
        };
        let v = Arc::get_mut(&mut v).unwrap();
        assert_eq!(v, &mut [0f32, 0f32]);
    }

}
