use std::rc::Rc;

use crate::*;
use crate::vecops::*;
use crate::pool::{MPVec,allocate_vec};

enum Data {
    Owned(Vec<DType>),
    Shared(Rc<Vec<DType>>),
    Pooled(MPVec)
}

struct Computation {
    value: Data
}

impl Computation {
    fn new(value: Vec<DType>) -> Self {
        Computation { value: Data::Owned(value) }
    }

    fn shared(value: Rc<Vec<DType>>) -> Self {
       Computation { value: Data::Shared(value) }
    }

    fn pooled(value: MPVec) -> Self {
        Computation { value: Data::Pooled(value) }
    }

    #[inline]
    fn get(&self) -> &[DType] {
        match &self.value {
            Data::Owned(v) => &v,
            Data::Shared(v) => &v,
            Data::Pooled(v) => v.as_ref().as_slice()
        }
    }
}

pub struct RequiresGrad(Rc<dyn Node>);

impl RequiresGrad {
    pub fn new(n: Rc<dyn Node>) -> Self {
        RequiresGrad(n)
    }
}

impl Node for RequiresGrad {

    fn op_name(&self) -> &str { "RequiresGrad" }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0.get_id() }

    #[inline]
    fn is_leaf(&self) -> bool { self.0.is_leaf() }

    #[inline]
    fn value(&self) -> &[DType] {
        &self.0.value()
    }

    #[inline]
    fn get_children(&self) -> Option<&[ANode]> { self.0.get_children() }

    #[inline]
    fn requires_grad(&self) -> bool { true }

    #[inline]
    fn compute_grad(&self, grad: &[DType], child_grads: &mut [&mut [DType]]) {
        self.0.compute_grad(grad, child_grads)
    }
}

pub struct Named(ANode, String);

impl Named {
    pub fn new(node: ANode, name: String) -> ANode {
        let v = Named(node, name);
        ANode::new(Rc::new(v))
    }
}

impl Node for Named {

    fn op_name(&self) -> &str { &self.1 }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0.get_id() }

    #[inline]
    fn is_leaf(&self) -> bool { self.0.is_leaf() }

    #[inline]
    fn value(&self) -> &[DType] { self.0.value() }

    #[inline]
    fn get_children(&self) -> Option<&[ANode]> { self.0.get_children() }

    #[inline]
    fn requires_grad(&self) -> bool { self.0.requires_grad() }

    #[inline]
    fn compute_grad(&self, grad: &[DType], child_grads: &mut [&mut [DType]]) {
        self.0.compute_grad(grad, child_grads)
    }
}

pub struct Variable(NodeIdx, Computation);

impl Variable {
    pub fn new(value: Vec<DType>) -> ANode {
        let v = Variable(NodeIdx::new(), Computation::new(value));
        ANode::new(Rc::new(v))
    }

    pub fn scalar(value: DType) -> ANode {
        Variable::new(vec![value])
    }
    
    pub fn shared(value: Rc<Vec<DType>>) -> ANode {
        let v = Variable(NodeIdx::new(), Computation::shared(value));
        ANode::new(Rc::new(v))
    }

    pub fn pooled(value: &[DType]) -> ANode {
        let mut mpv = allocate_vec(value.len());
        mpv.clone_from_slice(value);
        let v = Variable(NodeIdx::new(), Computation::pooled(mpv));
        ANode::new(Rc::new(v))
    }

}

impl Node for Variable {
    fn op_name(&self) -> &str { "Variable" }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0 }

    #[inline]
    fn is_leaf(&self) -> bool { true }

    #[inline]
    fn value(&self) -> &[DType] {
        &self.1.get()
    }

    #[inline]
    fn get_children(&self) -> Option<&[ANode]> { None }

    #[inline]
    fn requires_grad(&self) -> bool { true }

    #[inline]
    fn compute_grad(&self, _grad: &[DType], _child_grads: &mut [&mut [DType]]) {
        // Pass
    }
}

pub struct Constant(NodeIdx, Computation);

impl Constant {
    pub fn new(value: Vec<DType>) -> ANode {
        let c = Constant(NodeIdx::new(), Computation::new(value));
        ANode::new(Rc::new(c))
    }

    pub fn scalar(value: DType) -> ANode {
        let mut v = allocate_vec(1);
        v.as_mut()[0] = value;
        let c = Constant(NodeIdx::new(), Computation::pooled(v));
        ANode::new(Rc::new(c))
    }

}

impl Node for Constant {
    fn op_name(&self) -> &str { "Constant" }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0 }

    #[inline]
    fn get_children(&self) -> Option<&[ANode]> { None }

    #[inline]
    fn is_leaf(&self) -> bool { true }

    #[inline]
    fn value(&self) -> &[DType] {
        &self.1.get()
    }

    #[inline]
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

    fn get_size(left: &[DType], right: &[DType]) -> usize {
        let l_len = left.len();
        let r_len = right.len();
        match (l_len, r_len) {
            (l, r) if l == 0 || r == 0 => {
                panic!("Input vector is zero!");
            },
            (l, r) if l == r => l_len,
            (1, r) => r,
            (l, 1) => l,
            (_, _) => {
                panic!("Input vectors mismatched is zero!");
            }
        }
    }

    fn allocate_out(left: &[DType], right: &[DType]) -> MPVec {
        allocate_vec(Broadcast::get_size(left, right))
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
    cur_idx: usize
}

impl <'a> Updater<'a> {
    fn new(data: &'a mut [DType], max_size: usize) -> Self {
        let v_len = data.len();
        if v_len == max_size || v_len == 1 {
            Updater { data, cur_idx: 0 }
        } else {
            panic!("Cannot broadcast values!");
        }
    }

    #[inline]
    fn add(&mut self, v: DType) {
        if self.data.len() == 1 {
            unsafe {
                *self.data.get_unchecked_mut(0) += v;
            }
        } else {

            unsafe {
                *self.data.get_unchecked_mut(self.cur_idx) += v;
            }
            self.cur_idx += 1;
        }
    }
}

macro_rules! run_unary_op {
    ($left:tt, $out:tt, $func:expr) => {
        let left_len = $left.len();
        let out_len = $out.len();
        if left_len == out_len {
            $func(ArrayInput($left), ArrayOutput($out));
        } else if left_len == 1 {
            $func(BroadcastInput($left, out_len), ArrayOutput($out));
        } else if out_len == 1 {
            $func(ArrayInput($left), BroadcastOutput($out, left_len));
        } else {
            panic!("Left length: {}, Output Length: {}", left_len, out_len);
        }
    }
}

macro_rules! to_output {
    ($out:tt, $len:expr, $body:expr) => {
        if $out.len() == $len {
            $body(ArrayOutput($out));
        } else if $out.len() == 1 {
            $body(BroadcastOutput($out, $len));
        } else {
            panic!("Output is incompatible with input");
        }
    }
}

macro_rules! run_binary_op {
    ($left:tt, $right:tt, $out:tt, $func:expr) => {
        let left_len = $left.len();
        let right_len = $right.len();
        let max_len = left_len.max(right_len);
        to_output!($out, max_len, |output| {
            if left_len == right_len {
                $func(ArrayInput($left), ArrayInput($right), output);
            } else if left_len == 1 {
                $func(BroadcastInput($left, right_len), ArrayInput($right), output);
            } else if right_len == 1 {
                $func(ArrayInput($left), BroadcastInput($right, left_len), output);
            } else {
                panic!("Left length: {}, Right Length: {}", left_len, right_len);
            }
        });
    }
}

macro_rules! run_trinary_op {
    ($x:tt, $y:tt, $z:tt, $out:tt, $func:expr) => {
        let x_len = $x.len();
        let y_len = $y.len();
        let z_len = $z.len();
        let max_len = x_len.max(y_len).max(z_len);
        to_output!($out, max_len, |output| {
            if x_len > 0 && x_len == y_len && y_len == z_len {
                $func(ArrayInput($x), ArrayInput($y), ArrayInput($z), output);
            } else {
                match (x_len, y_len, z_len) {
                    (1, a, b) if a == b => {
                        $func(BroadcastInput($x, max_len), ArrayInput($y), ArrayInput($z), output)
                    },
                    (a, 1, b) if a == b => {
                        $func(ArrayInput($x), BroadcastInput($y, max_len), ArrayInput($z), output)
                    },
                    (a, b, 1) if a == b => {
                        $func(ArrayInput($x), ArrayInput($y), BroadcastInput($z, max_len), output)
                    },
                    _ => panic!("x: {}, y: {}, z: {}", x_len, y_len, z_len)
                }
            }
        });
    }
}

pub(crate) struct AddN(NodeIdx, [ANode; 2], Computation);

impl AddN {
    pub(crate) fn new(left: ANode, right: ANode) -> ANode {
        if left.get_id() == right.get_id() {
            Multiply::new(Constant::scalar(2f32), left)
        } else {
            let idx = NodeIdx::new();
            let value = AddN::compute(&left, &right);
            let node = AddN(idx, [left, right], Computation::pooled(value));
            ANode::new(Rc::new(node))
        }
    }

    fn compute(left: &ANode, right: &ANode) -> MPVec {
        let x = left.value();
        let y = right.value();
        let mut out = Broadcast::allocate_out(x, y);
        let o = &mut out;
        run_binary_op!(x, y, o, simd_add);
        out
    }
}

impl Node for AddN {
    fn op_name(&self) -> &str { "Add" }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0 }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [&mut [DType]]) {
        // f(x,y) = x + y
        // df(x,y)/dx = 1
        // df(x,y)/dy = 1
        for out in child_grads.iter_mut() {
            run_unary_op!(grad, out, simd_iadd); 
        }
    }

}

pub(crate) struct Subtract(NodeIdx, [ANode;2], Computation);

impl Subtract {
    pub(crate) fn new(left: ANode, right: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Subtract::compute(&left, &right);
        let node = Subtract(idx, [left, right], Computation::pooled(value));
        ANode::new(Rc::new(node))
    }

    fn compute(left: &ANode, right: &ANode) -> MPVec {
        let x = left.value();
        let y = right.value();
        let mut out = Broadcast::allocate_out(x, y);
        let o = &mut out;
        run_binary_op!(x, y, o, simd_sub);
        out
    }
}

impl Node for Subtract {
    fn op_name(&self) -> &str { "Subtract" }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0 }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [&mut [DType]]) {
        // f(x,y) = x - y
        // df(x,y)/dx = 1
        let out = &mut child_grads[0];
        run_unary_op!(grad, out, simd_iadd);

        // df(x,y)/dy = -1
        let out = &mut child_grads[1];
        run_unary_op!(grad, out, grad_sub_y);
    }

}

pub(crate) struct Multiply(NodeIdx, [ANode; 2], Computation);

impl Multiply {
    pub(crate) fn new(left: ANode, right: ANode) -> ANode {
        if left.get_id() == right.get_id() {
            Pow2::new(left)
        } else {
            let idx = NodeIdx::new();
            let value = Multiply::compute(&left, &right);
            let node = Multiply(idx, [left, right], Computation::pooled(value));
            ANode::new(Rc::new(node))
        }
    }

    fn compute(left: &ANode, right: &ANode) -> MPVec {
        let (x, y) = (left.value(), right.value());
        let mut out = Broadcast::allocate_out(x, y);
        let o = &mut out;
        run_binary_op!(x, y, o, simd_mul);
        out
    }
}

impl Node for Multiply {
    fn op_name(&self) -> &str { "Multiply" }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0 }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [&mut [DType]]) {
        // f(x,y) = x * y
        // df(x,y)/dx = y
        // df(x,y)/dy = x
        let x = self.1[0].value();
        let y = self.1[1].value();

        // df(x,y)/dx = y
        let cg = &mut child_grads[0];
        run_binary_op!(grad, y, cg, simd_mul);

        // df(x,y)/dy = x
        let cg = &mut child_grads[1];
        run_binary_op!(grad, x, cg, simd_mul);
    }

}

pub(crate) struct Divide(NodeIdx, [ANode; 2], Computation);

impl Divide {
    pub(crate) fn new(left: ANode, right: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Divide::compute(&left, &right);
        let node = Divide(idx, [left, right], Computation::pooled(value));
        ANode::new(Rc::new(node))
    }

    fn compute(left: &ANode, right: &ANode) -> MPVec {
        let x = left.value();
        let y = right.value();
        let mut out = Broadcast::allocate_out(x, y);
        let o = &mut out;
        run_binary_op!(x, y, o, simd_div);
        out
    }
}

impl Node for Divide {
    fn op_name(&self) -> &str { "Divide" }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0 }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [&mut [DType]]) {
        // f(x,y) = x / y
        let x = self.1[0].value();
        let y = self.1[1].value();
        
        // df(x,y)/dx = 1 / y
        let out = &mut child_grads[0];
        run_binary_op!(grad, y, out, grad_div_x);

        let out = &mut child_grads[1];
        // df(x,y)/dy = -x / y ^ 2
        run_trinary_op!(grad, x, y, out, grad_div_y);

    }

}

pub(crate) struct Power(NodeIdx, [ANode;2], Computation);

impl Power {
    pub(crate) fn new(base: ANode, exp: ANode) -> ANode {
        if exp.is_leaf() && !exp.requires_grad() {
            let v = exp.value();
            if v == &[2f32] {
                return Pow2::new(base.clone())
            } else if v == &[0.5f32] {
                return SquareRoot::new(base)
            }
        }

        let idx = NodeIdx::new();
        let value = Power::compute(&base, &exp);
        let node = Power(idx, [base, exp], Computation::pooled(value));
        ANode::new(Rc::new(node))
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
    fn op_name(&self) -> &str { "Power" }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0 }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [&mut [DType]]) {
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

pub(crate) struct SquareRoot(NodeIdx, [ANode;1], Computation);

impl SquareRoot {
    pub(crate) fn new(base: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = SquareRoot::compute(&base);
        let node = SquareRoot(idx, [base], Computation::pooled(value));
        ANode::new(Rc::new(node))
    }

    fn compute(left: &ANode) -> MPVec {
        let lv = left.value();
        let mut out = allocate_vec(lv.len());
        out.iter_mut().zip(lv).for_each(|(oi, lvi)| {
            *oi = lvi.sqrt();
        });
        out
    }
}

impl Node for SquareRoot {
    fn op_name(&self) -> &str { "Sqrt" }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0 }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [&mut [DType]]) {
        let x = self.value();

        // df(x)/dx = (1/2) / x ^ 0.5
        child_grads[0].iter_mut().zip(grad.iter().zip(x)).for_each(|(outi, (gi, xi))| {
            *outi += *gi * 0.5 / *xi;
        });
    }

}

pub(crate) struct Pow2(NodeIdx, [ANode;1], Computation);

impl Pow2 {
    pub(crate) fn new(base: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Pow2::compute(&base);
        let node = Pow2(idx, [base], Computation::pooled(value));
        ANode::new(Rc::new(node))
    }

    fn compute(left: &ANode) -> MPVec {
        let lv = left.value();
        let mut out = allocate_vec(lv.len());
        let o = &mut out;
        run_unary_op!(lv, o, simd_pow2);
        out
    }
}

impl Node for Pow2 {
    fn op_name(&self) -> &str { "Squared" }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0 }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [&mut [DType]]) {
        let x = self.1[0].value();

        let out: &mut [f32] = &mut child_grads[0];
        run_binary_op!(grad, x, out, grad_pow2);
    }

}

pub(crate) struct SumVec(NodeIdx, [ANode; 1], Computation);

impl SumVec {
    pub(crate) fn new(vec: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = SumVec::compute(&vec);
        let node = SumVec(idx, [vec], Computation::pooled(value));
        ANode::new(Rc::new(node))
    }

    fn compute(left: &ANode) -> MPVec {
        let lv = left.value();
        let mut out = allocate_vec(1);
        out[0] = lv.iter().sum::<f32>();
        out
    }
}

impl Node for SumVec {
    fn op_name(&self) -> &str { "SumVec" }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0 }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [&mut [DType]]) {
        // f(x) = x.sum()
        // df(x)/dx_1 = 1;
        for out in child_grads.iter_mut() {
            out.fill(grad[0]);
        }
    }
}

pub(crate) struct Cos(NodeIdx, [ANode;1], Computation);

impl Cos {
    pub(crate) fn new(vec: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Cos::compute(&vec);
        let node = Cos(idx, [vec], Computation::pooled(value));
        ANode::new(Rc::new(node))
    }

    fn compute(left: &ANode) -> MPVec {
        let lv = left.value();
        let mut out = allocate_vec(lv.len());
        out.iter_mut().zip(lv.iter()).for_each(|(oi, lvi)| *oi = lvi.cos());
        out
    }
}

impl Node for Cos {
    
    fn op_name(&self) -> &str { "Cos" }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0 }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [&mut [DType]]) {
        let x = self.1[0].value();
        let out = &mut child_grads[0];
        out.iter_mut().zip(grad.iter().zip(x.iter())).for_each(|(oi, (gi, xi))| {
            *oi = *gi * -xi.sin()
        });
    }
}

pub(crate) struct Sin(NodeIdx, [ANode;1], Computation);

impl Sin {
    pub(crate) fn new(vec: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Sin::compute(&vec);
        let node = Sin(idx, [vec], Computation::pooled(value));
        ANode::new(Rc::new(node))
    }

    fn compute(left: &ANode) -> MPVec {
        let lv = left.value();
        let mut out = allocate_vec(lv.len());
        out.iter_mut().zip(lv.iter()).for_each(|(oi, lvi)| *oi = lvi.sin());
        out
    }

}

impl Node for Sin {
    fn op_name(&self) -> &str { "Sin" }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0 }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [&mut [DType]]) {
        let x = self.1[0].value();
        let out = &mut child_grads[0];
        out.iter_mut().zip(grad.iter().zip(x.iter())).for_each(|(oi, (gi, xi))| {
            *oi = *gi * xi.cos()
        });
    }
}

pub(crate) struct Tanh(NodeIdx, [ANode;1], Computation);

impl Tanh {
    pub(crate) fn new(vec: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Tanh::compute(&vec);
        let node = Tanh(idx, [vec], Computation::pooled(value));
        ANode::new(Rc::new(node))
    }

    fn compute(left: &ANode) -> MPVec {
        let lv = left.value();
        let mut out = allocate_vec(lv.len());
        out.iter_mut().zip(lv.iter())
            .for_each(|(oi, lvi)| *oi = lvi.tanh());
        out
    }

}

impl Node for Tanh {
    fn op_name(&self) -> &str { "Tanh" }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0 }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [&mut [DType]]) {
        let x = self.2.get();
        let out = &mut child_grads[0];
        out.iter_mut().zip(grad.iter().zip(x.iter())).for_each(|(oi, (gi, xi))| {
            *oi = *gi * (1f32 - xi.powf(2.))
        });
    }
}

pub(crate) struct Ln(NodeIdx, [ANode;1], Computation);

impl Ln {
    pub(crate) fn new(vec: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Ln::compute(&vec);
        let node = Ln(idx, [vec], Computation::pooled(value));
        ANode::new(Rc::new(node))
    }

    fn compute(left: &ANode) -> MPVec {
        let lv = left.value();
        let mut out = allocate_vec(lv.len());
        out.iter_mut().zip(lv.iter()).for_each(|(oi, lvi)| *oi = lvi.ln());
        out
    }
}

impl Node for Ln {
    fn op_name(&self) -> &str { "ln" }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0 }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [&mut [DType]]) {
        let x = self.1[0].value();
        let out = &mut child_grads[0];
        out.iter_mut().zip(grad.iter().zip(x.iter())).for_each(|(oi, (gi, xi))| {
            *oi = *gi / *xi
        });
    }
}

pub(crate) struct Exp(NodeIdx, [ANode;1], Computation);

impl Exp {
    pub(crate) fn new(vec: ANode, approximate: bool) -> ANode {
        let idx = NodeIdx::new();
        let value = Exp::compute(&vec, approximate);
        let node = Exp(idx, [vec], Computation::pooled(value));
        ANode::new(Rc::new(node))
    }

    fn compute(left: &ANode, approximate: bool) -> MPVec {
        let lv = left.value();
        let mut out = allocate_vec(lv.len());
        let o = &mut out;
        if approximate {
            run_unary_op!(lv, o, simd_exp);
        } else {
            out.iter_mut().zip(lv.iter()).for_each(|(oi, lvi)| *oi = lvi.exp());
        }
        out
    }

}

impl Node for Exp {
    fn op_name(&self) -> &str { "Exp" }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0 }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [&mut [DType]]) {
        let x = self.value();
        let out = &mut child_grads[0];
        run_binary_op!(grad, x, out, simd_mul);
    }
}

pub(crate) struct Negate(NodeIdx, [ANode;1], Computation);

impl Negate {
    pub(crate) fn new(vec: ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Negate::compute(&vec);
        let node = Negate(idx, [vec], Computation::pooled(value));
        ANode::new(Rc::new(node))
    }

    fn compute(left: &ANode) -> MPVec {
        let lv = left.value();
        let mut out = allocate_vec(lv.len());
        out.iter_mut().zip(lv.iter()).for_each(|(oi, lvi)| *oi = -lvi);
        out
    }

}

impl Node for Negate {
    fn op_name(&self) -> &str { "Negate" }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0 }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [&mut [DType]]) {
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
        ANode::new(Rc::new(node))
    }

    fn compute(xs: &[ANode]) -> MPVec {
        let mut out = allocate_vec(xs[0].value().len());
        let o = &mut out;
        for x in xs {
            let v = x.value();
            run_unary_op!(v, o, simd_iadd);
            //iadd(&mut agg, x.value());
        }
        out
    }
}

impl Node for BulkSum {
    fn op_name(&self) -> &str { "BulkSum" }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0 }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    #[inline]
    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [&mut [DType]]) {
        // Just the gradient for each, easy peasy
        child_grads.iter_mut().for_each(|cgi| {
            cgi.clone_from_slice(grad);
        });
    }
}

pub(crate) struct Maximum(NodeIdx, [ANode;2], Computation);

impl Maximum {
    pub(crate) fn new(left: ANode, right:ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Maximum::compute(&left, &right);
        let node  = Maximum(idx, [left, right], Computation::pooled(value));
        ANode::new(Rc::new(node))
    }

    fn compute(left: &ANode, right: &ANode) -> MPVec {
        let (lv, rv) = Broadcast::from_pair(left.value(), right.value());
        let mut out = allocate_vec(lv.len);
        out.iter_mut().zip(lv.zip(rv)).for_each(|(oi, (lvi, rvi))| {
            *oi = lvi.max(*rvi)
        });
        out
    }
}

impl Node for Maximum {
    fn op_name(&self) -> &str { "Maximum" }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0 }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [&mut [DType]]) {
        // f(x,y) = x.max(y)
        let left = self.1[0].value();
        let right = self.1[1].value();
        let (lv, rv) = Broadcast::from_pair(left, right);
        let (left_grad, right_grad) = child_grads.split_at_mut(1);
        let mut left_out = Updater::new(&mut left_grad[0], grad.len());
        let mut right_out = Updater::new(&mut right_grad[0], grad.len());
        grad.iter().zip(lv.zip(rv)).for_each(|(gi, (xi, yi))| {
            if xi >= yi {
                left_out.add(*gi);
                right_out.add(0f32);
            } else {
                right_out.add(*gi);
                left_out.add(0f32);
            }
        });
    }
}

pub(crate) struct Minimum(NodeIdx, [ANode;2], Computation);

impl Minimum {
    pub(crate) fn new(left: ANode, right:ANode) -> ANode {
        let idx = NodeIdx::new();
        let value = Minimum::compute(&left, &right);
        let node  = Minimum(idx, [left, right], Computation::pooled(value));
        ANode::new(Rc::new(node))
    }

    fn compute(left: &ANode, right: &ANode) -> MPVec {
        let (lv, rv) = Broadcast::from_pair(left.value(), right.value());
        let mut out = allocate_vec(lv.len);
        out.iter_mut().zip(lv.zip(rv)).for_each(|(oi, (lvi, rvi))| {
            *oi = lvi.min(*rvi)
        });
        out
    }
}

impl Node for Minimum {
    fn op_name(&self) -> &str { "Minimum" }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0 }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [&mut [DType]]) {
        // f(x,y) = x.max(y)
        let left = self.1[0].value();
        let right = self.1[1].value();
        let (lv, rv) = Broadcast::from_pair(left, right);
        let (left_grad, right_grad) = child_grads.split_at_mut(1);
        let mut left_out = Updater::new(&mut left_grad[0], grad.len());
        let mut right_out = Updater::new(&mut right_grad[0], grad.len());
        grad.iter().zip(lv.zip(rv)).for_each(|(gi, (xi, yi))| {
            if xi >= yi {
                right_out.add(*gi);
                left_out.add(0f32);
            } else {
                left_out.add(*gi);
                right_out.add(0f32);
            }
        });
    }
}

pub(crate) struct Concat(NodeIdx, Vec<ANode>, Computation);

impl Concat {
    pub(crate) fn new(nodes: Vec<ANode>) -> ANode {
        let idx = NodeIdx::new();
        let value = Concat::compute(&nodes);
        let node  = Concat(idx, nodes, Computation::pooled(value));
        ANode::new(Rc::new(node))
    }

    fn compute(nodes: &[ANode]) -> MPVec {
        let size = nodes.iter().map(|n| n.value().len()).sum::<usize>();
        let mut out = allocate_vec(size);
        let mut i = 0;
        for node in nodes {
            for vi in node.value() {
                out[i] = *vi;
                i += 1;
            }
        }
        out
    }
}

impl Node for Concat {
    fn op_name(&self) -> &str { "Concat" }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0 }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        &self.2.get()
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [&mut [DType]]) {
        let mut i = 0;
        for cg in child_grads.iter_mut() {
            cg.iter_mut().for_each(|cgi| {
                *cgi += grad[i];
                i += 1;
            });
        }
    }
}

pub(crate) struct Slice(NodeIdx, [ANode; 1], (usize, usize));

impl Slice {
    pub(crate) fn new(node: ANode, start: usize, len: usize) -> ANode {
        let idx = NodeIdx::new();
        let slice  = Slice(idx, [node], (start, len));
        ANode::new(Rc::new(slice))
    }
}

impl Node for Slice {
    fn op_name(&self) -> &str { "Slice" }

    #[inline]
    fn get_id(&self) -> NodeIdx { self.0 }

    fn get_children(&self) -> Option<&[ANode]> { 
        Some(self.1.as_slice())
    }

    fn is_leaf(&self) -> bool { false }

    fn value(&self) -> &[DType] {
        let (start, len) = self.2;
        &self.1[0].value()[start..(start+len)]
    }

    fn requires_grad(&self) -> bool { false }

    fn compute_grad(&self, grad: &[DType], child_grads: &mut [&mut [DType]]) {
        let (start, len) = self.2;
        let child = &mut child_grads[0][start..(start+len)];
        child.iter_mut().zip(grad.iter()).for_each(|(ci, gi)| {
            *ci += gi;
        });
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
    fn test_add_simple() {
        let x = Variable::new(vec![0., 1.]);
        let res = AddN::new(x.clone(), x.clone()).sum();
        assert_eq!(res.value(), &[2.]);


        let mut graph = Graph::new();
        graph.backward(&res);

        let res = graph.get_grad(&x).unwrap();
        assert_eq!(res, &[2., 2.]);
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
    fn test_mul_4() {
        let x = Variable::new(vec![2., 3.]);
        let x2 = Multiply::new(x.clone(), x.clone());
        let x4 = Multiply::new(x2.clone(), x2);
        assert_eq!(x4.value(), &[2f32.powf(4.), 3f32.powf(4.)]);

        let mut graph = Graph::new();
        graph.backward(&x4);
        let x_grad = graph.get_grad(&x).unwrap();

        assert_eq!(x_grad, &[4. * 2f32.powf(3.), 4. * 3f32.powf(3.)]);
    }

    #[test]
    fn test_sqrt() {
        let x = Variable::new(vec![4., 9.]);
        let res = SquareRoot::new(x.clone());
        assert_eq!(res.value(), &[2., 3.]);

        let mut graph = Graph::new();
        graph.backward(&res);

        let x_1_g = 1f32 / (2f32 * 2f32);
        let x_2_g = 1f32 / (2f32 * 3f32);
        let x_grad = graph.get_grad(&x).unwrap();
        assert_eq!(x_grad, &[x_1_g, x_2_g]);
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
    fn test_tanh() {
        let x = Variable::new(vec![0., 1., 2.]);
        let out = (&x).tanh();
        assert_eq!(out.value(), &[0., 1f32.tanh(), 2f32.tanh()]);
        let mut graph = Graph::new();
        graph.backward(&out);
        let grad = graph.get_grad(&x).unwrap();
        assert_eq!(grad, &[1., (1f32 - 1f32.tanh().powf(2f32)), (1f32 - 2f32.tanh().powf(2f32))]);
    }

    #[test]
    fn test_exp() {
        let x = Variable::new(vec![0., 1., 2.]);
        let out = (&x).exp();
        let mut graph = Graph::new();
        graph.backward(&out);
        let grad = graph.get_grad(&x).unwrap();
        assert_eq!(out.value(), &[1., 1f32.exp(), 2f32.exp()]);
        assert_eq!(grad, &[1., 1f32.exp(), 2f32.exp()]);
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
    fn test_maximum() {
        let x = Variable::new(vec![1., 2.]);
        let y = Variable::new(vec![3., 5.]);

        let out = (&x).pow(4f32).maximum(2f32 * &y);

        let mut graph = Graph::new();
        graph.backward(&out);

        let x_grad = graph.get_grad(&x).unwrap();
        let y_grad = graph.get_grad(&y).unwrap();
        assert_eq!(x_grad, &[0f32, 32f32]);
        assert_eq!(y_grad, &[2f32, 0f32]);
    }

    #[test]
    fn test_minimum() {
        let x = Variable::new(vec![1., 2.]);
        let y = Variable::new(vec![3., 5.]);

        let out = (&x).pow(4f32).minimum(2f32 * &y);

        let mut graph = Graph::new();
        graph.backward(&out);

        let x_grad = graph.get_grad(&x).unwrap();
        let y_grad = graph.get_grad(&y).unwrap();
        assert_eq!(x_grad, &[4f32, 0f32]);
        assert_eq!(y_grad, &[0f32, 2f32]);
    }

    #[test]
    fn test_concat() {
        let x = Variable::new(vec![1., 2.]);
        let y = Variable::new(vec![3., 5.]);

        let mut out = vec![&x, &y].concat();
        out = out + 10f32;

        let mut graph = Graph::new();
        graph.backward(&out);

        let x_grad = graph.get_grad(&x).unwrap();
        let y_grad = graph.get_grad(&y).unwrap();
        assert_eq!(x_grad, &[1., 1.]);
        assert_eq!(y_grad, &[1., 1.]);
    }

    #[test]
    fn test_slice() {
        let x = Variable::new(vec![1., 2., 3.]);

        let x_slice = x.slice(1, 2);
        let out = x_slice * 2.;

        let mut graph = Graph::new();
        graph.backward(&out);

        let x_grad = graph.get_grad(&x).unwrap();
        assert_eq!(x_grad, &[0., 2., 2.]);
    }

    #[test]
    fn test_bulk_sum() {
        let mut v = Variable::new([1f32, 2f32, 3f32].to_vec());
        let vecs = vec![v.clone(); 10];
        let bs = BulkSum::new(vecs.into_iter());
        assert_eq!(bs.value(), &[10f32, 20f32, 30f32]);
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

    fn euclidean_distance(x: &ANode, y: &ANode) -> ANode {
        let minus = x - y;
        let pow = minus.pow(2f32);
        let sum = pow.sum();
        let sqrt = sum.pow(0.5);
        sqrt
    }

    #[test]
    fn test_backward_pass_complicated() {
        // (x+2) ^ 2 
        // x^2 + 4x + 4
        // 2x + 4
        let x      = Variable::new(vec![0f32]);
        let x2     = AddN::new(x.clone(), Constant::scalar(2f32));
        let x2_2   = Power::new(x2, Constant::scalar(2f32));

        assert_eq!(x2_2.value(), vec![4f32]);

        let mut graph = Graph::new();
        graph.backward(&x2_2);

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
            graph.zero_grads();
            graph.backward(&err);
            let x_grad = graph.get_grad(&x).unwrap();
            
            // SGD!
            v.iter_mut().zip(x_grad.iter()).for_each(|(vi, gi)| {
                *vi -= alpha * *gi;
            });
        }

        assert!((v[0] - y.value()[0]).abs() < 1e-5);
        assert!((v[1] - y.value()[1]).abs() < 1e-5);
    }

}
