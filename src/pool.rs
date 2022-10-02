use lazy_static::lazy_static;

use std::convert::{AsRef, AsMut};
use std::collections::HashMap;
use std::sync::Mutex;
use std::ops::{Drop,Deref,DerefMut};

use crate::DType;

lazy_static! {
    static ref POOL: Mutex<MemoryPool> = {
        let m = MemoryPool::new();
        Mutex::new(m)
    };
}

struct MemoryPool {
    data: HashMap<usize, Vec<Vec<DType>>>
}

impl MemoryPool {
    fn new() -> Self {
        MemoryPool { data: HashMap::new() }
    }

    fn get(&mut self, size: usize) -> MPVec {
        if let Some(vs) = self.data.get_mut(&size) {
            if let Some(mut v) = vs.pop() {
                v.fill(0f32);
                return MPVec(v)
            }
        }
        MPVec(vec![0f32; size])
    }

    fn ret(&mut self, v: Vec<DType>) {
        let e = self.data.entry(v.len()).or_insert_with(|| Vec::new());
        e.push(v);
    }

    fn clear(&mut self) {
        self.data.clear();
    }
}

pub fn allocate_vec(size: usize) -> MPVec {
    let mut pool = POOL.lock()
        .expect("Error accessing memory pool!");
    pool.get(size)
}

pub fn clear_pool() {
    let mut pool = POOL.lock()
        .expect("Error accessing memory pool!");
    pool.clear();
}

fn return_vec(v: Vec<DType>) {
    let mut pool = POOL.lock()
        .expect("Error accessing memory pool!");
    pool.ret(v);
}

pub struct MPVec(Vec<DType>);

impl Drop for MPVec {
    fn drop(&mut self) {
        let mut m = Vec::with_capacity(0);
        std::mem::swap(&mut m, &mut self.0);
        return_vec(m);
    }
}

impl AsRef<Vec<DType>> for MPVec {
    fn as_ref(&self) -> &Vec<DType> {
        &self.0
    }
}

impl AsMut<Vec<DType>> for MPVec {
    fn as_mut(&mut self) -> &mut Vec<DType> {
        &mut self.0
    }
}

impl Deref for MPVec {
    type Target = Vec<DType>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for MPVec {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
