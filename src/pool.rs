use hashbrown::HashMap;
use std::convert::{AsRef, AsMut};
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool,Ordering};
use std::ops::{Drop,Deref,DerefMut};

use crate::DType;

static USE_POOL: AtomicBool = AtomicBool::new(true);
thread_local! {
    static POOL: RefCell<MemoryPool> = {
        let m = MemoryPool::new();
        RefCell::new(m)
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

#[inline]
pub fn use_shared_pool(use_pool: bool) {
    USE_POOL.store(use_pool, Ordering::SeqCst);
}

#[inline]
fn should_use_pool() -> bool {
    USE_POOL.load(Ordering::Relaxed)
}

pub fn allocate_vec(size: usize) -> MPVec {
    if should_use_pool() {
        POOL.with(|p| {
            let mut pool = p.borrow_mut();
            pool.get(size)
        })
    } else {
        MPVec(vec![0.; size])
    }
}

pub fn clear_pool() {
    POOL.with(|p| {
        let mut pool = p.borrow_mut();
        pool.clear();
    });
}

fn return_vec(v: Vec<DType>) {
    if should_use_pool() {
        POOL.with(|p| {
            let mut pool = p.borrow_mut();
            pool.ret(v);
        });
    }
}

#[derive(Debug,Clone)]
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
