use std::arch::x86_64::*;

#[inline]
pub fn add(l: &[f32], r: &[f32], out: &mut [f32]) {
    l.iter().zip(r.iter()).zip(out.iter_mut()).for_each(|((li, ri), outi)| {
        *outi = li + ri;
    });
}

#[inline]
pub fn iadd(l: &mut [f32], r: &[f32]) {
    l.iter_mut().zip(r.iter()).for_each(|(li, ri)| {
        *li += ri;
    });
}

#[inline]
pub fn sub(l: &[f32], r: &[f32], out: &mut [f32]) {
    l.iter().zip(r.iter()).zip(out.iter_mut()).for_each(|((li, ri), outi)| {
        *outi = li - ri;
    });
}

#[inline]
pub fn isub(l: &mut [f32], r: &[f32]) {
    l.iter_mut().zip(r.iter()).for_each(|(li, ri)| {
        *li -= ri;
    });
}

#[inline]
pub fn mul(l: &[f32], r: &[f32], out: &mut [f32]) {
    l.iter().zip(r.iter()).zip(out.iter_mut()).for_each(|((li, ri), outi)| {
        *outi = li * ri;
    });
}

#[inline]
pub fn imul(l: &mut [f32], r: &[f32]) {
    l.iter_mut().zip(r.iter()).for_each(|(li, ri)| {
        *li *= ri;
    });
}

#[inline]
pub fn div(l: &[f32], r: &[f32], out: &mut [f32]) {
    l.iter().zip(r.iter()).zip(out.iter_mut()).for_each(|((li, ri), outi)| {
        *outi = li / ri;
    });
}

pub trait Input {
    unsafe fn fill_256(&self, idx: usize) -> __m256; 
    fn iter(self, idx: usize) -> impl Iterator<Item=f32>;
    fn len(&self) -> usize;
}

pub trait Output {
    unsafe fn fill_256(&self, idx: usize) -> __m256; 
    unsafe fn store(&mut self, register: __m256, idx: usize);
    fn store_one(&mut self, value: f32, idx: usize);
    fn len(&self) -> usize;
}

pub struct BroadcastInput<'a>(pub &'a [f32], pub usize);
impl <'a> Input for BroadcastInput<'a> {

    #[inline(always)]
    unsafe fn fill_256(&self, _idx: usize) -> __m256 {
        _mm256_set1_ps(self.0[0])
    }

    #[inline(always)]
    fn iter(self, _idx: usize) -> impl Iterator<Item=f32> {
        std::iter::repeat(self.0[0])
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.1
    }

}

pub struct ArrayInput<'a>(pub &'a [f32]);
impl <'a> Input for ArrayInput<'a> {

    #[inline(always)]
    unsafe fn fill_256(&self, idx: usize) -> __m256 {
        _mm256_loadu_ps(self.0.as_ptr().add(idx))
    }

    #[inline(always)]
    fn iter(self, idx: usize) -> impl Iterator<Item=f32> {
        self.0[idx..].iter().cloned()
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len()
    }

}

pub struct BroadcastOutput<'a>(pub &'a mut [f32], pub usize);
impl <'a> Output for BroadcastOutput<'a> {

    #[inline(always)]
    unsafe fn fill_256(&self, idx: usize) -> __m256 {
        _mm256_set1_ps(self.0[0])
    }

    #[inline(always)]
    unsafe fn store(&mut self, register: __m256, idx: usize) {
        self.0[0] += hsum_avx_ps(register)
    }

    #[inline(always)]
    fn store_one(&mut self, value: f32, idx: usize) {
        self.0[0] += value
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.1
    }
}

pub struct ArrayOutput<'a>(pub &'a mut [f32]);
impl <'a> Output for ArrayOutput<'a> {

    #[inline(always)]
    unsafe fn fill_256(&self, idx: usize) -> __m256 {
        _mm256_loadu_ps(self.0.as_ptr().add(idx))
    }

    #[inline(always)]
    unsafe fn store(&mut self, register: __m256, idx: usize) {
        let vo = self.fill_256(idx);
        let res = _mm256_add_ps(register, vo);
        _mm256_storeu_ps(self.0.as_mut_ptr().add(idx), res);
    }

    #[inline(always)]
    fn store_one(&mut self, value: f32, idx: usize) {
        self.0[idx] += value
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len()
    }
}

/// Horizontal sum of all 8 floats in an __m256, returning a single f32.
/// Uses pairwise `_mm_hadd_ps` in SSE after extracting the high 128 bits.
#[target_feature(enable = "avx")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn hsum_avx_ps(v: __m256) -> f32 {
    // Extract the high 128 bits
    let high = _mm256_extractf128_ps(v, 1);
    // Add the low 128 bits (cast) to the high 128 bits
    let sum_128 = _mm_add_ps(_mm256_castps256_ps128(v), high);
    // Now we have a 128-bit register with 4 floats. Use SSE horizontal add twice.
    let sum_128 = _mm_hadd_ps(sum_128, sum_128);
    let sum_128 = _mm_hadd_ps(sum_128, sum_128);
    // Move the lowest float out
    _mm_cvtss_f32(sum_128)
}

macro_rules! avx_detect {
    ($block:expr) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // Note that this `unsafe` block is safe because we're testing
            // that the `avx2` feature is indeed available on our CPU.
            if is_x86_feature_detected!("avx") {
                $block
            }
        }
    }
}

macro_rules! unary_op {
    ($fname:ident, $sim_op:expr, $fallback_op:expr) => {
        pub unsafe fn $fname(
            a: impl Input,
            mut out: impl Output
        ) {
            // Ensure slices have the same length
            assert_eq!(a.len(), out.len(), "Input slices must have the same length.");

            let length = a.len();
            let mut i = 0;

            avx_detect! {
                // Process in chunks of 8 floats
                while i + 8 <= length {
                    let va = a.fill_256(i);
                    let res = $sim_op(va);
                    out.store(res, i);
                    i += 8;
                }
            }

            a.iter(i).for_each(|ai| {
                out.store_one($fallback_op(ai), i);
                i += 1;
            });

        }

    }
}

macro_rules! binary_op {
    ($fname:ident, $sim_op:expr, $(&mut)? $fallback_op:expr) => {
        pub unsafe fn $fname(
            a: impl Input,
            b: impl Input,
            mut out: impl Output
        ) {
            // Ensure slices have the same length
            assert_eq!(a.len(), b.len(), "Input slices must have the same length.");
            assert_eq!(a.len(), out.len(), "Input slices must have the same length.");

            let length = a.len();
            let mut i = 0;

            avx_detect! {
                // Process in chunks of 8 floats
                while i + 8 <= length {
                    let va = a.fill_256(i);
                    let vb = b.fill_256(i);
                    let res = $sim_op(va, vb);
                    out.store(res, i);
                    i += 8;
                }
            }

            a.iter(i).zip(b.iter(i)).for_each(|(ai, bi)| {
                out.store_one($fallback_op(ai, bi), i);
                i += 1;
            });

        }

    }
}

macro_rules! trinary_op {
    ($fname:ident, $sim_op:expr, $fallback_op:expr) => {
        pub unsafe fn $fname(
            a: impl Input,
            b: impl Input,
            c: impl Input,
            mut out: impl Output
        ) {
            // Ensure slices have the same length
            assert_eq!(a.len(), b.len(), "Input slices must have the same length.");
            assert_eq!(a.len(), c.len(), "Input slices must have the same length.");
            assert_eq!(a.len(), out.len(), "Input slices must have the same length.");

            let length = a.len();
            let mut i = 0;

            avx_detect! {
                // Process in chunks of 8 floats
                while i + 8 <= length {
                    let va = a.fill_256(i);
                    let vb = b.fill_256(i);
                    let vc = c.fill_256(i);
                    let res = $sim_op(va, vb, vc);
                    out.store(res, i);
                    i += 8;
                }
            }

            a.iter(i).zip(b.iter(i).zip(c.iter(i))).for_each(|(ai, (bi, ci))| {
                out.store_one($fallback_op(ai, bi, ci), i);
                i += 1;
            });

        }

    }
}


binary_op!(
    simd_div,
    _mm256_div_ps,
    |xi, yi| { xi / yi }
);

binary_op!(
    grad_div_x,
    _mm256_div_ps,
    |gi, yi| { gi / yi }
);

trinary_op!(
    grad_div_y,
    |go, xo, yo| {
        // df(x,y)/dy = -x / y ^ 2
        let neg_x = _mm256_xor_ps(xo, _mm256_set1_ps(-0.0));
        let y_2 = _mm256_mul_ps(yo, yo);
        let g_neg_x = _mm256_mul_ps(neg_x, go);
        _mm256_div_ps(g_neg_x, y_2)
    },
    |gi: f32, xi: f32, yi: f32| { gi * (-xi / (yi * yi)) }
);

binary_op!(
    simd_mul,
    _mm256_mul_ps,
    |xi, yi| { xi * yi }
);

