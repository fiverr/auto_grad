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

