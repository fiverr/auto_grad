
pub fn add(l: &[f32], r: &[f32], out: &mut [f32]) {
    l.iter().zip(r.iter()).zip(out.iter_mut()).for_each(|((li, ri), outi)| {
        *outi = li + ri;
    });
}

pub fn iadd(l: &mut [f32], r: &[f32]) {
    l.iter_mut().zip(r.iter()).for_each(|(li, ri)| {
        *li += ri;
    });
}

pub fn sub(l: &[f32], r: &[f32], out: &mut [f32]) {
    l.iter().zip(r.iter()).zip(out.iter_mut()).for_each(|((li, ri), outi)| {
        *outi = li - ri;
    });
}

pub fn isub(l: &mut [f32], r: &[f32]) {
    l.iter_mut().zip(r.iter()).for_each(|(li, ri)| {
        *li -= ri;
    });
}

pub fn mul(l: &[f32], r: &[f32], out: &mut [f32]) {
    l.iter().zip(r.iter()).zip(out.iter_mut()).for_each(|((li, ri), outi)| {
        *outi = li * ri;
    });
}

pub fn imul(l: &mut [f32], r: &[f32]) {
    l.iter_mut().zip(r.iter()).for_each(|(li, ri)| {
        *li *= ri;
    });
}

pub fn div(l: &[f32], r: &[f32], out: &mut [f32]) {
    l.iter().zip(r.iter()).zip(out.iter_mut()).for_each(|((li, ri), outi)| {
        *outi = li / ri;
    });
}

pub fn idiv(l: &mut [f32], r: &[f32]) {
    l.iter_mut().zip(r.iter()).for_each(|(li, ri)| {
        *li /= ri;
    });
}

pub fn pow(l: &[f32], r: &[f32], out: &mut [f32]) {
    l.iter().zip(r.iter()).zip(out.iter_mut()).for_each(|((li, ri), outi)| {
        *outi = li.powf(*ri);
    });
}

pub fn ipow(l: &mut [f32], r: &[f32]) {
    l.iter_mut().zip(r.iter()).for_each(|(li, ri)| {
        *li = li.powf(*ri);
    });
}

