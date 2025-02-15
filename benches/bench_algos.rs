use criterion::{black_box, criterion_group, criterion_main, Criterion};
use float_ord::FloatOrd;
use simple_grad::*;

fn vec_ops(c: &mut Criterion) {
    let dims = 256;
    let num_vecs = 10;
    let mut embeddings = Vec::new();
    use_shared_pool(false);

    for i in 0..num_vecs {
        let mut row = Vec::with_capacity(dims);
        for dim in 0..dims {
            row.push((i*dim) as f32);
        }
        embeddings.push(Variable::new(row));
    }

    c.bench_function("bench mul", |b| b.iter(|| {
        let mut res = embeddings[0].clone();
        for i in 1..num_vecs {
            res = res * &embeddings[i];
        }
        let res = res.sum();
        let mut graph = Graph::new();
        graph.backward(&res);
    }));

    c.bench_function("bench pow", |b| b.iter(|| {
        let mut res = embeddings[0].clone();
        for i in 1..num_vecs {
            res = res.pow(&embeddings[i]);
        }
        let res = res.sum();
        let mut graph = Graph::new();
        graph.backward(&res);
    }));

    c.bench_function("bench sub", |b| b.iter(|| {
        let mut res = embeddings[0].clone();
        for i in 1..num_vecs {
            res = res - &embeddings[i];
        }
        let res = res.sum();
        let mut graph = Graph::new();
        graph.backward(&res);
    }));

    c.bench_function("bench add", |b| b.iter(|| {
        let mut res = embeddings[0].clone();
        for i in 1..num_vecs {
            res = res + &embeddings[i];
        }
        let res = res.sum();
        let mut graph = Graph::new();
        graph.backward(&res);
    }));

    c.bench_function("bench sqrt", |b| b.iter(|| {
        let mut res = embeddings[0].clone();
        for i in 1..num_vecs {
            res = res.pow(0.5);
        }
        let res = res.sum();
        let mut graph = Graph::new();
        graph.backward(&res);
    }));

    c.bench_function("bench pow2", |b| b.iter(|| {
        let mut res = embeddings[0].clone();
        for i in 1..num_vecs {
            res = res.pow(2f32);
        }
        let res = res.sum();
        let mut graph = Graph::new();
        graph.backward(&res);
    }));

    c.bench_function("bench exp", |b| b.iter(|| {
        let mut res = embeddings[0].clone();
        for i in 1..num_vecs {
            res = res.exp();
        }
        let res = res.sum();
        let mut graph = Graph::new();
        graph.backward(&res);
    }));

}

criterion_group!(benches, vec_ops);
criterion_main!(benches);
