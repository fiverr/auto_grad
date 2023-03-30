use criterion::{black_box, criterion_group, criterion_main, Criterion};
use simple_grad::*;

fn vec_ops(c: &mut Criterion) {
    let dims = 100;
    let mut embeddings = Vec::new();

    for i in 0..100000 {
        let mut row = Vec::with_capacity(dims);
        for dim in 0..dims {
            row.push((i*dim) as f32);
        }
        embeddings.push(Variable::new(row));
    }
    //c.bench_function("bench vecs", |b| b.iter(|| embeddings.clone().sum_all()));

    let results = embeddings.clone().sum_all();
    c.bench_function("bench backward", |b| b.iter(|| {
        let mut graph = Graph::new();
        graph.backward(&results);
    }));
}

criterion_group!(benches, vec_ops);
criterion_main!(benches);
