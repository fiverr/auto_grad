use criterion::{black_box, criterion_group, criterion_main, Criterion};
use float_ord::FloatOrd;
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

    use_shared_pool(false);
    let results = embeddings.clone().sum_all();
    c.bench_function("bench backward", |b| b.iter(|| {
        let mut graph = Graph::new();
        graph.backward(&results);
    }));
}

fn bench_attention(c: &mut Criterion) {
    let dims = 100;
    let mut embeddings = Vec::new();

    for i in 0..20 {
        let mut row = Vec::with_capacity(dims);
        for dim in 0..dims {
            row.push((i*dim) as f32);
        }
        embeddings.push((Variable::new(row), 1));
    }
    use_shared_pool(false);
    c.bench_function("bench backward", |b| b.iter(|| {
        let e = attention_mean(embeddings.iter(), 20, None);
        let mut graph = Graph::new();
        graph.backward(&e.sum());
    }));
}

pub fn attention_mean<'a>(
    it: impl Iterator<Item=&'a (ANode, usize)>,
    attention_dims: usize,
    window: Option<usize>
) -> ANode {

    let items: Vec<_> = it.map(|(node, count)| {
        (Attention::new(node, attention_dims), *count)
    }).collect();

    if items.len() == 1 {
        return items[0].0.value.clone()
    }

    // Compute attention matrix
    let attention_matrix = compute_attention_matrix(&items, window);

    let att = compute_attention_softmax(attention_matrix, attention_dims);

    let summed_weights = att.sum_all();
    let n = items.len() as f32;
    items.into_iter().enumerate()
        .map(|(i, (at_i, _c))| at_i.value * summed_weights.slice(i, 1))
        .collect::<Vec<_>>().sum_all() / n
}

fn compute_attention_matrix(
    items: &[(Attention, usize)],
    window: Option<usize>
) -> Vec<Vec<ANode>> {

     // Get the attention for each feature
    let zero = Constant::scalar(0.);
    let mut scaled = vec![vec![zero; items.len()]; items.len()];
    for i in 0..items.len() {
        let (j_start, j_end) = match window {
            Some(size) => {
                let start = if size > i { 0 } else {i - size };
                let stop = (i + size + 1).min(items.len());
                (start, stop)
            },
            None => (0, items.len())
        };

        let (at_i, ic) = &items[i];
        let row = &mut scaled[i];
        for j in j_start..j_end {
            let (at_j, jc) = &items[j];
            let mut dot_i_j = (&at_i.query).dot(&at_j.key);
            let num = ic * jc;
            if num >= 1 && window.is_none() {
                dot_i_j = dot_i_j * (num as f32);
            }
            row[j] = dot_i_j;
        }
    }
    scaled
}


fn compute_attention_softmax(
    attention_matrix: Vec<Vec<ANode>>,
    d_k: usize
) -> Vec<ANode> {
    // Compute softmax
    let d_k = Constant::scalar((d_k as f32).sqrt());

    // Compute softmax for each feature
    let mut att = Vec::with_capacity(attention_matrix.len());
    for row in attention_matrix.into_iter() {
        let row = row.concat() / &d_k;
        let sm = softmax(row);
        att.push(sm);
    }

    att
}

fn softmax(numers: ANode) -> ANode {

    let max_value = numers.value().iter()
        .max_by_key(|v| FloatOrd(**v))
        .expect("Shouldn't be non-zero!");
    let mv = Constant::scalar(*max_value);
    let n = (numers - &mv).exp();
    &n / n.sum()
}

#[derive(Clone)]
struct Attention {
    query: ANode,
    key: ANode,
    value: ANode
}

impl Attention {
    fn new(node: &ANode, attention_dims: usize) -> Self {
        let query = get_query_vec(&node, attention_dims);
        let key = get_key_vec(&node, attention_dims);
        let value = get_value_vec(&node, attention_dims);
        Attention {query, key, value}
    }
}

fn get_value_vec(emb: &ANode, dims: usize) -> ANode {
    let v = emb.value().len();
    emb.slice(2*dims, v - 2*dims)
}

fn get_query_vec(emb: &ANode, dims: usize) -> ANode {
    emb.slice(0, dims)
}

fn get_key_vec(emb: &ANode, dims: usize) -> ANode {
    emb.slice(dims, dims)
}


//criterion_group!(benches, vec_ops);
criterion_group!(benches, bench_attention);
criterion_main!(benches);
