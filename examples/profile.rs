//! Standalone harness mirroring tests/benchmark.py workloads for fast
//! native iteration and callgrind profiling. Not shipped; dev-only.

use std::hint::black_box;
use std::time::Instant;

use kdtree::tree::Tree;

// Simple deterministic RNG (xorshift64*) + Box-Muller for normals.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn make_data(distribution: &str, n_points: usize, dims: usize) -> Vec<f64> {
    let mut rng = Rng::new(1234);
    let mut data = vec![0.0; n_points * dims];
    match distribution {
        "bimodal" => {
            for (i, v) in data.iter_mut().enumerate() {
                let row = i / dims;
                *v = rng.normal() + if row >= n_points / 2 { 1.0 } else { 0.0 };
            }
        }
        "uniform" => {
            for v in data.iter_mut() {
                *v = rng.uniform();
            }
        }
        "sorted" => {
            for row in 0..n_points {
                let val = (n_points - row) as f64 / n_points as f64;
                for d in 0..dims {
                    data[row * dims + d] = val;
                }
            }
        }
        _ => unreachable!(),
    }
    data
}

fn time_it<F: FnMut()>(mut f: F, warmup: usize, iters: usize) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        f();
        let dt = t.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
    }
    best
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("all");
    let n_points = 10_000;
    let n_queries = 1_000;

    if mode == "all" || mode == "build" {
        for dims in [3, 8, 16] {
            for dist in ["bimodal", "uniform", "sorted"] {
                let data = make_data(dist, n_points, dims);
                let best = time_it(
                    || {
                        black_box(Tree::new(black_box(data.clone()), dims, 16).unwrap());
                    },
                    8,
                    30,
                );
                println!("build d{dims} {dist}: {:.1}us", best * 1e6);
            }
        }
    }

    // Mirror tests/benchmark.py: data and queries come from the same seeded
    // stream, so uniform queries coincide exactly with the first data rows.
    if mode == "all" || mode == "query" {
        for dims in [3, 8, 16] {
            let data = make_data("uniform", n_points, dims);
            let queries: Vec<f64> = data[..n_queries * dims].to_vec();
            for leafsize in [8, 128] {
                let tree = Tree::new(data.clone(), dims, leafsize).unwrap();
                for p in [1.0, 2.0, f64::INFINITY] {
                    let best = time_it(
                        || {
                            black_box(
                                tree.query(black_box(&queries), 1, p, None, 0.0, false).unwrap(),
                            );
                        },
                        8,
                        30,
                    );
                    println!(
                        "query d{dims} leaf{leafsize} p{p}: {:.1}us",
                        best * 1e6
                    );
                }
            }
            // Degenerate diagonal data ("sorted"), uniform queries, leafsize 16.
            let sorted_data = make_data("sorted", n_points, dims);
            let mut rng = Rng::new(99);
            let uq: Vec<f64> = (0..n_queries * dims).map(|_| rng.uniform()).collect();
            let tree = Tree::new(sorted_data, dims, 16).unwrap();
            let best = time_it(
                || {
                    black_box(tree.query(black_box(&uq), 1, 2.0, None, 0.0, false).unwrap());
                },
                3,
                10,
            );
            println!("query d{dims} sorted leaf16 p2: {:.1}us", best * 1e6);
        }
    }

    // single hot loop for callgrind: `profile callgrind-build` / `callgrind-query`
    if mode == "callgrind-build" {
        for dims in [3, 8, 16] {
            let data = make_data("bimodal", n_points, dims);
            for _ in 0..20 {
                black_box(Tree::new(black_box(data.clone()), dims, 16).unwrap());
            }
        }
    }
    if mode == "callgrind-query" {
        for dims in [3, 8, 16] {
            let data = make_data("uniform", n_points, dims);
            let queries: Vec<f64> = data[..n_queries * dims].to_vec();
            for leafsize in [8, 128] {
                let tree = Tree::new(data.clone(), dims, leafsize).unwrap();
                for p in [1.0, 2.0, f64::INFINITY] {
                    for _ in 0..10 {
                        black_box(tree.query(black_box(&queries), 1, p, None, 0.0, false).unwrap());
                    }
                }
            }
        }
    }
    if mode == "callgrind-query-sorted" {
        for dims in [3, 8, 16] {
            let sorted_data = make_data("sorted", n_points, dims);
            let mut rng = Rng::new(99);
            let uq: Vec<f64> = (0..n_queries * dims).map(|_| rng.uniform()).collect();
            let tree = Tree::new(sorted_data, dims, 16).unwrap();
            for _ in 0..3 {
                black_box(tree.query(black_box(&uq), 1, 2.0, None, 0.0, false).unwrap());
            }
        }
    }
}
