//! The workload grid of `tests/benchmark.py`, without Python. That file stays the
//! SciPy comparison and the parity bar; ids here are formatted exactly as
//! `benchmark.py` formats them, so the two reports line up case by case.
//!
//! Same distributions as `benchmark.py`, but drawn from a self-contained
//! SplitMix64 stream — not the same bits as numpy's PCG64. The grid is a
//! superset in one direction: parallel build has no counterpart there, because
//! SciPy cannot build in parallel and there would be nothing to compare against.
//!
//! ```text
//! cargo bench --bench grid -- --list
//! cargo bench --bench grid -- query-d8          # ids containing the argument
//! cargo bench --bench grid -- --iters 200 build-d8-n10000-serial
//! ./scripts/callgrind.sh build-d8-n10000-serial # one case, instruction counts
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use kdtree::tree::Tree;

const DATA_SEED: u64 = 123_456_789;
const QUERY_SEED: u64 = 987_654_321;

/// `(ndim, n_points, n_queries)`.
const SIZES: [(usize, usize, usize); 3] =
    [(3, 10_000, 1_000), (8, 10_000, 1_000), (16, 10_000, 1_000)];

const MIN_ITERS: usize = 5;
const MAX_ITERS: usize = 1_000;
const MIN_TOTAL: Duration = Duration::from_millis(300);

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
    }

    /// Box-Muller, dropping the second variate: caching it would make a sample
    /// depend on how many were drawn before it.
    fn normal(&mut self) -> f64 {
        let u = 1.0 - self.uniform();
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

#[derive(Clone, Copy)]
enum Distribution {
    Bimodal,
    Uniform,
    /// Every point on the diagonal: split planes stop separating anything.
    Sorted,
}

impl Distribution {
    fn name(self) -> &'static str {
        match self {
            Self::Bimodal => "bimodal",
            Self::Uniform => "uniform",
            Self::Sorted => "sorted",
        }
    }
}

fn make_data(distribution: Distribution, n_points: usize, ndim: usize) -> Vec<f64> {
    match distribution {
        Distribution::Bimodal => {
            let mut rng = Rng::new(DATA_SEED);
            let half = n_points / 2;
            let mut data = Vec::with_capacity(n_points * ndim);
            data.extend((0..half * ndim).map(|_| rng.normal()));
            data.extend((0..(n_points - half) * ndim).map(|_| rng.normal() + 1.0));
            data
        }
        Distribution::Uniform => {
            let mut rng = Rng::new(DATA_SEED);
            (0..n_points * ndim).map(|_| rng.uniform()).collect()
        }
        Distribution::Sorted => (0..n_points)
            .flat_map(|i| {
                let v = (n_points - i) as f64 / n_points as f64;
                std::iter::repeat_n(v, ndim)
            })
            .collect(),
    }
}

fn make_queries(n_queries: usize, ndim: usize) -> Vec<f64> {
    let mut rng = Rng::new(QUERY_SEED);
    (0..n_queries * ndim).map(|_| rng.uniform()).collect()
}

enum Spec {
    Build {
        ndim: usize,
        n_points: usize,
        distribution: Distribution,
        leafsize: usize,
        parallel: bool,
        /// Which of `benchmark.py`'s two families this belongs to — it selects
        /// the id prefix and nothing else. Explicit rather than derived from
        /// `distribution`, because the balanced query sweep also draws
        /// `Uniform`, and varies `leafsize` and `p` instead.
        unbalanced: bool,
    },
    Query {
        ndim: usize,
        n_points: usize,
        n_queries: usize,
        distribution: Distribution,
        leafsize: usize,
        p: f64,
        parallel: bool,
        unbalanced: bool,
    },
}

fn parallel_tag(parallel: bool) -> &'static str {
    if parallel { "parallel" } else { "serial" }
}

/// `benchmark.py` formats `p` with `:g`, which agrees with `{p:.0}` only for the
/// integral values the grid uses. `{p:.0}` rounds, so adding `p = 1.5` here
/// would not merely diverge from Python — it would emit `p2` a second time; the
/// id uniqueness assert in `main` is what turns that into a hard error.
fn p_tag(p: f64) -> String {
    if p.is_infinite() {
        "inf".to_string()
    } else {
        format!("{p:.0}")
    }
}

impl Spec {
    /// The four arms are `benchmark.py`'s four `pytest.param` lists. They live
    /// next to each other because matching those format strings character for
    /// character is the whole reason the two reports line up.
    fn id(&self) -> String {
        match *self {
            Self::Build {
                ndim,
                n_points,
                parallel,
                unbalanced: false,
                ..
            } => format!(
                "build-d{ndim}-n{n_points}-{tag}",
                tag = parallel_tag(parallel)
            ),
            Self::Build {
                ndim,
                n_points,
                distribution,
                parallel,
                unbalanced: true,
                ..
            } => format!(
                "build-unbalanced-d{ndim}-n{n_points}-{dist}-{tag}",
                dist = distribution.name(),
                tag = parallel_tag(parallel)
            ),
            Self::Query {
                ndim,
                n_points,
                n_queries,
                distribution,
                parallel,
                unbalanced: true,
                ..
            } => format!(
                "query-unbalanced-d{ndim}-n{n_points}-q{n_queries}-{dist}-{tag}",
                dist = distribution.name(),
                tag = parallel_tag(parallel)
            ),
            Self::Query {
                ndim,
                n_points,
                n_queries,
                leafsize,
                p,
                parallel,
                unbalanced: false,
                ..
            } => format!(
                "query-d{ndim}-n{n_points}-q{n_queries}-leaf{leafsize}-p{p}-{tag}",
                p = p_tag(p),
                tag = parallel_tag(parallel)
            ),
        }
    }
}

/// Owns everything a measured iteration touches, so [`kdtree_bench_target`]
/// only ever does the work under test.
enum Job {
    Build {
        data: Vec<f64>,
        ndim: usize,
        leafsize: usize,
        parallel: bool,
    },
    Query {
        tree: Tree,
        queries: Vec<f64>,
        p: f64,
        parallel: bool,
    },
}

/// The measured region, named and out of line so callgrind can gate collection
/// on it (`--toggle-collect`) and leave data generation out of the profile.
///
/// `Tree::new` consumes its data, so a build iteration clones first — as the
/// Python path does too, copying out of numpy on every call.
#[inline(never)]
#[unsafe(no_mangle)]
fn kdtree_bench_target(job: &Job) {
    match job {
        Job::Build {
            data,
            ndim,
            leafsize,
            parallel,
        } => {
            let tree = Tree::new(data.clone(), *ndim, *leafsize, *parallel).expect("build");
            black_box(&tree);
        }
        Job::Query {
            tree,
            queries,
            p,
            parallel,
        } => {
            let out = tree
                .query(queries, 1, *p, None, 0.0, *parallel)
                .expect("query");
            black_box(&out);
        }
    }
}

fn cases() -> Vec<Spec> {
    let mut cases = Vec::new();

    for (ndim, n_points, _) in SIZES {
        for parallel in [false, true] {
            cases.push(Spec::Build {
                ndim,
                n_points,
                distribution: Distribution::Bimodal,
                leafsize: 16,
                parallel,
                unbalanced: false,
            });
        }
    }
    for (ndim, n_points, _) in SIZES {
        for distribution in [Distribution::Uniform, Distribution::Sorted] {
            for parallel in [false, true] {
                cases.push(Spec::Build {
                    ndim,
                    n_points,
                    distribution,
                    leafsize: 16,
                    parallel,
                    unbalanced: true,
                });
            }
        }
    }

    for (ndim, n_points, n_queries) in SIZES {
        for distribution in [Distribution::Uniform, Distribution::Sorted] {
            for parallel in [false, true] {
                cases.push(Spec::Query {
                    ndim,
                    n_points,
                    n_queries,
                    distribution,
                    leafsize: 16,
                    p: 2.0,
                    parallel,
                    unbalanced: true,
                });
            }
        }
    }
    for (ndim, n_points, n_queries) in SIZES {
        for p in [1.0, 2.0, f64::INFINITY] {
            for leafsize in [8, 128] {
                for parallel in [false, true] {
                    cases.push(Spec::Query {
                        ndim,
                        n_points,
                        n_queries,
                        distribution: Distribution::Uniform,
                        leafsize,
                        p,
                        parallel,
                        unbalanced: false,
                    });
                }
            }
        }
    }

    cases
}

fn prepare(spec: &Spec) -> Job {
    match *spec {
        Spec::Build {
            ndim,
            n_points,
            distribution,
            leafsize,
            parallel,
            ..
        } => Job::Build {
            data: make_data(distribution, n_points, ndim),
            ndim,
            leafsize,
            parallel,
        },
        Spec::Query {
            ndim,
            n_points,
            n_queries,
            distribution,
            leafsize,
            p,
            parallel,
            ..
        } => Job::Query {
            tree: Tree::new(
                make_data(distribution, n_points, ndim),
                ndim,
                leafsize,
                false,
            )
            .expect("build"),
            queries: make_queries(n_queries, ndim),
            p,
            parallel,
        },
    }
}

/// A fixed count is for the external counters in `scripts/`: it skips the
/// warm-up, so exactly `iters` calls land inside the measured region and there
/// is nothing else in there to divide out.
fn measure(job: &Job, fixed: Option<usize>) -> Vec<Duration> {
    if fixed.is_none() {
        kdtree_bench_target(job);
    }
    let mut samples = Vec::new();
    let started = Instant::now();
    loop {
        let iteration = Instant::now();
        kdtree_bench_target(job);
        samples.push(iteration.elapsed());
        let done = match fixed {
            Some(n) => samples.len() >= n,
            None => {
                samples.len() >= MAX_ITERS
                    || (samples.len() >= MIN_ITERS && started.elapsed() >= MIN_TOTAL)
            }
        };
        if done {
            return samples;
        }
    }
}

fn main() {
    let mut iters = None;
    let mut list = false;
    let mut filters = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--list" => list = true,
            "--iters" => {
                let value = args.next().expect("--iters takes a count");
                iters = Some(value.parse::<usize>().expect("--iters takes a count"));
            }
            // What `cargo bench` and `cargo test --benches` prepend.
            "--bench" | "--test" => {}
            other => filters.push(other.to_string()),
        }
    }

    let cases = cases()
        .into_iter()
        .map(|spec| (spec.id(), spec))
        .collect::<Vec<_>>();
    // The scripts select by substring and count the whole process, so two
    // workloads behind one id would be summed with nothing to show it.
    let mut seen = std::collections::HashSet::new();
    for (id, _) in &cases {
        assert!(seen.insert(id.as_str()), "workload id {id} is not unique");
    }

    let selected = cases
        .into_iter()
        .filter(|(id, _)| filters.is_empty() || filters.iter().any(|f| id.contains(f)))
        .collect::<Vec<_>>();
    assert!(!selected.is_empty(), "no workload matched {filters:?}");

    if list {
        for (id, _) in &selected {
            println!("{id}");
        }
        return;
    }

    let width = selected.iter().map(|(id, _)| id.len()).max().unwrap_or(0);
    println!(
        "{:<width$}  {:>12}  {:>12}  {:>6}",
        "workload", "min ns", "median ns", "iters"
    );
    for (id, spec) in &selected {
        let job = prepare(spec);
        let mut samples = measure(&job, iters);
        samples.sort_unstable();
        let min = samples[0].as_nanos();
        let median = samples[samples.len() / 2].as_nanos();
        println!(
            "{id:<width$}  {min:>12}  {median:>12}  {:>6}",
            samples.len()
        );
    }
}
