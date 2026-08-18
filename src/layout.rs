pub(crate) trait Width: Copy + Send + Sync {
    fn ndim(self) -> usize;
}

#[derive(Clone, Copy)]
pub(crate) struct Dyn(pub(crate) usize);

#[derive(Clone, Copy)]
pub(crate) struct Fixed<const N: usize>;

impl Width for Dyn {
    #[inline(always)]
    fn ndim(self) -> usize {
        self.0
    }
}

impl<const N: usize> Width for Fixed<N> {
    #[inline(always)]
    fn ndim(self) -> usize {
        N
    }
}

/// The width list is the call site's own tuning table; nothing requires two call
/// sites to agree on it.
macro_rules! with_width {
    ($ndim:expr, [$($n:literal),* $(,)?], |$w:ident| $body:expr) => {{
        match $ndim {
            $( $n => { let $w = $crate::layout::Fixed::<$n>; $body } )*
            other => { let $w = $crate::layout::Dyn(other); $body }
        }
    }};
}
pub(crate) use with_width;

#[derive(Clone, Copy)]
pub(crate) struct Rows<'a, W: Width = Dyn> {
    flat: &'a [f64],
    width: W,
}

impl<'a, W: Width> Rows<'a, W> {
    #[inline(always)]
    pub(crate) fn new(flat: &'a [f64], width: W) -> Self {
        debug_assert!(flat.len().is_multiple_of(width.ndim()));
        Self { flat, width }
    }

    #[inline(always)]
    pub(crate) fn ndim(self) -> usize {
        self.width.ndim()
    }

    #[inline(always)]
    pub(crate) fn flat(self) -> &'a [f64] {
        self.flat
    }

    #[inline(always)]
    pub(crate) fn slice(self, start: usize, end: usize) -> Self {
        let d = self.width.ndim();
        Self {
            flat: &self.flat[start * d..end * d],
            width: self.width,
        }
    }

    #[inline(always)]
    pub(crate) fn iter(self) -> impl ExactSizeIterator<Item = &'a [f64]> {
        self.flat.chunks_exact(self.width.ndim())
    }
}

pub(crate) struct RowsMut<'a, W: Width = Dyn> {
    flat: &'a mut [f64],
    width: W,
}

impl<'a, W: Width> RowsMut<'a, W> {
    #[inline(always)]
    pub(crate) fn new(flat: &'a mut [f64], width: W) -> Self {
        debug_assert!(flat.len().is_multiple_of(width.ndim()));
        Self { flat, width }
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.flat.len() / self.width.ndim()
    }

    #[inline(always)]
    pub(crate) fn as_ref(&self) -> Rows<'_, W> {
        Rows {
            flat: self.flat,
            width: self.width,
        }
    }

    #[inline(always)]
    pub(crate) fn reborrow(&mut self) -> RowsMut<'_, W> {
        RowsMut {
            flat: self.flat,
            width: self.width,
        }
    }

    #[inline(always)]
    pub(crate) fn coord(&self, row: usize, dim: usize) -> f64 {
        self.flat[row * self.width.ndim() + dim]
    }

    #[inline(always)]
    pub(crate) fn split_at(self, mid: usize) -> (Self, Self) {
        let d = self.width.ndim();
        let (left, right) = self.flat.split_at_mut(mid * d);
        (
            RowsMut {
                flat: left,
                width: self.width,
            },
            RowsMut {
                flat: right,
                width: self.width,
            },
        )
    }

    #[inline(always)]
    pub(crate) fn swap(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        let d = self.width.ndim();
        debug_assert!(a * d + d <= self.flat.len() && b * d + d <= self.flat.len());
        // SAFETY: `a != b` makes the two `d`-element rows disjoint, and both lie
        // inside `flat` per the assertion above.
        unsafe {
            let base = self.flat.as_mut_ptr();
            std::ptr::swap_nonoverlapping(base.add(a * d), base.add(b * d), d);
        }
    }
}

/// Offset from a coordinate to the interval `[lo, hi]`; zero inside.
#[inline(always)]
pub(crate) fn axis_offset(q: f64, lo: f64, hi: f64) -> f64 {
    (lo - q).max(q - hi).max(0.0)
}

#[derive(Clone, Copy)]
pub(crate) struct BBox<'a> {
    pub(crate) lo: &'a [f64],
    pub(crate) hi: &'a [f64],
}

impl<'a> BBox<'a> {
    #[inline(always)]
    pub(crate) fn ndim(self) -> usize {
        self.lo.len()
    }

    #[inline(always)]
    pub(crate) fn extent(self, dim: usize) -> f64 {
        self.hi[dim] - self.lo[dim]
    }

    pub(crate) fn widest_axis(self) -> usize {
        let mut best_dim = 0;
        let mut best_span = self.extent(0);
        for dim in 1..self.ndim() {
            let span = self.extent(dim);
            if span > best_span {
                best_span = span;
                best_dim = dim;
            }
        }
        best_dim
    }
}

pub(crate) struct BBoxMut<'a> {
    pub(crate) lo: &'a mut [f64],
    pub(crate) hi: &'a mut [f64],
}

impl BBoxMut<'_> {
    #[inline(always)]
    pub(crate) fn as_ref(&self) -> BBox<'_> {
        BBox {
            lo: self.lo,
            hi: self.hi,
        }
    }
}

/// One box per node, `[lo[0..ndim] | hi[0..ndim]]` each, indexed by node id.
#[derive(PartialEq)]
pub(crate) struct Boxes {
    values: Vec<f64>,
    ndim: usize,
}

impl Boxes {
    pub(crate) fn zeroed(n_nodes: usize, ndim: usize) -> Self {
        Self {
            values: vec![0.0_f64; 2 * ndim * n_nodes],
            ndim,
        }
    }

    #[inline(always)]
    pub(crate) fn of(&self, id: u32) -> BBox<'_> {
        let base = 2 * self.ndim * id as usize;
        let (lo, hi) = self.values[base..base + 2 * self.ndim].split_at(self.ndim);
        BBox { lo, hi }
    }

    pub(crate) fn as_mut(&mut self) -> BoxesMut<'_> {
        BoxesMut {
            values: &mut self.values,
            ndim: self.ndim,
        }
    }
}

pub(crate) struct BoxesMut<'a> {
    values: &'a mut [f64],
    ndim: usize,
}

impl<'a> BoxesMut<'a> {
    #[inline(always)]
    pub(crate) fn first_mut(&mut self) -> BBoxMut<'_> {
        let (lo, hi) = self.values[..2 * self.ndim].split_at_mut(self.ndim);
        BBoxMut { lo, hi }
    }

    #[inline(always)]
    pub(crate) fn first(&self) -> BBox<'_> {
        let (lo, hi) = self.values[..2 * self.ndim].split_at(self.ndim);
        BBox { lo, hi }
    }

    #[inline(always)]
    pub(crate) fn split_first(self) -> (BBoxMut<'a>, BoxesMut<'a>) {
        let (own, rest) = self.values.split_at_mut(2 * self.ndim);
        let (lo, hi) = own.split_at_mut(self.ndim);
        (
            BBoxMut { lo, hi },
            BoxesMut {
                values: rest,
                ndim: self.ndim,
            },
        )
    }

    #[inline(always)]
    pub(crate) fn split_at(self, n_nodes: usize) -> (Self, Self) {
        let (left, right) = self.values.split_at_mut(2 * self.ndim * n_nodes);
        (
            BoxesMut {
                values: left,
                ndim: self.ndim,
            },
            BoxesMut {
                values: right,
                ndim: self.ndim,
            },
        )
    }

    #[inline(always)]
    pub(crate) fn reborrow(&mut self) -> BoxesMut<'_> {
        BoxesMut {
            values: self.values,
            ndim: self.ndim,
        }
    }
}
