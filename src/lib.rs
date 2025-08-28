//! fpcoa: Principal Coordinate Analysis (PCoA) with fixed-rank randomized SVD.
//! Assumptions:
//! - `dist` is dense, strictly symmetric, with zeros on the diagonal.
//! Pipeline (Halko–Martinsson–Tropp range finder + symmetric core):
//! 1) Build B = -0.5 * J * (D ∘ D) * J   (double-centering)
//! 2) Q = subspace_iteration_full(B, rank = k + oversample, nbiter = q)   (annembed/lax)
//! 3) B_r = Qᵀ * B * Q           (r × r, symmetric)
//! 4) Symmetric eig (nalgebra):  B_r = V Λ Vᵀ    (Λ sorted ↓)
//! 5) Zero-out negative eigenvalues (and their eigenvectors)
//! 6) U ≈ Q * V_k
//! 7) Coordinates = U * sqrt(Λ_k)

use ndarray::{s, Array1, Array2, Axis};
use ndarray::parallel::prelude::*;
use std::cmp::Ordering;

// randomized subspace iteration (range finder) from annembed (uses lax under the hood)
use annembed::tools::svdapprox::subspace_iteration_full;

// small symmetric eigen (fast, r is tiny)
use nalgebra as na;

/// Result of PCoA.
pub struct PCoAResult {
    /// Approximated eigenvalues λᵢ of the centered matrix (after zeroing negatives).
    pub eigenvalues: Array1<f64>,        // length k_kept
    /// Sample scores: N × k_kept (U_k * sqrt(Λ_k)).
    pub coordinates: Array2<f64>,        // (n, k_kept)
    /// Proportion explained: λᵢ / trace(B).
    pub proportion_explained: Array1<f64>,
}

/// Options for randomized PCoA.
#[derive(Clone, Copy, Debug)]
pub struct FpcoaOptions {
    /// Number of output components k (will be clamped to [1, n]).
    pub k: usize,
    /// Oversampling added to k when sketching rank r = k + oversample, typically >= 2.
    pub oversample: usize,
    /// Number of subspace (power) iterations q (QR steps). Typical: 1–3.
    pub nbiter: usize,
    /// Symmetrize D as 0.5*(D + Dᵀ) before centering (safety).
    pub symmetrize_input: bool,
}

impl Default for FpcoaOptions {
    fn default() -> Self {
        Self {
            k: 10,
            oversample: 8,
            nbiter: 2,
            symmetrize_input: true,
        }
    }
}

/// Perform PCoA via randomized range finder + symmetric eig on the small core.
/// * `dist`: N×N dense symmetric distance matrix, with zeros on the diagonal.
/// * `opts.k`: number of returned components.
pub fn pcoa_randomized(dist: &Array2<f64>, opts: FpcoaOptions) -> PCoAResult {
    let (n, m) = dist.dim();
    assert_eq!(n, m, "distance matrix must be square");
    assert!(opts.k > 0, "k must be > 0");

    // Optional symmetry enforcement (numerical safety)
    let d = if opts.symmetrize_input {
        let mut sym = dist.clone();
        sym.zip_mut_with(&dist.t().to_owned(), |a, b| *a = 0.5 * (*a + *b));
        sym
    } else {
        dist.clone()
    };

    // E = -0.5 * (D ∘ D), collect row means and global mean
    let mut b = Array2::<f64>::zeros((n, n));
    let (row_means, global_mean) = e_matrix_means(&d, &mut b);

    // B = E - row_means - col_means + global_mean
    f_matrix_inplace(&row_means, global_mean, &mut b);

    // Total variance = trace(B)
    let _trace_b: f64 = b.diag().sum();

    // randomized range finder (fixed-rank)
    let k_wanted = opts.k.min(n);
    let r = (k_wanted + opts.oversample).min(n).max(k_wanted); // sketch rank

    // Q: (n × r), orthonormal columns
    let q = subspace_iteration_full::<f64>(&b, r, opts.nbiter);

    // small symmetric core:B_r=QᵀBQ 
    let mut b_small = q.t().dot(&b).dot(&q); // (r × r)
    symmetrize_inplace_upper(&mut b_small);  // numerical safety

    // symmetric eig (nalgebra)
    let (mut vals, mut vecs) = symmetric_eigh_small(&b_small); // vals len r; vecs r×r (columns = eigenvectors)

    // sort eigenpairs in descending order
    sort_eig_desc_inplace(&mut vals, &mut vecs);

    // zero out negatives
    let zero_thresh = 0.0_f64;
    let num_pos = vals.iter().take_while(|x| **x >= zero_thresh).count();
    if num_pos < vals.len() {
        for i in num_pos..vals.len() {
            vals[i] = 0.0;
            let mut col = vecs.column_mut(i);
            col.fill(0.0);
        }
    }

    // truncate to top-k
    let k_keep = k_wanted.min(vals.len()).min(vecs.ncols());
    let vals_k = vals.slice_move(s![..k_keep]);
    let vecs_k = vecs.slice_move(s![.., ..k_keep]); // (r × k)

    // U ≈ Q * V_k  => (n × k)
    let u_left = q.dot(&vecs_k);

    // Coordinates = U * sqrt(Λ_k)
    let sqrt_vals = vals_k.mapv(f64::sqrt);
    let mut coords = u_left.clone();
    for (mut col, s) in coords.axis_iter_mut(Axis(1)).zip(sqrt_vals.iter()) {
        if *s > 0.0 {
            col *= *s;
        } else {
            col.fill(0.0);
        }
    }

    // Proportion explained
    let denom = vals_k.sum().max(1e-300);   // sum of returned (k) eigenvalues
    let prop  = &vals_k / denom;

    PCoAResult {
        eigenvalues: vals_k,
        coordinates: coords,
        proportion_explained: prop,
    }
}

/// Compute E = -0.5 * (D ∘ D), along with row means and global mean(E).
/// Parallelized by rows without borrowing the whole matrix mutably.
fn e_matrix_means(dist: &Array2<f64>, centered_out: &mut Array2<f64>) -> (Array1<f64>, f64) {
    let n = dist.nrows();
    assert_eq!(n, dist.ncols());
    assert_eq!(centered_out.dim(), (n, n));

    let row_sums: Vec<f64> = centered_out
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .map(|(i, mut row)| {
            let mut sum = 0.0;
            let di = dist.row(i);
            for j in 0..n {
                let dij = di[j];
                let e = -0.5 * dij * dij;
                row[j] = e;
                sum += e;
            }
            sum
        })
        .collect();

    let row_means = Array1::from_iter(row_sums.iter().map(|s| *s / (n as f64)));
    let global_mean = row_sums.iter().sum::<f64>() / (n as f64) / (n as f64);
    (row_means, global_mean)
}

/// Double-centering: B = E - row_means - col_means + global_mean, in-place.
/// Parallelized by rows.
fn f_matrix_inplace(row_means: &Array1<f64>, global_mean: f64, centered: &mut Array2<f64>) {
    let n = centered.nrows();
    assert_eq!(n, centered.ncols());
    assert_eq!(row_means.len(), n);

    centered
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let gr_mean = global_mean - row_means[i];
            for j in 0..n {
                row[j] += gr_mean - row_means[j];
            }
        });
}

/// Force strict symmetry by mirroring the upper triangle into the lower (and averaging).
fn symmetrize_inplace_upper(a: &mut Array2<f64>) {
    let n = a.nrows();
    debug_assert_eq!(n, a.ncols());
    for i in 0..n {
        for j in (i + 1)..n {
            let s = 0.5 * (a[[i, j]] + a[[j, i]]);
            a[[i, j]] = s;
            a[[j, i]] = s;
        }
    }
}

/// Symmetric eigen on an r×r matrix using nalgebra (fast enough; r is small).
/// Returns (vals, vecs) with eigenvectors in columns.
fn symmetric_eigh_small(mat: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let n = mat.nrows();
    debug_assert_eq!(n, mat.ncols());

    // nalgebra expects column-major DMatrix; build from row slice is fine.
    let dm = na::DMatrix::<f64>::from_row_slice(n, n, mat.as_slice().expect("contiguous"));

    // nalgebra’s SymmetricEigen returns ascending eigenvalues; eigenvectors as columns.
    let se = na::SymmetricEigen::new(dm);

    // Copy out
    let mut vals = Array1::<f64>::zeros(n);
    for i in 0..n {
        vals[i] = se.eigenvalues[i];
    }
    let mut vecs = Array2::<f64>::zeros((n, n));
    let ev = se.eigenvectors; // n×n
    for c in 0..n {
        for r in 0..n {
            vecs[[r, c]] = ev[(r, c)];
        }
    }
    (vals, vecs)
}

/// Sort eigenpairs in **descending** order, in-place.
/// `vals` length = r. `vecs` is r×r with eigenvectors in columns.
fn sort_eig_desc_inplace(vals: &mut Array1<f64>, vecs: &mut Array2<f64>) {
    let r = vals.len();
    let mut idx: Vec<usize> = (0..r).collect();
    idx.sort_unstable_by(|&i, &j| {
        vals[j]
            .partial_cmp(&vals[i])
            .unwrap_or(Ordering::Equal)
    });

    let vals_sorted = Array1::from_iter(idx.iter().map(|&i| vals[i]));
    let mut vecs_sorted = Array2::<f64>::zeros((r, r));
    for (new_c, &old_c) in idx.iter().enumerate() {
        let src = vecs.column(old_c);
        let mut dst = vecs_sorted.column_mut(new_c);
        dst.assign(&src);
    }

    *vals = vals_sorted;
    *vecs = vecs_sorted;
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use pcoa::apply_pcoa;
    use pcoa::nalgebra as na; 
    #[test]
    fn test_pcoa() {
        // load TSV distance matrix
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/test_dm.tsv");
        let text = std::fs::read_to_string(path).expect("failed to read data/test_dm.tsv");

        let mut lines = text.lines();
        let header = lines.next().expect("empty file");
        let headers: Vec<&str> = header.split('\t').map(str::trim).filter(|s| !s.is_empty()).collect();
        let n = headers.len();
        assert!(n > 0);

        let mut dm = Array2::<f64>::zeros((n, n));
        let mut i = 0usize;
        for line in lines {
            let line = line.trim_end();
            if line.is_empty() { continue; }
            let mut toks = line.split('\t').map(str::trim).filter(|s| !s.is_empty());
            let _row_label = toks.next().expect("missing row label");
            for (j, tok) in toks.enumerate() {
                dm[[i, j]] = tok.parse::<f64>().expect("parse float");
            }
            i += 1;
        }
        assert_eq!(i, n, "row count != header count");

        // randomized PCoA 
        let k = 2;
        // oversample [8,10] as recommened in Halko–Martinsson–Tropp paper
        let opts = super::FpcoaOptions { k, oversample: 8, nbiter: 2, symmetrize_input: true };
        let rand_out = super::pcoa_randomized(&dm, opts);
        let x = rand_out.coordinates.clone(); // (n × k)

        // exact PCoA
        let mut row_major = Vec::with_capacity(n * n);
        for r in 0..n { for c in 0..n { row_major.push(dm[[r, c]]); } }
        let dm_na = na::DMatrix::<f64>::from_row_slice(n, n, &row_major);

        let exact = apply_pcoa(dm_na, k).expect("exact pcoa failed");
        let y_na = if exact.nrows() == k && exact.ncols() == n { exact.transpose() } else { exact };
        let mut y = Array2::<f64>::zeros((n, k));
        for r in 0..n { for c in 0..k { y[[r, c]] = y_na[(r, c)]; } }

        // Procrustes (orthogonal) alignment: R = argmin || X R - Y ||_F
        let mut cxy = na::DMatrix::<f64>::zeros(k, k);
        for a in 0..k {
            for b in 0..k {
                let mut s = 0.0;
                for r in 0..n { s += x[[r, a]] * y[[r, b]]; }
                cxy[(a, b)] = s;
            }
        }
        let svd = na::SVD::new(cxy.clone(), true, true);
        let (u, v_t) = (svd.u.expect("svd u"), svd.v_t.expect("svd vt"));
        let rmat = &u * &v_t; // k×k

        // X_aligned = X * R
        let mut x_aligned = Array2::<f64>::zeros((n, k));
        for r in 0..n {
            for c in 0..k {
                let mut s = 0.0;
                for t in 0..k { s += x[[r, t]] * rmat[(t, c)]; }
                x_aligned[[r, c]] = s;
            }
        }

        // coordinate error
        let mut diff2 = 0.0;
        let mut ref2  = 0.0;
        for r in 0..n { for c in 0..k {
            let d = x_aligned[[r,c]] - y[[r,c]];
            diff2 += d*d;
            ref2  += y[[r,c]] * y[[r,c]];
        } }
        let rel_coords = (diff2.sqrt()) / ref2.sqrt().max(1e-16);
        assert!(rel_coords < 1e-10, "relative coord error too large: {:.3e}", rel_coords);

        // variance (sum λ) check (rotation invariant)
        let sum_rand: f64 = rand_out.eigenvalues.iter().take(k).sum();
        let sum_exact: f64 = y.iter().map(|v| v*v).sum();
        let rel_var = ((sum_rand - sum_exact).abs()) / sum_exact.max(1e-16);
        assert!(rel_var < 1e-12, "variance (sum λ) mismatch too large: {:.3e}", rel_var);
    }
}