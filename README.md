# Fast PCoA based on randomized SVD
This crate provides a fast, in-memory Principle Coordinate Analysis (PCoA) (1) based on randomized SVD. 

Note that we rely on the power iteration or subspace iteration type rSVD, instead of the randomized block Krylov method (2), the later has stronger accuracy guarantee but is not pass friendly (3). Fixed rank, instead of fixed precision mode was used for subspace iteration type rSVD due to the same reason.

This crate is the building block for a streaming, pass-friendly PCoA algorithm for millions of samples. 

## Install
Add the below line to your Cargo.toml
```bash
fpcoa = "0.1.0"
```

## Usage
```bash
use fpcoa::{pcoa_randomized, FpcoaOptions};
use ndarray::Array2;

fn main() {
    // Five 2D points
    let pts = [
        [0.0, 0.0], // A
        [1.0, 0.0], // B
        [0.0, 1.0], // C
        [1.0, 1.0], // D
        [2.0, 0.0], // E
    ];

    // Build a symmetric 5×5 distance matrix with zero diagonal
    let n = pts.len();
    let mut dist = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in i + 1..n {
            let dx = pts[i][0] - pts[j][0];
            let dy = pts[i][1] - pts[j][1];
            let d = (dx * dx + dy * dy).sqrt();
            dist[[i, j]] = d;
            dist[[j, i]] = d;
        }
    }

    // Recommended sketch params: oversample ≈ 8–10, nbiter = 2
    let opts = FpcoaOptions {
        k: 2,               // return first 2 axes
        oversample: 8,      // p
        nbiter: 2,          // q
        symmetrize_input: true,
    };

    let res = pcoa_randomized(&dist, opts);

    println!("eigenvalues (top k):     {:?}", res.eigenvalues);
    println!("proportion explained:    {:?}", res.proportion_explained);
    println!("coordinates (n × k):");
    for i in 0..n {
        println!(
            "  {}: [{:8.4}, {:8.4}]",
            (b'A' + i as u8) as char,
            res.coordinates[[i, 0]],
            res.coordinates[[i, 1]]
        );
    }
}
```

## References

1.Legendre, P. and Legendre, L., 2012. Numerical ecology (Vol. 24). Elsevier.

2.Halko, N., Martinsson, P.G. and Tropp, J.A., 2011. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM review, 53(2), pp.217-288.

3.Musco, C. and Musco, C., 2015. Randomized block krylov methods for stronger and faster approximate singular value decomposition. Advances in neural information processing systems, 28.