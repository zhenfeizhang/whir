use std::ops::Index;

// use ark_ff::Field;
// use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
// use serde::{Deserialize, Serialize};
use arith::Field;

use super::{lagrange_iterator::LagrangePolynomialIterator, multilinear::MultilinearPoint};

/// Represents a multilinear polynomial `f` in `num_variables` unknowns, stored via its evaluations
/// over the hypercube `{0,1}^{num_variables}`.
///
/// The vector `evals` contains function evaluations at **lexicographically ordered** points.
// #[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default, Debug, Clone, PartialEq, Eq)]
// #[serde(bound = "F: CanonicalSerialize + CanonicalDeserialize")]
pub struct EvaluationsList<F> {
    /// Stores evaluations in **lexicographic order**.
    // #[serde(with = "crate::ark_serde")]
    evals: Vec<F>,
    /// Number of variables in the multilinear polynomial.
    /// Ensures `evals.len() = 2^{num_variables}`.
    num_variables: usize,
}

impl<F> EvaluationsList<F>
where
    F: Field,
{
    /// Constructs an `EvaluationsList` from a given vector of evaluations.
    ///
    /// - The `evals` vector must have a **length that is a power of two** since it represents
    ///   evaluations over an `n`-dimensional binary hypercube.
    /// - The ordering of evaluation points follows **lexicographic order**.
    ///
    /// **Mathematical Constraint:**
    /// If `evals.len() = 2^n`, then `num_variables = n`, ensuring correct indexing.
    ///
    /// **Panics:**
    /// - If `evals.len()` is **not** a power of two.
    pub fn new(evals: Vec<F>) -> Self {
        let len = evals.len();
        assert!(
            len.is_power_of_two(),
            "Evaluation list length must be a power of two."
        );

        Self {
            evals,
            num_variables: len.ilog2() as usize,
        }
    }

    /// Evaluates the polynomial at a given multilinear point.
    ///
    /// - If `point` belongs to the binary hypercube `{0,1}^n`, we directly return the precomputed
    ///   evaluation.
    /// - Otherwise, we **reconstruct** the evaluation using Lagrange interpolation.
    ///
    /// Mathematical definition:
    /// Given evaluations `f(x)` stored in `evals`, we compute:
    ///
    /// ```ignore
    /// f(p) = Σ_{x ∈ {0,1}^n} eq(x, p) * f(x)
    /// ```
    ///
    /// where `eq(x, p)` is the Lagrange basis polynomial.
    pub fn evaluate(&self, point: &MultilinearPoint<F>) -> F {
        if let Some(binary_index) = point.to_hypercube() {
            return self.evals[binary_index.0];
        }

        self.evals
            .iter()
            .zip(LagrangePolynomialIterator::from(point))
            .map(|(eval, (_, lag))| *eval * lag)
            .sum()
    }

    pub fn eval_extension(&self, point: &MultilinearPoint<F>) -> F {
        if let Some(point) = point.to_hypercube() {
            return self.evals[point.0];
        }
        eval_multilinear(&self.evals, &point.0)
    }

    /// Returns an immutable reference to the evaluations vector.
    #[allow(clippy::missing_const_for_fn)]
    pub fn evals(&self) -> &[F] {
        &self.evals
    }

    /// Returns a mutable reference to the evaluations vector.
    #[allow(clippy::missing_const_for_fn)]
    pub fn evals_mut(&mut self) -> &mut [F] {
        &mut self.evals
    }

    /// Returns the total number of stored evaluations.
    ///
    /// Mathematical Invariant:
    /// ```ignore
    /// num_evals = 2^{num_variables}
    /// ```
    pub fn num_evals(&self) -> usize {
        self.evals.len()
    }

    /// Returns the number of variables in the multilinear polynomial.
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    pub fn to_coeffs(&self) -> crate::poly_utils::coeffs::CoefficientList<F> {
        let mut coeffs = self.evals.clone();
        crate::ntt::inverse_wavelet_transform(&mut coeffs);
        crate::poly_utils::coeffs::CoefficientList::new(coeffs)
    }
}

impl<F> Index<usize> for EvaluationsList<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.evals[index]
    }
}

fn eval_multilinear<F: Field>(evals: &[F], point: &[F]) -> F {
    debug_assert_eq!(evals.len(), 1 << point.len());
    match point {
        [] => evals[0],
        [x] => evals[0] + (evals[1] - evals[0]) * *x,
        [x0, x1] => {
            let a0 = evals[0] + (evals[1] - evals[0]) * *x1;
            let a1 = evals[2] + (evals[3] - evals[2]) * *x1;
            a0 + (a1 - a0) * *x0
        }
        [x0, x1, x2] => {
            let a00 = evals[0] + (evals[1] - evals[0]) * *x2;
            let a01 = evals[2] + (evals[3] - evals[2]) * *x2;
            let a10 = evals[4] + (evals[5] - evals[4]) * *x2;
            let a11 = evals[6] + (evals[7] - evals[6]) * *x2;
            let a0 = a00 + (a01 - a00) * *x1;
            let a1 = a10 + (a11 - a10) * *x1;
            a0 + (a1 - a0) * *x0
        }
        [x0, x1, x2, x3] => {
            let a000 = evals[0] + (evals[1] - evals[0]) * *x3;
            let a001 = evals[2] + (evals[3] - evals[2]) * *x3;
            let a010 = evals[4] + (evals[5] - evals[4]) * *x3;
            let a011 = evals[6] + (evals[7] - evals[6]) * *x3;
            let a100 = evals[8] + (evals[9] - evals[8]) * *x3;
            let a101 = evals[10] + (evals[11] - evals[10]) * *x3;
            let a110 = evals[12] + (evals[13] - evals[12]) * *x3;
            let a111 = evals[14] + (evals[15] - evals[14]) * *x3;
            let a00 = a000 + (a001 - a000) * *x2;
            let a01 = a010 + (a011 - a010) * *x2;
            let a10 = a100 + (a101 - a100) * *x2;
            let a11 = a110 + (a111 - a110) * *x2;
            let a0 = a00 + (a01 - a00) * *x1;
            let a1 = a10 + (a11 - a10) * *x1;
            a0 + (a1 - a0) * *x0
        }
        [x, tail @ ..] => {
            let (f0, f1) = evals.split_at(evals.len() / 2);
            #[cfg(not(feature = "parallel"))]
            let (f0, f1) = (eval_multilinear(f0, tail), eval_multilinear(f1, tail));
            #[cfg(feature = "parallel")]
            let (f0, f1) = {
                let work_size: usize = (1 << 15) / std::mem::size_of::<F>();
                if evals.len() > work_size {
                    rayon::join(|| eval_multilinear(f0, tail), || eval_multilinear(f1, tail))
                } else {
                    (eval_multilinear(f0, tail), eval_multilinear(f1, tail))
                }
            };
            f0 + (f1 - f0) * *x
        }
    }
}

#[cfg(test)]
#[allow(clippy::should_panic_without_expect)]
mod tests {
    // use ark_ff::AdditiveGroup;
    use arith::Field;
    use goldilocks::Goldilocks;

    use super::*;
    use crate::poly_utils::hypercube::BinaryHypercube;

    #[test]
    fn test_new_evaluations_list() {
        let evals = vec![
            Goldilocks::ZERO,
            Goldilocks::ONE,
            Goldilocks::ZERO,
            Goldilocks::ONE,
        ];
        let evaluations_list = EvaluationsList::new(evals.clone());

        assert_eq!(evaluations_list.num_evals(), evals.len());
        assert_eq!(evaluations_list.num_variables(), 2);
        assert_eq!(evaluations_list.evals(), &evals);
    }

    #[test]
    #[should_panic]
    fn test_new_evaluations_list_invalid_length() {
        // Length is not a power of two, should panic
        let _ = EvaluationsList::new(vec![Goldilocks::ONE, Goldilocks::ZERO, Goldilocks::ONE]);
    }

    #[test]
    fn test_indexing() {
        let evals = vec![
            Goldilocks::from(1u32),
            Goldilocks::from(2u32),
            Goldilocks::from(3u32),
            Goldilocks::from(4u32),
        ];
        let evaluations_list = EvaluationsList::new(evals.clone());

        assert_eq!(evaluations_list[0], evals[0]);
        assert_eq!(evaluations_list[1], evals[1]);
        assert_eq!(evaluations_list[2], evals[2]);
        assert_eq!(evaluations_list[3], evals[3]);
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds() {
        let evals = vec![
            Goldilocks::ZERO,
            Goldilocks::ONE,
            Goldilocks::ZERO,
            Goldilocks::ONE,
        ];
        let evaluations_list = EvaluationsList::new(evals);

        let _ = evaluations_list[4]; // Index out of range, should panic
    }

    #[test]
    fn test_mutability_of_evals() {
        let mut evals = EvaluationsList::new(vec![
            Goldilocks::ZERO,
            Goldilocks::ONE,
            Goldilocks::ZERO,
            Goldilocks::ONE,
        ]);

        assert_eq!(evals.evals()[1], Goldilocks::ONE);

        evals.evals_mut()[1] = Goldilocks::from(5u32);

        assert_eq!(evals.evals()[1], Goldilocks::from(5u32));
    }

    #[test]
    fn test_evaluate_on_hypercube_points() {
        let evaluations_vec = vec![
            Goldilocks::ZERO,
            Goldilocks::ONE,
            Goldilocks::ZERO,
            Goldilocks::ONE,
        ];
        let evals = EvaluationsList::new(evaluations_vec.clone());

        for i in BinaryHypercube::new(2) {
            assert_eq!(
                evaluations_vec[i.0],
                evals.evaluate(&MultilinearPoint::from_binary_hypercube_point(i, 2))
            );
        }
    }

    #[test]
    fn test_evaluate_on_non_hypercube_points() {
        let evals = EvaluationsList::new(vec![
            Goldilocks::from(1u32),
            Goldilocks::from(2u32),
            Goldilocks::from(3u32),
            Goldilocks::from(4u32),
        ]);

        let point = MultilinearPoint(vec![Goldilocks::from(2u32), Goldilocks::from(3u32)]);

        let result = evals.evaluate(&point);

        // The result should be computed using Lagrange interpolation.
        let expected = LagrangePolynomialIterator::from(&point)
            .map(|(b, lag)| lag * evals[b.0])
            .sum();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_evaluate_edge_cases() {
        let e1 = Goldilocks::from(7u32);
        let e2 = Goldilocks::from(8u32);
        let e3 = Goldilocks::from(9u32);
        let e4 = Goldilocks::from(10u32);

        let evals = EvaluationsList::new(vec![e1, e2, e3, e4]);

        // Evaluating at a binary hypercube point should return the direct value
        assert_eq!(
            evals.evaluate(&MultilinearPoint(vec![Goldilocks::ZERO, Goldilocks::ZERO])),
            e1
        );
        assert_eq!(
            evals.evaluate(&MultilinearPoint(vec![Goldilocks::ZERO, Goldilocks::ONE])),
            e2
        );
        assert_eq!(
            evals.evaluate(&MultilinearPoint(vec![Goldilocks::ONE, Goldilocks::ZERO])),
            e3
        );
        assert_eq!(
            evals.evaluate(&MultilinearPoint(vec![Goldilocks::ONE, Goldilocks::ONE])),
            e4
        );
    }

    #[test]
    fn test_num_evals() {
        let evals = EvaluationsList::new(vec![
            Goldilocks::ONE,
            Goldilocks::ZERO,
            Goldilocks::ONE,
            Goldilocks::ZERO,
        ]);
        assert_eq!(evals.num_evals(), 4);
    }

    #[test]
    fn test_num_variables() {
        let evals = EvaluationsList::new(vec![
            Goldilocks::ONE,
            Goldilocks::ZERO,
            Goldilocks::ONE,
            Goldilocks::ZERO,
        ]);
        assert_eq!(evals.num_variables(), 2);
    }

    #[test]
    fn test_eval_extension_on_hypercube_points() {
        let evals = vec![
            Goldilocks::from(1u32),
            Goldilocks::from(2u32),
            Goldilocks::from(3u32),
            Goldilocks::from(4u32),
        ];
        let eval_list = EvaluationsList::new(evals.clone());

        for i in BinaryHypercube::new(2) {
            assert_eq!(
                eval_list.eval_extension(&MultilinearPoint::from_binary_hypercube_point(i, 2)),
                evals[i.0]
            );
        }
    }

    #[test]
    fn test_eval_extension_on_non_hypercube_points() {
        let evals = EvaluationsList::new(vec![
            Goldilocks::from(1u32),
            Goldilocks::from(2u32),
            Goldilocks::from(3u32),
            Goldilocks::from(4u32),
        ]);

        let point = MultilinearPoint(vec![Goldilocks::from(2u32), Goldilocks::from(3u32)]);

        let result = evals.eval_extension(&point);

        // Expected result using `eval_multilinear`
        let expected = eval_multilinear(evals.evals(), &point.0);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_eval_multilinear_1d() {
        let a = Goldilocks::from(5u32);
        let b = Goldilocks::from(10u32);
        let evals = vec![a, b];

        // Evaluate at midpoint `x = 1/2`
        let x = Goldilocks::INV_2;
        let expected = a + (b - a) * x;

        assert_eq!(eval_multilinear(&evals, &[x]), expected);
    }

    #[test]
    fn test_eval_multilinear_2d() {
        let a = Goldilocks::from(1u32);
        let b = Goldilocks::from(2u32);
        let c = Goldilocks::from(3u32);
        let d = Goldilocks::from(4u32);

        // The evaluations are stored in lexicographic order for (x, y)
        // f(0,0) = a, f(0,1) = c, f(1,0) = b, f(1,1) = d
        let evals = vec![a, b, c, d];

        // Evaluate at `(x, y) = (1/2, 1/2)`
        let x = Goldilocks::INV_2;
        let y = Goldilocks::INV_2;

        // Interpolation formula:
        // f(x, y) = (1-x)(1-y) * f(0,0) + (1-x)y * f(0,1) + x(1-y) * f(1,0) + xy * f(1,1)
        let expected = (Goldilocks::ONE - x) * (Goldilocks::ONE - y) * a
            + (Goldilocks::ONE - x) * y * c
            + x * (Goldilocks::ONE - y) * b
            + x * y * d;

        assert_eq!(eval_multilinear(&evals, &[x, y]), expected);
    }

    #[test]
    fn test_eval_multilinear_3d() {
        let a = Goldilocks::from(1u32);
        let b = Goldilocks::from(2u32);
        let c = Goldilocks::from(3u32);
        let d = Goldilocks::from(4u32);
        let e = Goldilocks::from(5u32);
        let f = Goldilocks::from(6u32);
        let g = Goldilocks::from(7u32);
        let h = Goldilocks::from(8u32);

        // The evaluations are stored in lexicographic order for (x, y, z)
        // f(0,0,0) = a, f(0,0,1) = c, f(0,1,0) = b, f(0,1,1) = e
        // f(1,0,0) = d, f(1,0,1) = f, f(1,1,0) = g, f(1,1,1) = h
        let evals = vec![a, b, c, e, d, f, g, h];

        let x = Goldilocks::from(1u32) * Goldilocks::from(3u32).inv().unwrap();
        let y = Goldilocks::from(1u32) * Goldilocks::from(3u32).inv().unwrap();
        let z = Goldilocks::from(1u32) * Goldilocks::from(3u32).inv().unwrap();

        // Using trilinear interpolation formula:
        let expected = (Goldilocks::ONE - x) * (Goldilocks::ONE - y) * (Goldilocks::ONE - z) * a
            + (Goldilocks::ONE - x) * (Goldilocks::ONE - y) * z * c
            + (Goldilocks::ONE - x) * y * (Goldilocks::ONE - z) * b
            + (Goldilocks::ONE - x) * y * z * e
            + x * (Goldilocks::ONE - y) * (Goldilocks::ONE - z) * d
            + x * (Goldilocks::ONE - y) * z * f
            + x * y * (Goldilocks::ONE - z) * g
            + x * y * z * h;

        assert_eq!(eval_multilinear(&evals, &[x, y, z]), expected);
    }

    #[test]
    fn test_eval_multilinear_4d() {
        let a = Goldilocks::from(1u32);
        let b = Goldilocks::from(2u32);
        let c = Goldilocks::from(3u32);
        let d = Goldilocks::from(4u32);
        let e = Goldilocks::from(5u32);
        let f = Goldilocks::from(6u32);
        let g = Goldilocks::from(7u32);
        let h = Goldilocks::from(8u32);
        let i = Goldilocks::from(9u32);
        let j = Goldilocks::from(10u32);
        let k = Goldilocks::from(11u32);
        let l = Goldilocks::from(12u32);
        let m = Goldilocks::from(13u32);
        let n = Goldilocks::from(14u32);
        let o = Goldilocks::from(15u32);
        let p = Goldilocks::from(16u32);

        // Evaluations stored in lexicographic order for (x, y, z, w)
        let evals = vec![a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p];

        let x = Goldilocks::from(1u32) * Goldilocks::from(2u32).inv().unwrap();
        let y = Goldilocks::from(2u32) * Goldilocks::from(3u32).inv().unwrap();
        let z = Goldilocks::from(1u32) * Goldilocks::from(4u32).inv().unwrap();
        let w = Goldilocks::from(3u32) * Goldilocks::from(5u32).inv().unwrap();

        // Quadlinear interpolation formula
        let expected = (Goldilocks::ONE - x)
            * (Goldilocks::ONE - y)
            * (Goldilocks::ONE - z)
            * (Goldilocks::ONE - w)
            * a
            + (Goldilocks::ONE - x) * (Goldilocks::ONE - y) * (Goldilocks::ONE - z) * w * b
            + (Goldilocks::ONE - x) * (Goldilocks::ONE - y) * z * (Goldilocks::ONE - w) * c
            + (Goldilocks::ONE - x) * (Goldilocks::ONE - y) * z * w * d
            + (Goldilocks::ONE - x) * y * (Goldilocks::ONE - z) * (Goldilocks::ONE - w) * e
            + (Goldilocks::ONE - x) * y * (Goldilocks::ONE - z) * w * f
            + (Goldilocks::ONE - x) * y * z * (Goldilocks::ONE - w) * g
            + (Goldilocks::ONE - x) * y * z * w * h
            + x * (Goldilocks::ONE - y) * (Goldilocks::ONE - z) * (Goldilocks::ONE - w) * i
            + x * (Goldilocks::ONE - y) * (Goldilocks::ONE - z) * w * j
            + x * (Goldilocks::ONE - y) * z * (Goldilocks::ONE - w) * k
            + x * (Goldilocks::ONE - y) * z * w * l
            + x * y * (Goldilocks::ONE - z) * (Goldilocks::ONE - w) * m
            + x * y * (Goldilocks::ONE - z) * w * n
            + x * y * z * (Goldilocks::ONE - w) * o
            + x * y * z * w * p;

        // Validate against the function output
        assert_eq!(eval_multilinear(&evals, &[x, y, z, w]), expected);
    }
}
