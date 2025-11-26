// use ark_ff::Field;

use arith::Field;

/// A univariate polynomial represented in coefficient form.
///
/// The coefficient of `x^i` is stored at index `i`.
///
/// Designed for verifier use: avoids parallelism by enforcing sequential Horner evaluation.
/// The verifier should be run on a cheap device.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct WhirDensePolynomial<F: Field> {
    /// The coefficient of `x^i` is stored at location `i` in `self.coeffs`.
    pub coeffs: Vec<F>,
}

impl<F: Field> WhirDensePolynomial<F> {
    /// Constructs a new polynomial from a list of coefficients.
    pub(crate) fn from_coefficients_slice(coeffs: &[F]) -> Self {
        Self::from_coefficients_vec(coeffs.to_vec())
    }

    /// Constructs a new polynomial from a list of coefficients.
    pub(crate) fn from_coefficients_vec(coeffs: Vec<F>) -> Self {
        let mut result = Self { coeffs };
        // While there are zeros at the end of the coefficient vector, pop them off.
        result.truncate_leading_zeros();
        // Check that either the coefficients vec is empty or that the last coeff is
        // non-zero.
        assert!(result.coeffs.last().is_none_or(|coeff| !coeff.is_zero()));
        result
    }

    fn truncate_leading_zeros(&mut self) {
        while self.coeffs.last().is_some_and(|c| c.is_zero()) {
            self.coeffs.pop();
        }
    }

    /// Checks if the given polynomial is zero.
    fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.iter().all(|coeff| coeff.is_zero())
    }

    /// Evaluates `self` at the given `point` in `Self::Point`.
    pub fn evaluate(&self, point: &F) -> F {
        if self.is_zero() {
            return F::ZERO;
        } else if point.is_zero() {
            return self.coeffs[0];
        }
        self.horner_evaluate(point)
    }

    // Horner's method for polynomial evaluation
    fn horner_evaluate(&self, point: &F) -> F {
        self.coeffs
            .iter()
            .rfold(F::zero(), move |result, coeff| result * point + coeff)
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::{AdditiveGroup, Zero};
    use goldilocks::Goldilocks;

    use super::*;
    // use crate::crypto::fields::Goldilocks;

    #[test]
    fn test_zero_polynomial() {
        // A zero polynomial has no coefficients
        let poly = WhirDensePolynomial::<Goldilocks>::from_coefficients_vec(vec![]);
        assert!(poly.is_zero());
        assert_eq!(poly.evaluate(&Goldilocks::from(42u32)), Goldilocks::zero());
    }

    #[test]
    fn test_constant_polynomial() {
        // Polynomial: f(x) = 7
        let c0 = Goldilocks::from(7u32);
        let poly = WhirDensePolynomial::from_coefficients_vec(vec![c0]);

        // f(0)
        assert_eq!(poly.evaluate(&Goldilocks::zero()), c0);
        // f(1)
        assert_eq!(poly.evaluate(&Goldilocks::ONE), c0);
        // f(42)
        assert_eq!(poly.evaluate(&Goldilocks::from(42u32)), c0);
    }

    #[test]
    fn test_linear_polynomial() {
        // Polynomial: f(x) = 3 + 4x
        let c0 = Goldilocks::from(3u32);
        let c1 = Goldilocks::from(4u32);
        let poly = WhirDensePolynomial::from_coefficients_vec(vec![c0, c1]);

        // f(0)
        assert_eq!(poly.evaluate(&Goldilocks::zero()), c0);
        // f(1)
        assert_eq!(poly.evaluate(&Goldilocks::ONE), c0 + c1 * Goldilocks::ONE);
        // f(2)
        assert_eq!(
            poly.evaluate(&Goldilocks::from(2u32)),
            c0 + c1 * Goldilocks::from(2u32)
        );
    }

    #[test]
    fn test_quadratic_polynomial() {
        // Polynomial: f(x) = 2 + 0x + 5x²
        let c0 = Goldilocks::from(2u32);
        let c1 = Goldilocks::from(0u32);
        let c2 = Goldilocks::from(5u32);
        let poly = WhirDensePolynomial::from_coefficients_vec(vec![c0, c1, c2]);

        // f(0)
        assert_eq!(poly.evaluate(&Goldilocks::zero()), c0);
        // f(1)
        assert_eq!(poly.evaluate(&Goldilocks::ONE), c0 + c2);
        // f(2)
        assert_eq!(
            poly.evaluate(&Goldilocks::from(2u32)),
            c0 + c2 * Goldilocks::from(4u32)
        );
    }

    #[test]
    fn test_cubic_polynomial() {
        // Polynomial: f(x) = 1 + 2x + 3x² + 4x³
        let c0 = Goldilocks::from(1u32);
        let c1 = Goldilocks::from(2u32);
        let c2 = Goldilocks::from(3u32);
        let c3 = Goldilocks::from(4u32);
        let poly = WhirDensePolynomial::from_coefficients_vec(vec![c0, c1, c2, c3]);

        // f(0)
        assert_eq!(poly.evaluate(&Goldilocks::zero()), c0);
        // f(1)
        assert_eq!(poly.evaluate(&Goldilocks::ONE), c0 + c1 + c2 + c3);

        // f(2)
        assert_eq!(
            poly.evaluate(&Goldilocks::from(2u32)),
            c0 + c1 * Goldilocks::from(2u32)
                + c2 * Goldilocks::from(4u32)
                + c3 * Goldilocks::from(8u32)
        );
    }

    #[test]
    fn test_leading_zeros_trimmed() {
        // Polynomial: f(x) = 1 + 2x, with trailing zeroes
        let c0 = Goldilocks::from(1u32);
        let c1 = Goldilocks::from(2u32);
        let poly = WhirDensePolynomial::from_coefficients_vec(vec![
            c0,
            c1,
            Goldilocks::ZERO,
            Goldilocks::ZERO,
        ]);

        // Should be trimmed to degree 1
        assert_eq!(poly.coeffs.len(), 2);
        assert_eq!(
            poly.evaluate(&Goldilocks::from(3u32)),
            c0 + c1 * Goldilocks::from(3u32)
        );
    }

    #[test]
    fn test_is_zero_various_cases() {
        let zero_poly = WhirDensePolynomial::<Goldilocks>::from_coefficients_vec(vec![]);
        assert!(zero_poly.is_zero());

        let zero_poly_all_zeros =
            WhirDensePolynomial::<Goldilocks>::from_coefficients_vec(vec![Goldilocks::ZERO; 5]);
        assert!(zero_poly_all_zeros.is_zero());

        let non_zero_poly =
            WhirDensePolynomial::<Goldilocks>::from_coefficients_vec(vec![Goldilocks::ONE]);
        assert!(!non_zero_poly.is_zero());
    }
}
