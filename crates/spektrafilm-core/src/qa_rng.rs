//! Fixed NumPy 2.4.6 QA streams. Choice follows Generator's tail-shuffle /
//! Floyd paths and Lemire's bounded integers; no generic seed API is needed.
//! https://github.com/numpy/numpy/blob/v2.4.6/numpy/random/_generator.pyx
//! https://github.com/numpy/numpy/blob/v2.4.6/numpy/random/src/distributions/distributions.c
#[path = "qa_normal_tables.rs"]
mod normal_tables;
struct Pcg {
    state: u128,
    increment: u128,
    cached: Option<u32>,
}
impl Pcg {
    fn zero() -> Self {
        Self {
            state: 35399562948360463058890781895381311971,
            increment: 87136372517582989555478159403783844777,
            cached: None,
        }
    }
    fn reference() -> Self {
        Self {
            state: 300936217048912661423338589542005123575,
            increment: 247270855025706022817204781204357037513,
            cached: None,
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(47026247687942121848144207491837523525)
            .wrapping_add(self.increment);
        ((self.state >> 64) as u64 ^ self.state as u64).rotate_right((self.state >> 122) as u32)
    }
    fn uniform_double(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1. / 9007199254740992.)
    }
    fn normal(&mut self) -> f64 {
        use normal_tables::*;
        loop {
            let word = self.next_u64();
            let index = (word & 255) as usize;
            let negative = ((word >> 8) & 1) != 0;
            let magnitude = (word >> 9) & 0x000f_ffff_ffff_ffff;
            let x = magnitude as f64 * WI_DOUBLE[index] * if negative { -1. } else { 1. };
            if magnitude < KI_DOUBLE[index] {
                return x;
            }
            if index == 0 {
                loop {
                    let excess = -ZIGGURAT_NOR_INV_R * (-self.uniform_double()).ln_1p();
                    let rejection = -(-self.uniform_double()).ln_1p();
                    if 2. * rejection > excess * excess {
                        let value = ZIGGURAT_NOR_R + excess;
                        return if ((magnitude >> 8) & 1) != 0 {
                            -value
                        } else {
                            value
                        };
                    }
                }
            }
            if (FI_DOUBLE[index - 1] - FI_DOUBLE[index]) * self.uniform_double() + FI_DOUBLE[index]
                < (-0.5 * x * x).exp()
            {
                return x;
            }
        }
    }
    fn next_u32(&mut self) -> u32 {
        if let Some(value) = self.cached.take() {
            value
        } else {
            let word = self.next_u64();
            self.cached = Some((word >> 32) as u32);
            word as u32
        }
    }
    fn bounded(&mut self, upper: usize) -> usize {
        if upper == 0 {
            return 0;
        }
        assert!(upper < u32::MAX as usize);
        let span = upper as u32 + 1;
        let threshold = (u32::MAX - upper as u32) % span;
        loop {
            let product = self.next_u32() as u64 * span as u64;
            if product as u32 >= threshold {
                return (product >> 32) as usize;
            }
        }
    }
    fn shuffle_tail(&mut self, values: &mut [usize], first: usize) {
        for i in (first..values.len()).rev() {
            let j = self.bounded(i);
            values.swap(i, j);
        }
    }
}
pub(crate) fn reference_samples(count: usize) -> Vec<[f64; 3]> {
    let mut rng = Pcg::reference();
    (0..count)
        .map(|_| {
            std::array::from_fn(|_| {
                (((rng.next_u64() >> 11) as f64 * (1. / 9007199254740992.)) as f32) as f64
            })
        })
        .collect()
}
pub(crate) fn hull_indices(population: usize) -> Vec<usize> {
    let count = population.min(8000);
    let mut rng = Pcg::zero();
    if population > 10000 && count > population / 50 {
        let mut values: Vec<_> = (0..population).collect();
        rng.shuffle_tail(&mut values, (population - count).max(1));
        return values.split_off(population - count);
    }
    let mut seen = std::collections::HashSet::with_capacity(count);
    let mut out = Vec::with_capacity(count);
    for j in population - count..population {
        let candidate = rng.bounded(j);
        let value = if seen.insert(candidate) {
            candidate
        } else {
            seen.insert(j);
            j
        };
        out.push(value);
    }
    rng.shuffle_tail(&mut out, 1);
    out
}
#[cfg(test)]
mod tests {
    #[test]
    fn hull_sampling_matches_numpy_choice_across_both_algorithms() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/hull_sampling.json")).unwrap();
        for case in fixture.as_array().unwrap() {
            let population = case["population"].as_u64().unwrap() as usize;
            let expected: Vec<usize> = serde_json::from_value(case["indices"].clone()).unwrap();
            assert_eq!(
                super::hull_indices(population),
                expected,
                "population {population}"
            );
        }
    }
}

pub(crate) fn normal_values(count: usize) -> Vec<f64> {
    let mut rng = Pcg::zero();
    (0..count).map(|_| rng.normal()).collect()
}
#[cfg(test)]
mod normal_tests {
    #[test]
    fn seeded_normal_field_matches_numpy_including_rejection_tails() {
        let expected: Vec<f64> =
            serde_json::from_str(include_str!("../tests/fixtures/normal_field.json")).unwrap();
        let actual = super::normal_values(expected.len());
        for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
            assert!((a - b).abs() < 1e-14, "normal sample {i}: {a} vs {b}");
        }
    }
}
