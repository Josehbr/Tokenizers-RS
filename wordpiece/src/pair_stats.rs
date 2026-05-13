use rustc_hash::FxHashMap;

/// Contagens de pares e tokens individuais.
/// WordPiece precisa de ambas para calcular freq(ab) / (freq(a) × freq(b)).
#[derive(Debug, Default)]
pub struct PairStats {
    pub pair_counts: FxHashMap<u64, u64>,
    pub token_counts: FxHashMap<u32, u64>,
}

#[inline(always)]
pub fn pack_pair(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | (b as u64)
}

#[inline(always)]
pub fn unpack_pair(p: u64) -> (u32, u32) {
    ((p >> 32) as u32, (p & 0xFFFF_FFFF) as u32)
}

/// Score WordPiece: `freq(ab) / (freq(a) × freq(b))`.
/// Equivalente ao PMI sem os denominadores globais B e T (constantes no argmax).
#[inline]
pub fn wordpiece_score(c_ab: u64, c_a: u64, c_b: u64) -> f64 {
    if c_a == 0 || c_b == 0 {
        return 0.0;
    }
    (c_ab as f64) / ((c_a as f64) * (c_b as f64))
}

impl PairStats {
    pub fn from_words(words: &[Vec<u32>]) -> Self {
        use rayon::prelude::*;
        words
            .par_chunks(2048)
            .map(|chunk| {
                let mut s = PairStats::default();
                for w in chunk {
                    for &t in w {
                        *s.token_counts.entry(t).or_insert(0) += 1;
                    }
                    for win in w.windows(2) {
                        let key = pack_pair(win[0], win[1]);
                        *s.pair_counts.entry(key).or_insert(0) += 1;
                    }
                }
                s
            })
            .reduce(PairStats::default, |mut a, b| {
                for (k, v) in b.token_counts {
                    *a.token_counts.entry(k).or_insert(0) += v;
                }
                for (k, v) in b.pair_counts {
                    *a.pair_counts.entry(k).or_insert(0) += v;
                }
                a
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_proportional_to_cooccurrence() {
        // freq(ab)=10, freq(a)=10, freq(b)=10 → score=0.1
        assert!((wordpiece_score(10, 10, 10) - 0.1).abs() < 1e-9);
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let (a, b) = (0xABCD_1234u32, 0x5678_9ABCu32);
        assert_eq!(unpack_pair(pack_pair(a, b)), (a, b));
    }
}
