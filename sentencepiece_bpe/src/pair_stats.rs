use rustc_hash::FxHashMap;

#[derive(Debug, Default)]
pub struct PairStats {
    pub pair_counts: FxHashMap<u64, u64>,
}

#[inline(always)]
pub fn pack_pair(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | (b as u64)
}

#[inline(always)]
pub fn unpack_pair(p: u64) -> (u32, u32) {
    ((p >> 32) as u32, (p & 0xFFFF_FFFF) as u32)
}

impl PairStats {
    pub fn from_words(words: &[Vec<u32>]) -> Self {
        use rayon::prelude::*;
        words
            .par_chunks(2048)
            .map(|chunk| {
                let mut s = PairStats::default();
                for w in chunk {
                    for win in w.windows(2) {
                        let key = pack_pair(win[0], win[1]);
                        *s.pair_counts.entry(key).or_insert(0) += 1;
                    }
                }
                s
            })
            .reduce(PairStats::default, |mut a, b| {
                for (k, v) in b.pair_counts {
                    *a.pair_counts.entry(k).or_insert(0) += v;
                }
                a
            })
    }
}
