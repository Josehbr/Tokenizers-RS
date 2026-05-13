use crate::pair_stats::{unpack_pair, PairStats};
use rustc_hash::FxHashMap;

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub num_merges: usize,
    pub min_pair_count: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self { num_merges: 25, min_pair_count: 2 }
    }
}

pub type MergeRule = (String, String);

struct Vocab {
    strs: Vec<String>,
    by_str: FxHashMap<String, u32>,
}

impl Vocab {
    fn new() -> Self {
        Self { strs: Vec::new(), by_str: FxHashMap::default() }
    }
    fn intern(&mut self, s: String) -> u32 {
        if let Some(&id) = self.by_str.get(&s) {
            return id;
        }
        let id = self.strs.len() as u32;
        self.by_str.insert(s.clone(), id);
        self.strs.push(s);
        id
    }
}

/// Treina BPE clássico: a cada passo mescla o par de tokens mais frequente.
pub fn train_bpe(corpus_words: &[String], cfg: &TrainConfig) -> Vec<MergeRule> {
    let mut vocab = Vocab::new();

    let mut words: Vec<Vec<u32>> = corpus_words
        .iter()
        .filter_map(|w| {
            if w.is_empty() {
                return None;
            }
            let ids: Vec<u32> = w.chars().map(|c| vocab.intern(c.to_string())).collect();
            if ids.is_empty() { None } else { Some(ids) }
        })
        .collect();

    let mut rules: Vec<MergeRule> = Vec::with_capacity(cfg.num_merges);

    for _ in 0..cfg.num_merges {
        let stats = PairStats::from_words(&words);
        let Some((left_id, right_id)) = best_pair_by_freq(&stats, &vocab, cfg.min_pair_count)
        else {
            break;
        };
        let left = vocab.strs[left_id as usize].clone();
        let right = vocab.strs[right_id as usize].clone();
        let merged = format!("{left}{right}");
        let merged_id = vocab.intern(merged);
        apply_merge(&mut words, left_id, right_id, merged_id);
        rules.push((left, right));
    }

    rules
}

fn best_pair_by_freq(stats: &PairStats, vocab: &Vocab, min_pair_count: u64) -> Option<(u32, u32)> {
    let mut candidates: Vec<(u64, u32, u32)> = Vec::with_capacity(stats.pair_counts.len());
    for (&pair_key, &count) in &stats.pair_counts {
        if count < min_pair_count {
            continue;
        }
        let (a, b) = unpack_pair(pair_key);
        candidates.push((count, a, b));
    }
    if candidates.is_empty() {
        return None;
    }
    // Desempate determinístico: maior frequência; em empate, menor par lexicográfico.
    candidates.sort_by(|x, y| {
        y.0.cmp(&x.0)
            .then_with(|| vocab.strs[x.1 as usize].as_str().cmp(vocab.strs[y.1 as usize].as_str()))
            .then_with(|| vocab.strs[x.2 as usize].as_str().cmp(vocab.strs[y.2 as usize].as_str()))
    });
    let (_, a, b) = candidates[0];
    Some((a, b))
}

/// Two-pointer in-place: substitui (left, right) por merged em cada palavra.
fn apply_merge(words: &mut [Vec<u32>], left: u32, right: u32, merged: u32) {
    for w in words.iter_mut() {
        if w.len() < 2 {
            continue;
        }
        let mut write = 0usize;
        let mut read = 0usize;
        let n = w.len();
        while read < n {
            if read + 1 < n && w[read] == left && w[read + 1] == right {
                w[write] = merged;
                write += 1;
                read += 2;
            } else {
                w[write] = w[read];
                write += 1;
                read += 1;
            }
        }
        w.truncate(write);
    }
}

/// Aplica as regras de merge (na ordem) a uma palavra, partindo de caracteres individuais.
pub fn apply_merges_to_word(word: &str, merges: &[MergeRule]) -> Vec<String> {
    let mut seq: Vec<String> = word.chars().map(|c| c.to_string()).collect();
    for (left, right) in merges {
        let merged = format!("{left}{right}");
        let mut write = 0usize;
        let mut read = 0usize;
        while read < seq.len() {
            if read + 1 < seq.len() && seq[read] == *left && seq[read + 1] == *right {
                seq[write] = merged.clone();
                write += 1;
                read += 2;
            } else {
                if write != read {
                    seq[write] = std::mem::take(&mut seq[read]);
                }
                write += 1;
                read += 1;
            }
        }
        seq.truncate(write);
    }
    seq
}

/// Frequências de tokens na segmentação final após todos os merges.
pub fn token_frequencies(corpus_words: &[String], merges: &[MergeRule]) -> FxHashMap<String, u64> {
    let mut m: FxHashMap<String, u64> = FxHashMap::default();
    for w in corpus_words {
        if w.is_empty() {
            continue;
        }
        for t in apply_merges_to_word(w, merges) {
            *m.entry(t).or_insert(0) += 1;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bpe_merges_most_frequent_pair() {
        // "aab" aparece 3×, "ac" aparece 1× → par "a","a" deve ser mesclado primeiro
        let corpus: Vec<String> = vec!["aab".into(), "aab".into(), "aab".into(), "ac".into()];
        let cfg = TrainConfig { num_merges: 1, min_pair_count: 2 };
        let rules = train_bpe(&corpus, &cfg);
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0], ("a".to_string(), "a".to_string()));
    }

    #[test]
    fn apply_merges_to_word_basic() {
        let rules = vec![("a".to_string(), "b".to_string())];
        let tokens = apply_merges_to_word("abc", &rules);
        assert_eq!(tokens, vec!["ab", "c"]);
    }

    #[test]
    fn more_merges_means_fewer_tokens() {
        // Qualidade: mais merges → melhor compressão (menos tokens por palavra).
        let corpus: Vec<String> = std::iter::repeat("abcabc".to_string()).take(100).collect();
        let cfg_few = TrainConfig { num_merges: 1, min_pair_count: 2 };
        let cfg_more = TrainConfig { num_merges: 3, min_pair_count: 2 };

        let rules_few = train_bpe(&corpus, &cfg_few);
        let rules_more = train_bpe(&corpus, &cfg_more);

        let tokens_few = apply_merges_to_word("abcabc", &rules_few).len();
        let tokens_more = apply_merges_to_word("abcabc", &rules_more).len();

        assert!(tokens_more <= tokens_few, "mais merges deve produzir igual ou menos tokens");
    }

    #[test]
    fn merge_is_deterministic() {
        // O mesmo corpus deve sempre produzir as mesmas regras (desempate lexicográfico).
        let corpus: Vec<String> = vec!["ab".into(), "ab".into(), "cd".into(), "cd".into()];
        let cfg = TrainConfig { num_merges: 2, min_pair_count: 2 };
        let rules1 = train_bpe(&corpus, &cfg);
        let rules2 = train_bpe(&corpus, &cfg);
        assert_eq!(rules1, rules2);
    }

    #[test]
    fn compression_ratio_bpe_vs_charwise() {
        // BPE deve produzir menos tokens que a segmentação char-a-char.
        let corpus: Vec<String> = std::iter::repeat("prefixo".to_string()).take(50).collect();
        let cfg = TrainConfig { num_merges: 4, min_pair_count: 2 };
        let rules = train_bpe(&corpus, &cfg);
        let tokens = apply_merges_to_word("prefixo", &rules).len();
        let chars = "prefixo".chars().count();
        assert!(tokens < chars, "BPE deve ter menos tokens que chars individuais");
    }
}
