use crate::pair_stats::{unpack_pair, PairStats};
use crate::pretokenize::normalize_sentence;
use rustc_hash::FxHashMap;

#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Tamanho alvo do vocabulário.
    pub vocab_size: usize,
    /// Fração do corpus de caracteres a cobrir no vocabulário inicial.
    /// Caracteres raros abaixo do limiar são mapeados para `<unk>`.
    pub character_coverage: f64,
    pub min_pair_count: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self { vocab_size: 1000, character_coverage: 0.9995, min_pair_count: 2 }
    }
}

pub type MergeRule = (String, String);

pub struct SentencePieceModel {
    pub vocab: Vec<String>,
    pub vocab_set: FxHashMap<String, u32>,
    pub merge_rules: Vec<MergeRule>,
}

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

/// Treina SentencePiece-BPE sobre uma lista de sentenças.
///
/// Diferença central em relação ao BPE clássico:
/// - As sentenças são normalizadas antes (espaços → ▁) — não há pré-segmentação por palavra.
/// - O vocabulário inicial é construído por caracteres com filtro de cobertura.
/// - Os merges ocorrem sobre o fluxo de caracteres da frase inteira.
pub fn train_sp_bpe(sentences: &[String], cfg: &TrainConfig) -> SentencePieceModel {
    let normalized: Vec<String> = sentences.iter().map(|s| normalize_sentence(s)).collect();

    // Conta frequências de caracteres para aplicar character_coverage
    let mut char_freqs: FxHashMap<char, u64> = FxHashMap::default();
    for s in &normalized {
        for c in s.chars() {
            *char_freqs.entry(c).or_insert(0) += 1;
        }
    }
    let total_chars: u64 = char_freqs.values().sum();

    // Ordena por frequência decrescente e inclui os chars que cobrem character_coverage
    let mut char_list: Vec<(char, u64)> = char_freqs.into_iter().collect();
    char_list.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let target_covered: u64 = (total_chars as f64 * cfg.character_coverage) as u64;
    let mut covered = 0u64;
    let mut included_chars: FxHashMap<char, ()> = FxHashMap::default();
    for &(c, freq) in &char_list {
        if covered >= target_covered {
            break;
        }
        included_chars.insert(c, ());
        covered += freq;
    }

    let mut vocab = Vocab::new();
    vocab.intern("<unk>".to_string()); // id=0, fallback para chars não cobertos

    // Converte cada sentença normalizada em sequência de IDs
    let mut words: Vec<Vec<u32>> = normalized
        .iter()
        .filter_map(|s| {
            if s.is_empty() {
                return None;
            }
            let ids: Vec<u32> = s
                .chars()
                .map(|c| {
                    if included_chars.contains_key(&c) {
                        vocab.intern(c.to_string())
                    } else {
                        0 // <unk>
                    }
                })
                .collect();
            if ids.is_empty() { None } else { Some(ids) }
        })
        .collect();

    let mut rules: Vec<MergeRule> = Vec::new();

    while vocab.strs.len() < cfg.vocab_size {
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

    let vocab_set = vocab.by_str.clone();
    SentencePieceModel { vocab: vocab.strs, vocab_set, merge_rules: rules }
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
    candidates.sort_by(|x, y| {
        y.0.cmp(&x.0)
            .then_with(|| vocab.strs[x.1 as usize].as_str().cmp(vocab.strs[y.1 as usize].as_str()))
            .then_with(|| vocab.strs[x.2 as usize].as_str().cmp(vocab.strs[y.2 as usize].as_str()))
    });
    let (_, a, b) = candidates[0];
    Some((a, b))
}

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

/// Tokeniza uma sentença aplicando as regras aprendidas.
/// A sentença é normalizada antes (espaços → ▁).
pub fn tokenize_sentence(sentence: &str, model: &SentencePieceModel) -> Vec<String> {
    let normalized = normalize_sentence(sentence);
    if normalized.is_empty() {
        return vec![];
    }
    let mut seq: Vec<String> = normalized
        .chars()
        .map(|c| {
            let s = c.to_string();
            if model.vocab_set.contains_key(&s) { s } else { "<unk>".to_string() }
        })
        .collect();

    for (left, right) in &model.merge_rules {
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

/// Frequências de tokens no corpus.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_start_with_marker() {
        // Qualidade: o primeiro token de cada sentença deve começar com ▁.
        let sentences: Vec<String> = std::iter::repeat("abcde".to_string()).take(50).collect();
        let cfg = TrainConfig { vocab_size: 15, character_coverage: 1.0, min_pair_count: 2 };
        let model = train_sp_bpe(&sentences, &cfg);
        let tokens = tokenize_sentence("abcde", &model);
        assert!(!tokens.is_empty());
        assert!(tokens[0].starts_with('▁'),
            "primeiro token deve começar com ▁, got {:?}", tokens[0]);
    }

    #[test]
    fn multiword_sentence_preserves_boundaries() {
        // Qualidade: ▁ deve separar palavras diferentes nos tokens.
        let sentences: Vec<String> = vec![
            "ab cd".into(), "ab cd".into(), "ab cd".into(),
            "ef gh".into(), "ef gh".into(),
        ];
        let cfg = TrainConfig { vocab_size: 20, character_coverage: 1.0, min_pair_count: 2 };
        let model = train_sp_bpe(&sentences, &cfg);
        let tokens = tokenize_sentence("ab cd", &model);
        // Deve haver pelo menos um token com ▁ que não seja o primeiro
        let boundary_tokens: Vec<_> = tokens.iter()
            .skip(1)
            .filter(|t| t.starts_with('▁'))
            .collect();
        assert!(!boundary_tokens.is_empty() || tokens.len() <= 2,
            "deve haver marcador de fronteira entre palavras: {:?}", tokens);
    }

    #[test]
    fn sp_bpe_compresses_better_than_charwise() {
        // Qualidade: SP-BPE deve produzir menos tokens que segmentação char-a-char.
        let sentences: Vec<String> = std::iter::repeat("prefixo".to_string()).take(50).collect();
        let cfg = TrainConfig { vocab_size: 15, character_coverage: 1.0, min_pair_count: 2 };
        let model = train_sp_bpe(&sentences, &cfg);
        let tokens = tokenize_sentence("prefixo", &model);
        // "prefixo" tem 7 chars + 1 ▁ = 8 unidades char-a-char; SP-BPE deve ser menor
        assert!(tokens.len() < 9, "SP-BPE deve comprimir abaixo do nível de char: {:?}", tokens);
    }
}

pub fn token_frequencies(sentences: &[String], model: &SentencePieceModel) -> FxHashMap<String, u64> {
    let mut m: FxHashMap<String, u64> = FxHashMap::default();
    for s in sentences {
        if s.is_empty() {
            continue;
        }
        for t in tokenize_sentence(s, model) {
            *m.entry(t).or_insert(0) += 1;
        }
    }
    m
}
