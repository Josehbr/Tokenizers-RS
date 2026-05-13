use crate::em::{em_step, prune_vocab, UnigramModel};
use crate::lattice::viterbi;
use crate::pretokenize::{build_initial_vocab, normalize_sentence};
use rustc_hash::FxHashMap;

#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Tamanho alvo do vocabulário final.
    pub target_vocab_size: usize,
    /// Vocab inicial = `target_vocab_size × initial_vocab_factor`.
    pub initial_vocab_factor: f64,
    /// Iterações de EM entre cada rodada de poda.
    pub em_iterations: usize,
    /// Fração do vocab a remover por rodada de poda.
    pub prune_ratio: f64,
    pub character_coverage: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            target_vocab_size: 1000,
            initial_vocab_factor: 10.0,
            em_iterations: 5,
            prune_ratio: 0.2,
            character_coverage: 0.9995,
        }
    }
}

/// Treina o modelo Unigram Language Model.
///
/// Algoritmo (Kudo 2018, versão educacional com hard Viterbi EM):
/// 1. Constrói vocab inicial grande (top substrings por frequência)
/// 2. Repete até `|vocab| ≤ target_vocab_size`:
///    a. `em_iterations` iterações de EM (Viterbi → normalização)
///    b. Poda `prune_ratio` dos tokens menos importantes
/// 3. Rodada final de EM para estabilizar probabilidades
pub fn train_unigram(sentences: &[String], cfg: &TrainConfig) -> UnigramModel {
    let initial_max = (cfg.target_vocab_size as f64 * cfg.initial_vocab_factor) as usize;
    let initial_vocab = build_initial_vocab(sentences, initial_max);

    let mut model = UnigramModel::from_vocab(initial_vocab);
    let mut last_counts: FxHashMap<u32, f64> = FxHashMap::default();

    while model.vocab_size() > cfg.target_vocab_size {
        for _ in 0..cfg.em_iterations {
            let (_, counts) = em_step(&mut model, sentences);
            last_counts = counts;
        }
        if model.vocab_size() <= cfg.target_vocab_size {
            break;
        }
        prune_vocab(&mut model, &last_counts, cfg.prune_ratio);
    }

    // Rodada final de EM para estabilizar
    for _ in 0..cfg.em_iterations {
        em_step(&mut model, sentences);
    }

    model
}

/// Tokeniza uma sentença com o modelo treinado (Viterbi).
pub fn tokenize_sentence(sentence: &str, model: &UnigramModel) -> Vec<String> {
    let normalized = normalize_sentence(sentence);
    viterbi(&normalized, &model.token_index)
        .into_iter()
        .map(|n| n.token)
        .collect()
}

/// Frequências de tokens no corpus usando Viterbi.
pub fn token_frequencies(sentences: &[String], model: &UnigramModel) -> FxHashMap<String, u64> {
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
