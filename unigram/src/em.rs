use crate::lattice::{viterbi, viterbi_token_counts};
use crate::pretokenize::normalize_sentence;
use rustc_hash::FxHashMap;

pub struct UnigramModel {
    pub vocab: Vec<String>,
    pub log_probs: Vec<f64>,
    /// `token_index[token] = (id, log_prob)` — reconstruído a cada atualização.
    pub token_index: FxHashMap<String, (u32, f64)>,
    /// Tokens protegidos da poda: chars de 1 grafema e tokens especiais.
    protected: Vec<bool>,
}

impl UnigramModel {
    /// Cria o modelo a partir de um vocabulário inicial com probabilidades uniformes.
    pub fn from_vocab(vocab: Vec<(String, f64)>) -> Self {
        let n = vocab.len();
        let log_uniform = -(n as f64).ln();

        let mut strs = Vec::with_capacity(n);
        let mut log_probs = Vec::with_capacity(n);
        let mut token_index: FxHashMap<String, (u32, f64)> = FxHashMap::default();
        let mut protected = Vec::with_capacity(n);

        for (i, (s, _)) in vocab.into_iter().enumerate() {
            let lp = log_uniform;
            token_index.insert(s.clone(), (i as u32, lp));
            let is_protected = s.chars().count() == 1 || s.starts_with('<');
            strs.push(s);
            log_probs.push(lp);
            protected.push(is_protected);
        }

        UnigramModel { vocab: strs, log_probs, token_index, protected }
    }

    fn rebuild_index(&mut self) {
        self.token_index.clear();
        for (i, s) in self.vocab.iter().enumerate() {
            self.token_index.insert(s.clone(), (i as u32, self.log_probs[i]));
        }
    }

    /// M-step: normaliza contagens → novos log_probs.
    /// Pseudocount 1e-10 evita log(0).
    pub fn update_probs(&mut self, token_counts: &FxHashMap<u32, f64>) {
        let total: f64 = token_counts.values().sum::<f64>() + 1e-10 * self.vocab.len() as f64;
        for (i, lp) in self.log_probs.iter_mut().enumerate() {
            let count = token_counts.get(&(i as u32)).copied().unwrap_or(0.0) + 1e-10;
            *lp = (count / total).ln();
        }
        self.rebuild_index();
    }

    /// Log-verossimilhança do corpus: Σ_i log P*(x_i).
    pub fn corpus_log_prob(&self, sentences: &[String]) -> f64 {
        sentences
            .iter()
            .map(|s| {
                let norm = normalize_sentence(s);
                viterbi(&norm, &self.token_index)
                    .iter()
                    .map(|n| n.log_prob)
                    .sum::<f64>()
            })
            .sum()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// Uma iteração de EM (hard Viterbi):
/// - E-step: segmentação ótima de cada sentença via Viterbi
/// - M-step: normaliza contagens acumuladas
///
/// Retorna `(log_likelihood, token_counts)`.
pub fn em_step(
    model: &mut UnigramModel,
    sentences: &[String],
) -> (f64, FxHashMap<u32, f64>) {
    let normalized: Vec<String> = sentences.iter().map(|s| normalize_sentence(s)).collect();

    // E-step
    let token_counts = viterbi_token_counts(&normalized, &model.token_index);

    // M-step
    model.update_probs(&token_counts);

    let ll = model.corpus_log_prob(sentences);
    (ll, token_counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pretokenize::build_initial_vocab;

    fn simple_corpus() -> Vec<String> {
        std::iter::repeat("abcabc".to_string()).take(30).collect()
    }

    #[test]
    fn em_step_updates_probs() {
        // Qualidade: depois de um passo EM, os log_probs devem mudar (o modelo aprendeu).
        let corpus = simple_corpus();
        let vocab = build_initial_vocab(&corpus, 50);
        let mut model = UnigramModel::from_vocab(vocab);

        let initial_lp: Vec<f64> = model.log_probs.clone();
        em_step(&mut model, &corpus);

        let changed = model.log_probs.iter().zip(initial_lp.iter())
            .any(|(new, old)| (new - old).abs() > 1e-10);
        assert!(changed, "EM step deve atualizar as probabilidades");
    }

    #[test]
    fn em_improves_log_likelihood() {
        // Qualidade: mais iterações de EM devem melhorar (ou manter) a log-verossimilhança.
        let corpus = simple_corpus();
        let vocab = build_initial_vocab(&corpus, 50);
        let mut model = UnigramModel::from_vocab(vocab);

        let ll_before = model.corpus_log_prob(&corpus);
        em_step(&mut model, &corpus);
        let ll_after = model.corpus_log_prob(&corpus);

        assert!(ll_after >= ll_before - 1e-6,
            "EM deve não piorar a log-verossimilhança: {ll_before:.4} → {ll_after:.4}");
    }

    #[test]
    fn prune_reduces_vocab_size() {
        // Qualidade: prune_vocab deve remover tokens e reduzir o tamanho do vocab.
        let corpus = simple_corpus();
        let vocab = build_initial_vocab(&corpus, 50);
        let mut model = UnigramModel::from_vocab(vocab);
        let (_, counts) = em_step(&mut model, &corpus);

        let size_before = model.vocab.len();
        let removed = prune_vocab(&mut model, &counts, 0.2);
        let size_after = model.vocab.len();

        assert!(removed > 0, "deve remover ao menos 1 token");
        assert_eq!(size_before - removed, size_after);
    }

    #[test]
    fn prune_preserves_single_char_tokens() {
        // Qualidade: tokens de 1 char nunca devem ser removidos (fallback do Viterbi).
        let corpus = simple_corpus();
        let vocab = build_initial_vocab(&corpus, 50);
        let mut model = UnigramModel::from_vocab(vocab);
        let (_, counts) = em_step(&mut model, &corpus);

        // Poda agressiva
        prune_vocab(&mut model, &counts, 0.5);

        let single_chars: Vec<_> = model.vocab.iter()
            .filter(|t| t.chars().count() == 1)
            .collect();
        assert!(!single_chars.is_empty(), "tokens de 1 char devem sobreviver à poda");
    }
}

/// Poda: remove `(vocab_size * prune_ratio)` tokens com menor contribuição marginal.
///
/// Contribuição de t: `|count(t) * log_prob(t)|`. Tokens com menor contribuição
/// (mais próximos de zero) são removidos primeiro. Tokens protegidos são mantidos.
///
/// Retorna o número de tokens removidos.
pub fn prune_vocab(
    model: &mut UnigramModel,
    token_counts: &FxHashMap<u32, f64>,
    prune_ratio: f64,
) -> usize {
    let n = model.vocab.len();
    let n_to_remove = ((n as f64) * prune_ratio) as usize;
    if n_to_remove == 0 {
        return 0;
    }

    // score = |count * log_prob| — menor = menos importante
    let mut scores: Vec<(usize, f64, bool)> = model
        .log_probs
        .iter()
        .enumerate()
        .map(|(i, &lp)| {
            let count = token_counts.get(&(i as u32)).copied().unwrap_or(0.0);
            let score = (count * lp).abs();
            (i, score, model.protected[i])
        })
        .collect();

    // Menor score → remover primeiro
    scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut to_remove: Vec<bool> = vec![false; n];
    let mut removed = 0;
    for (i, _score, is_protected) in &scores {
        if removed >= n_to_remove {
            break;
        }
        if !is_protected {
            to_remove[*i] = true;
            removed += 1;
        }
    }

    // Reconstrói vocab sem os removidos
    let new_vocab: Vec<String> = model
        .vocab
        .iter()
        .enumerate()
        .filter(|(i, _)| !to_remove[*i])
        .map(|(_, s)| s.clone())
        .collect();
    let new_log_probs: Vec<f64> = model
        .log_probs
        .iter()
        .enumerate()
        .filter(|(i, _)| !to_remove[*i])
        .map(|(_, &lp)| lp)
        .collect();
    let new_protected: Vec<bool> = model
        .protected
        .iter()
        .enumerate()
        .filter(|(i, _)| !to_remove[*i])
        .map(|(_, &p)| p)
        .collect();

    model.vocab = new_vocab;
    model.log_probs = new_log_probs;
    model.protected = new_protected;
    model.rebuild_index();

    removed
}
