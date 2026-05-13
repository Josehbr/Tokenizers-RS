use rustc_hash::FxHashMap;

/// Normalização idêntica ao SentencePiece: espaços → ▁, colapsa consecutivos.
pub fn normalize_sentence(sentence: &str) -> String {
    let mut result = String::with_capacity(sentence.len() + 4);
    let mut prev_was_space = true;
    for c in sentence.chars() {
        if c == ' ' || c == '\t' {
            prev_was_space = true;
        } else {
            if prev_was_space {
                result.push('▁');
                prev_was_space = false;
            }
            result.push(c);
        }
    }
    result
}

/// Constrói vocabulário inicial com os top-`max_vocab` substrings mais frequentes.
///
/// Tokens de 1 char são sempre incluídos (necessários como fallback do Viterbi).
/// Substrings de tamanho 2–16 são adicionados por frequência até `max_vocab`.
pub fn build_initial_vocab(sentences: &[String], max_vocab: usize) -> Vec<(String, f64)> {
    let mut substr_freqs: FxHashMap<String, u64> = FxHashMap::default();
    let mut char_freqs: FxHashMap<String, u64> = FxHashMap::default();

    for sentence in sentences {
        let norm = normalize_sentence(sentence);
        let chars: Vec<char> = norm.chars().collect();
        let n = chars.len();
        for start in 0..n {
            let mut substr = String::new();
            for &c in &chars[start..n.min(start + 16)] {
                substr.push(c);
                if substr.chars().count() == 1 {
                    *char_freqs.entry(substr.clone()).or_insert(0) += 1;
                }
                *substr_freqs.entry(substr.clone()).or_insert(0) += 1;
            }
        }
    }

    // Sempre incluir todos os tokens de 1 char (fallback do Viterbi)
    let mut vocab: Vec<(String, f64)> = char_freqs.keys().map(|k| (k.clone(), 0.0)).collect();
    let char_set: FxHashMap<&str, ()> = vocab.iter().map(|(s, _)| (s.as_str(), ())).collect();

    // Adicionar substrings maiores por frequência decrescente
    let mut extras: Vec<(String, u64)> = substr_freqs
        .into_iter()
        .filter(|(s, _)| !char_set.contains_key(s.as_str()))
        .collect();
    extras.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let remaining = max_vocab.saturating_sub(vocab.len());
    for (s, _) in extras.into_iter().take(remaining) {
        vocab.push((s, 0.0));
    }

    // Probabilidade inicial uniforme
    let n = vocab.len() as f64;
    for (_, p) in &mut vocab {
        *p = 1.0 / n;
    }

    vocab
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_adds_prefix() {
        assert_eq!(normalize_sentence("ab"), "▁ab");
    }

    #[test]
    fn normalize_spaces_to_marker() {
        assert_eq!(normalize_sentence("a b"), "▁a▁b");
    }

    #[test]
    fn build_vocab_includes_single_chars() {
        let sentences = vec!["ab".to_string()];
        let vocab = build_initial_vocab(&sentences, 100);
        let tokens: Vec<&str> = vocab.iter().map(|(s, _)| s.as_str()).collect();
        // ▁ e ▁a e ▁b e a e b devem aparecer
        assert!(tokens.contains(&"▁"));
        assert!(tokens.contains(&"a"));
        assert!(tokens.contains(&"b"));
    }
}
