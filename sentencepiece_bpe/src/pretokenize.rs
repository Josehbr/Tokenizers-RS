/// Normaliza uma sentença para o formato SentencePiece:
/// - Substitui espaços (e tabulações) por ▁ (U+2581)
/// - Colapsa espaços consecutivos num único ▁
/// - Prefixa a primeira palavra com ▁ (marcador de início)
///
/// Exemplos:
/// - `"hello world"` → `"▁hello▁world"`
/// - `"  hello  world  "` → `"▁hello▁world"`
/// - `"tokenização"` → `"▁tokenização"`
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_word_gets_prefix() {
        assert_eq!(normalize_sentence("hello"), "▁hello");
    }

    #[test]
    fn spaces_become_separator() {
        assert_eq!(normalize_sentence("hello world"), "▁hello▁world");
    }

    #[test]
    fn multiple_spaces_collapsed() {
        assert_eq!(normalize_sentence("hello  world"), "▁hello▁world");
    }

    #[test]
    fn leading_trailing_spaces_ignored() {
        assert_eq!(normalize_sentence("  hello  "), "▁hello");
    }

    #[test]
    fn empty_string() {
        assert_eq!(normalize_sentence(""), "");
    }

    #[test]
    fn marker_separates_words_in_tokens() {
        // Qualidade: ▁ deve aparecer como prefixo de início de cada palavra.
        let normalized = normalize_sentence("hello world foo");
        assert!(normalized.starts_with('▁'));
        // Conta marcadores: 3 palavras → 3 marcadores ▁
        let count = normalized.chars().filter(|&c| c == '▁').count();
        assert_eq!(count, 3);
    }

    #[test]
    fn single_word_produces_one_marker() {
        let normalized = normalize_sentence("palavra");
        let count = normalized.chars().filter(|&c| c == '▁').count();
        assert_eq!(count, 1);
    }
}
