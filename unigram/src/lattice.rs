use rustc_hash::FxHashMap;

#[derive(Debug, Clone)]
pub struct LatticeNode {
    pub token: String,
    pub token_id: u32,
    pub log_prob: f64,
    pub start: usize,
    pub end: usize,
}

/// Viterbi sobre um lattice de segmentação.
///
/// Retorna a segmentação de máxima log-probabilidade da `sentence` normalizada,
/// usando `token_index: token_string → (id, log_prob)`.
///
/// Se nenhuma segmentação for possível (token de 1 char ausente do índice),
/// retorna uma segmentação char-a-char com log_prob=-100 como fallback.
pub fn viterbi(sentence: &str, token_index: &FxHashMap<String, (u32, f64)>) -> Vec<LatticeNode> {
    let chars: Vec<char> = sentence.chars().collect();
    let n = chars.len();
    if n == 0 {
        return vec![];
    }

    const NEG_INF: f64 = f64::NEG_INFINITY;
    let mut dp: Vec<f64> = vec![NEG_INF; n + 1];
    let mut back: Vec<Option<LatticeNode>> = (0..=n).map(|_| None).collect();
    dp[0] = 0.0;

    for end in 1..=n {
        for start in 0..end {
            if dp[start] == NEG_INF {
                continue;
            }
            let substr: String = chars[start..end].iter().collect();
            if let Some(&(id, lp)) = token_index.get(&substr) {
                let score = dp[start] + lp;
                if score > dp[end] {
                    dp[end] = score;
                    back[end] = Some(LatticeNode {
                        token: substr,
                        token_id: id,
                        log_prob: lp,
                        start,
                        end,
                    });
                }
            }
        }
    }

    if dp[n] == NEG_INF {
        // Fallback: cada char individualmente com log_prob baixo
        return chars
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let s = c.to_string();
                let (id, lp) = token_index.get(&s).copied().unwrap_or((u32::MAX, -100.0));
                LatticeNode { token: s, token_id: id, log_prob: lp, start: i, end: i + 1 }
            })
            .collect();
    }

    // Reconstrói o caminho de trás para frente
    let mut path = Vec::new();
    let mut pos = n;
    while pos > 0 {
        if let Some(node) = back[pos].take() {
            pos = node.start;
            path.push(node);
        } else {
            break;
        }
    }
    path.reverse();
    path
}

/// Acumula contagens de tokens via Viterbi em paralelo (rayon).
pub fn viterbi_token_counts(
    sentences: &[String],
    token_index: &FxHashMap<String, (u32, f64)>,
) -> FxHashMap<u32, f64> {
    use rayon::prelude::*;
    sentences
        .par_iter()
        .map(|s| {
            let path = viterbi(s, token_index);
            let mut counts: FxHashMap<u32, f64> = FxHashMap::default();
            for node in path {
                if node.token_id != u32::MAX {
                    *counts.entry(node.token_id).or_insert(0.0) += 1.0;
                }
            }
            counts
        })
        .reduce(FxHashMap::default, |mut a, b| {
            for (k, v) in b {
                *a.entry(k).or_insert(0.0) += v;
            }
            a
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_index() -> FxHashMap<String, (u32, f64)> {
        let mut m = FxHashMap::default();
        m.insert("▁".to_string(), (0, -1.0));
        m.insert("a".to_string(), (1, -1.0));
        m.insert("b".to_string(), (2, -1.0));
        m.insert("▁a".to_string(), (3, -0.5)); // mais provável
        m.insert("▁ab".to_string(), (4, -0.1)); // mais provável ainda
        m
    }

    #[test]
    fn viterbi_chooses_best_path() {
        let idx = simple_index();
        let path = viterbi("▁ab", &idx);
        // "▁ab" tem log_prob=-0.1 > "▁a"+"b"=(-0.5)+(-1.0)=-1.5
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].token, "▁ab");
    }
}
