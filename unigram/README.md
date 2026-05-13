# Unigram Language Model

Implementação do Unigram Tokenizer (Kudo, 2018) em Rust, com `rustc-hash` e `rayon`.

## Fundamento matemático

Ao contrário do BPE/WordPiece que **adicionam** tokens por merge, o Unigram **parte de um vocabulário grande e poda** tokens iterativamente.

Para uma segmentação `S = (t₁, ..., tₘ)`, a probabilidade é:

```
P(S) = ∏ p(tⱼ)
```

O objetivo é encontrar o vocabulário `V` que maximize:

```
Σᵢ log P*(xᵢ | V)
```

onde `P*(xᵢ)` é a segmentação mais provável (Viterbi) de cada sentença xᵢ.

### Algoritmo (versão educacional com hard Viterbi EM)

1. **Vocab inicial**: top-K substrings por frequência (`K = target_size × initial_vocab_factor`)
2. **Repetir até** `|vocab| ≤ target_vocab_size`:
   - **E-step**: segmentação ótima de cada sentença via Viterbi → acumula contagens
   - **M-step**: normaliza contagens → novos `log_probs`
   - **Poda**: remove `prune_ratio` dos tokens com menor contribuição marginal
3. **Rodada final de EM** para estabilizar as probabilidades

**Viterbi**: programação dinâmica que encontra a segmentação de máxima log-probabilidade em O(L²) para uma sentença de comprimento L.

**Tokens protegidos**: chars de 1 grafema e tokens especiais nunca são removidos — são o fallback universal do Viterbi.

**Normalização com ▁** (igual ao SentencePiece): espaços são substituídos por ▁, sem fronteiras de palavra explícitas.

## API pública

| Função / Tipo | Descrição |
|---|---|
| `train_unigram(sentences, cfg) -> UnigramModel` | Treina o modelo |
| `tokenize_sentence(sentence, model) -> Vec<String>` | Tokeniza via Viterbi |
| `token_frequencies(sentences, model) -> FxHashMap<String, u64>` | Contagem de tokens |
| `UnigramModel { vocab, log_probs, token_index }` | Modelo treinado |
| `TrainConfig { target_vocab_size, initial_vocab_factor, em_iterations, prune_ratio }` | Configuração |

## Exemplo em Rust

```rust
use unigram::{train_unigram, tokenize_sentence, TrainConfig};

let sentences: Vec<String> = /* lista de frases */;
let cfg = TrainConfig { target_vocab_size: 500, ..Default::default() };
let model = train_unigram(&sentences, &cfg);
let tokens = tokenize_sentence("aprendizado de máquina", &model);
// Exemplo: ["▁", "ap", "r", "en", "d", "iz", "ado", "▁de", "▁", "m", "áquina"]
```

## Benchmark e CLI

```bash
# Benchmark (mais lento — use --max-lines para testar rapidamente)
cargo run -p unigram --release -- --max-lines 10000 --vocab-size 500

# Tokenizar uma sentença
cargo run -p unigram --release -- --max-lines 10000 --vocab-size 500 --tokenize "aprendizado"

# Salvar vocabulário com log_probs
cargo run -p unigram --release -- --max-lines 10000 --vocab-size 500 --dump-vocab /tmp/unigram_vocab.tsv
```

### Flags

| Flag | Default | Descrição |
|---|---|---|
| `--file <CAMINHO>` | `/usr/share/dict/brazilian` | Arquivo (uma sentença por linha) |
| `--max-lines <N>` | (sem limite) | Limita as primeiras N linhas |
| `--vocab-size <V>` | 1000 | Tamanho alvo do vocabulário |
| `--vocab-factor <F>` | 10.0 | Fator do vocab inicial |
| `--em-iters <E>` | 5 | Iterações EM entre podas |
| `--prune-ratio <P>` | 0.2 | Fração do vocab a remover por rodada |
| `--runs <R>` | 1 | Repetições |
| `--dump-vocab <PATH>` | — | Salva vocab com log_probs e sai |
| `--tokenize <SENTENÇA>` | — | Tokeniza e imprime |

## Estrutura de arquivos

```
unigram/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs             — re-exports públicos
    ├── pretokenize.rs     — normalize_sentence, build_initial_vocab
    ├── lattice.rs         — Viterbi, LatticeNode, viterbi_token_counts
    ├── em.rs              — UnigramModel, em_step, prune_vocab
    ├── trainer.rs         — train_unigram, tokenize_sentence
    └── bin/
        └── unigram_train.rs — CLI
```

## Testes

```bash
cargo test -p unigram
```

## Qualidade e corpus size

Com `vocab=500` e apenas **5k linhas**, Unigram alcança 8,05 tokens/palavra — acima do BPE/SP-BPE mas abaixo do WordPiece. A qualidade melhora significativamente com mais dados:

| Corpus | Tokens/palavra (est.) |
|---|:---:|
| 5k linhas | ~8,0 |
| 50k linhas | ~5–6 |
| corpus completo (276k) + mais EM iters | ~4–5 |

O `▁` aparece como token separado na maioria das segmentações, o que consome 1 posição extra por palavra em relação ao SP-BPE (onde `▁` é prefixo do primeiro token).

**Vantagem única:** o Unigram é o único entre os quatro que produz **probabilidades de segmentação**. Dado "aprendizado", é possível calcular não só a segmentação mais provável mas também segmentações alternativas com suas probabilidades — útil para regularização em modelos de linguagem (subword regularization).

Ver [RESULTS.md](../RESULTS.md) para comparação completa.

## Nota sobre performance

O Unigram é significativamente mais lento que BPE/WordPiece porque o E-step roda Viterbi em todas as sentenças a cada iteração. O Viterbi tem complexidade O(L²) por sentença e L cresce com os merges do BPE — aqui o vocabulário é grande desde o início, tornando o lattice mais denso.

Para experimentos rápidos, use `--max-lines 5000 --vocab-size 300 --em-iters 2`.
