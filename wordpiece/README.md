# WordPiece

Implementação do WordPiece (Schuster & Nakamura, 2012; usado no BERT) em Rust, com `rustc-hash` e `rayon`.

## Fundamento matemático

Em vez de selecionar o par mais frequente, o WordPiece maximiza o ganho de log-verossimilhança do corpus. O score de cada par é:

```
score(a, b) = freq(ab) / (freq(a) × freq(b))
```

Isso equivale à razão de probabilidade `P(ab) / (P(a)·P(b))`, semelhante ao PMI mas sem os denominadores globais (que são constantes no argmax). O algoritmo favorece pares que ocorrem juntos **mais do que o esperado pelo acaso**.

**Diferença fundamental em relação ao BPE**: o BPE seleciona por `freq(ab)` diretamente; o WordPiece divide pela frequência marginal de cada parte, tornando-o mais seletivo com tokens comuns.

**Critério de parada**: `vocab_size` (não `num_merges`). O treinamento para quando o vocabulário atinge o tamanho alvo.

**Prefixo `##`**: tokens que aparecem no meio de palavras recebem o prefixo `##`:
```
"playing" → ["p", "##l", "##a", "##y", "##i", "##n", "##g"]
```
Isso preserva a informação de posição dentro da palavra. Após os merges:
- `"p" + "##l"` → `"pl"` (sem `##`)
- `"##a" + "##y"` → `"##ay"` (mantém `##`)

**Inferência** (tokenize_word): greedy longest-match da esquerda para direita. Tenta o maior substring que esteja no vocabulário; avança e repete. Retorna `["[UNK]"]` se nenhuma decomposição for possível.

## API pública

| Função / Tipo | Descrição |
|---|---|
| `train_wordpiece(words, cfg) -> WordPieceModel` | Treina o modelo até `vocab_size` |
| `tokenize_word(word, model) -> Vec<String>` | Inferência greedy longest-match |
| `wordpiece_pretokenize(word) -> Vec<String>` | Divide palavra com prefixo `##` |
| `token_frequencies(words, model) -> FxHashMap<String, u64>` | Contagem de tokens |
| `wordpiece_score(c_ab, c_a, c_b) -> f64` | Score `freq(ab)/(freq(a)×freq(b))` |
| `TrainConfig { vocab_size, min_pair_count }` | Configuração |
| `WordPieceModel { vocab, vocab_set, merge_rules }` | Modelo treinado |

## Exemplo em Rust

```rust
use wordpiece::{train_wordpiece, tokenize_word, TrainConfig};

let corpus: Vec<String> = /* lista de palavras */;
let cfg = TrainConfig { vocab_size: 1000, min_pair_count: 2 };
let model = train_wordpiece(&corpus, &cfg);
let tokens = tokenize_word("tokenização", &model);
// Exemplo: ["t", "##o", "##k", "##en", "##iz", "##a", "##ção"]
```

## Benchmark e CLI

```bash
# Benchmark (vocab_size=1000)
cargo run -p wordpiece --release

# Personalizar tamanho do vocabulário
cargo run -p wordpiece --release -- --vocab-size 500 --runs 3

# Tokenizar uma palavra
cargo run -p wordpiece --release -- --vocab-size 1000 --tokenize "aprendizado"

# Salvar vocabulário em arquivo
cargo run -p wordpiece --release -- --vocab-size 500 --dump-vocab /tmp/wp_vocab.txt

# Ver tokens mais frequentes
cargo run -p wordpiece --release -- --vocab-size 500 --top-tokens 20
```

### Flags

| Flag | Default | Descrição |
|---|---|---|
| `--file <CAMINHO>` | `/usr/share/dict/brazilian` | Lista de palavras |
| `--max-words <N>` | (sem limite) | Limita as primeiras N palavras |
| `--vocab-size <V>` | 1000 | Tamanho alvo do vocabulário |
| `--min-pair <K>` | 2 | Contagem mínima para mesclar |
| `--runs <R>` | 3 | Repetições |
| `--dump-vocab <PATH>` | — | Salva vocabulário e sai |
| `--tokenize <PALAVRA>` | — | Tokeniza e imprime |
| `--top-tokens <N>` | — | Imprime top-N tokens por frequência |

## Estrutura de arquivos

```
wordpiece/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs           — re-exports públicos
    ├── pair_stats.rs    — PairStats, wordpiece_score
    ├── trainer.rs       — train_wordpiece, tokenize_word, Vocab interno
    └── bin/
        └── wp_train.rs  — CLI
```

## Qualidade e compressão — atenção ao tamanho do vocabulário

> **WordPiece precisa de vocab grande para ser eficiente.**

Com `vocab=500` no corpus brasileiro, WordPiece alcança **10,40 tokens/palavra** — mais que o dobro do BPE (4,70). A causa é o score seletivo:

```
score(a, b) = freq(ab) / (freq(a) × freq(b))
```

Letras como `"a"`, `"e"`, `"o"` têm `freq` marginal altíssima em português. Qualquer merge envolvendo essas letras tem score baixo e é descartado. Com vocab=500, pouquíssimos merges passam o critério.

| vocab | Tokens/palavra (est.) | Comportamento |
|---|:---:|---|
| 500 | ~10,4 | Quase nível de caractere |
| 5.000 | ~4–6 | Começa a mostrar tokens morfológicos |
| 30.000 | ~2–3 | Nível do BERT — comportamento real |

**Fragmentação observada com vocab=500:**

| Palavra | Tokens |
|---|---|
| desenvolvimento | `["d","##e","##s","##e","##n","##v","##o","##l","##v","##i","##m","##e","##n","##t","##o"]` — 15 tokens |
| transformação | `["t","##r","##a","##n","##s","##f","##o","##r","##m","##a","##ção"]` — 11 tokens |

**Quando usar WordPiece:** somente quando o vocab alvo for ≥5.000. Abaixo disso, BPE ou SP-BPE comprimem muito melhor no mesmo tempo de treino.

Ver [RESULTS.md](../RESULTS.md) para comparação completa.

## Testes

```bash
cargo test -p wordpiece
```
