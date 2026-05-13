# BPE — Byte Pair Encoding Clássico

Implementação do BPE clássico (Sennrich et al., 2016) em Rust, usando as crates `rustc-hash` e `rayon`.

## Fundamento matemático

A cada iteração, calcula-se a frequência de todos os pares adjacentes de tokens no corpus:

```
f(a, b) = Σᵢ 1[(a, b) ∈ xᵢ]
```

O par com maior frequência `f(a, b)` é mesclado, e o novo token `ab` passa a fazer parte do vocabulário. O algoritmo é **guloso e determinístico**: maximiza a compressão local a cada passo.

**Diferença em relação ao PMI-BPE** (`pmi-bpe-rust`): aqui o critério é a frequência absoluta `f(a,b)`, não o score PMI `c_xy·T² / (B·c_x·c_y)`. O BPE clássico não considera a probabilidade marginal de cada token.

## API pública

| Função / Tipo | Descrição |
|---|---|
| `train_bpe(words, cfg) -> Vec<MergeRule>` | Treina o BPE e retorna as regras de merge |
| `apply_merges_to_word(word, rules) -> Vec<String>` | Aplica regras a uma palavra |
| `token_frequencies(words, rules) -> FxHashMap<String, u64>` | Conta tokens após os merges |
| `TrainConfig { num_merges, min_pair_count }` | Configuração do treino |
| `MergeRule = (String, String)` | Par (left, right) mesclado |
| `PairStats` | Contagens de pares usadas internamente |

## Exemplo em Rust

```rust
use bpe::{train_bpe, apply_merges_to_word, TrainConfig};

let corpus = vec!["baixo".into(), "baixa".into(), "baixos".into()];
let cfg = TrainConfig { num_merges: 5, min_pair_count: 2 };
let rules = train_bpe(&corpus, &cfg);
let tokens = apply_merges_to_word("baixo", &rules);
// Exemplo: ["baix", "o"]
```

## Otimizações

| Técnica | Detalhe |
|---|---|
| `FxHashMap<u64, u64>` | Hash rápido para pares codificados como u64 |
| `pack_pair(a, b) → u64` | Par de IDs u32 empacotado em u64 |
| `rayon par_chunks(2048)` | Contagem paralela de pares |
| Two-pointer in-place | `apply_merge` em O(n) por palavra |
| `lto = "fat"`, `codegen-units = 1` | Definido no workspace root |

## Benchmark e CLI

```bash
# Benchmark com o corpus padrão (/usr/share/dict/brazilian)
cargo run -p bpe --release

# Opções avançadas
cargo run -p bpe --release -- --merges 50 --runs 3

# Tokenizar uma palavra
cargo run -p bpe --release -- --merges 25 --tokenize "aprendizado"

# Salvar regras em TSV
cargo run -p bpe --release -- --merges 25 --dump-rules /tmp/bpe_rules.tsv
```

### Flags

| Flag | Default | Descrição |
|---|---|---|
| `--file <CAMINHO>` | `/usr/share/dict/brazilian` | Lista de palavras |
| `--max-words <N>` | (sem limite) | Limita as primeiras N palavras |
| `--merges <M>` | 25 | Número de passos de merge |
| `--min-pair <K>` | 2 | Contagem mínima para mesclar |
| `--runs <R>` | 3 | Repetições para medir tempo |
| `--dump-rules <PATH>` | — | Salva regras TSV e sai |
| `--tokenize <PALAVRA>` | — | Tokeniza e imprime |

## Estrutura de arquivos

```
bpe/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs           — re-exports públicos
    ├── pair_stats.rs    — PairStats, pack_pair, unpack_pair
    ├── trainer.rs       — train_bpe, apply_merges_to_word, Vocab interno
    └── bin/
        └── bpe_train.rs — CLI
```

## Qualidade e compressão

Com `merges=450` no corpus `/usr/share/dict/brazilian`, BPE alcança **4,70 tokens/palavra** em média — a melhor compressão entre os quatro algoritmos com o mesmo orçamento de tempo (~5,8 s).

| Palavra | Tokens BPE (450 merges) |
|---|---|
| transformação | `["trans", "form", "ação"]` — 3 tokens |
| computador | `["comp", "u", "t", "ador"]` — 4 tokens |
| desenvolvimento | `["desen", "vol", "vi", "ment", "o"]` — 5 tokens |

**Por que BPE comprime bem em vocab pequeno?** O critério `argmax freq(a,b)` não impõe penalidade sobre tokens frequentes — qualquer par com alta coocorrência é mesclado. Isso maximiza compressão local a cada passo, mesmo com poucos merges.

**Limitação:** tokens morfologicamente arbitrários. `"form"` em "transformação" coincide com um morfema real, mas `"desen"` ou `"ment"` são artefatos estatísticos do corpus, não morfemas linguísticos.

Ver [RESULTS.md](../RESULTS.md) para comparação completa com os outros algoritmos.

## Testes

```bash
cargo test -p bpe
```
