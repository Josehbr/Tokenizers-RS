# Tokenizers-RS

Quatro algoritmos clássicos de tokenização por subpalavras implementados em Rust, usando as crates `rustc-hash` e `rayon`.

Este projeto complementa o [`Tokenizacao-PMI-BPE`](https://github.com/Josehbr/Tokenizacao-PMI-BPE), que compara o mesmo algoritmo em Python, Rust e C++. Aqui o foco é comparar **algoritmos diferentes** numa mesma linguagem.

## Algoritmos

| Crate | Critério de merge/seleção | Input | Principal característica |
|---|---|---|---|
| [`bpe`](bpe/) | `argmax freq(a, b)` | lista de palavras | BPE clássico, guloso por frequência |
| [`wordpiece`](wordpiece/) | `argmax freq(ab) / (freq(a)·freq(b))` | lista de palavras | Prefixo `##`, parada por vocab_size |
| [`sentencepiece_bpe`](sentencepiece_bpe/) | `argmax freq(a, b)` | lista de frases | Sem fronteira de palavra; usa `▁` (U+2581) |
| [`unigram`](unigram/) | remoção do menor `ΔL` | lista de frases | Começa grande, poda iterativamente; Viterbi |

## Estrutura do workspace

```
Tokenizers-RS/
├── Cargo.toml                 ← workspace root (profile.release compartilhado)
├── README.md                  ← este arquivo
├── RESULTS.md                 ← benchmarks e comparação qualitativa
├── contexto.md                ← fundamentação matemática dos algoritmos
├── bpe/                       ← BPE clássico (Sennrich et al., 2016)
├── wordpiece/                 ← WordPiece (Schuster & Nakamura, 2012)
├── sentencepiece_bpe/         ← SentencePiece-BPE (Kudo & Richardson, 2018)
└── unigram/                   ← Unigram LM (Kudo, 2018)
```

## Crates compartilhadas

- **`rustc-hash = "2"`**: fornece `FxHashMap` — hash map otimizado para chaves inteiras pequenas, mais rápido que `SipHash` da stdlib no hot path de contagem de pares
- **`rayon = "1"`**: paralelismo de dados com `par_chunks` e `par_iter` para contagem de pares e E-step do Unigram

## Quick start

```bash
# Compilar tudo em modo release
cargo build --release --workspace

# BPE: 25 merges, corpus de palavras brasileiras
cargo run -p bpe --release

# WordPiece: vocabulário de 1000 tokens
cargo run -p wordpiece --release -- --vocab-size 1000

# SentencePiece-BPE: vocabulário de 1000 tokens
cargo run -p sentencepiece_bpe --release -- --vocab-size 1000

# Unigram: vocabulário alvo de 500 tokens (mais lento)
cargo run -p unigram --release -- --max-lines 10000 --vocab-size 500

# Tokenizar a mesma palavra com cada algoritmo
cargo run -p bpe --release -- --merges 50 --tokenize "aprendizado"
cargo run -p wordpiece --release -- --vocab-size 1000 --tokenize "aprendizado"
cargo run -p sentencepiece_bpe --release -- --vocab-size 1000 --tokenize "aprendizado"
cargo run -p unigram --release -- --max-lines 10000 --vocab-size 500 --tokenize "aprendizado"
```

## Testes

```bash
cargo test --workspace
```

## Resultados principais

Benchmark controlado: `hyperfine` + `taskset -c 2`, corpus `/usr/share/dict/brazilian`, `vocab=500`.

**Tempo de treino** (BPE merges=450 ≈ vocab 500, corpus completo 276k palavras):

| Algoritmo | Tempo médio | Relativo |
|---|---:|:---:|
| BPE (merges=450) | 5,78 s | 1,00× |
| SP-BPE (vocab=500) | 5,87 s | 1,02× |
| WordPiece (vocab=500) | 5,97 s | 1,03× |
| Unigram (vocab=500, 5k linhas) | 0,63 s | — ¹ |

¹ Unigram usa apenas 5k linhas — corpus diferente, não comparável diretamente.

**Qualidade** (tokens/palavra, menor = melhor compressão):

| Algoritmo | Tokens/palavra | Insight |
|---|:---:|---|
| BPE | **4,70** | Melhor compressão em vocab pequeno |
| SP-BPE | **4,90** | Igual ao BPE + marca fronteiras de palavra (▁) |
| Unigram | 8,05 | Melhora muito com corpus maior |
| WordPiece | 10,40 | **Ineficiente em vocab ≤ 1000** — precisa de ≥ 5k tokens |

> Com o mesmo tempo de treino (~5,8s), os algoritmos entregam **qualidades muito diferentes**.
> Ver [RESULTS.md](RESULTS.md) para a análise completa.

## Testes

```bash
cargo test --workspace
```

## Referências

- Sennrich et al. (2016) — [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (BPE)
- Schuster & Nakamura (2012) — Japanese and Korean Voice Search (WordPiece)
- Kudo & Richardson (2018) — [SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226)
- Kudo (2018) — [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/abs/1804.10959) (Unigram)
