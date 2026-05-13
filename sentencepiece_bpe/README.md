# SentencePiece-BPE

Implementação da variante BPE do SentencePiece (Kudo & Richardson, 2018) em Rust, com `rustc-hash` e `rayon`.

## Fundamento matemático

O SentencePiece usa o mesmo critério do BPE clássico — `argmax freq(a, b)` — mas muda radicalmente o **pré-processamento**:

- Não assume espaços como fronteiras de palavras.
- Substitui espaços pelo caractere especial **▁** (U+2581, "LOWER ONE EIGHTH BLOCK").
- Prefixa cada sentença com ▁, tratando o texto como uma sequência plana de caracteres.

```
"hello world"  →  "▁hello▁world"
"tokenização"  →  "▁tokenização"
```

Isso permite modelar diretamente a relação entre palavras, o que é robusto para idiomas sem espaços explícitos (chinês, japonês) e preserva o contexto de início de palavra em todos os idiomas.

**Diferença em relação ao BPE clássico**: o input não é uma lista de palavras isoladas — é uma lista de frases (ou palavras tratadas como frases). Os merges ocorrem sobre o fluxo de caracteres da frase inteira, incluindo ▁ como token.

**Filtro por `character_coverage`**: caracteres que aparecem abaixo de um limiar de frequência são mapeados para `<unk>`, reduzindo o vocabulário inicial sem perder cobertura significativa.

## API pública

| Função / Tipo | Descrição |
|---|---|
| `train_sp_bpe(sentences, cfg) -> SentencePieceModel` | Treina o modelo |
| `tokenize_sentence(sentence, model) -> Vec<String>` | Tokeniza uma sentença |
| `normalize_sentence(s) -> String` | Substitui espaços por ▁ |
| `token_frequencies(sentences, model) -> FxHashMap<String, u64>` | Contagem de tokens |
| `TrainConfig { vocab_size, character_coverage, min_pair_count }` | Configuração |
| `SentencePieceModel { vocab, vocab_set, merge_rules }` | Modelo treinado |

## Exemplo em Rust

```rust
use sentencepiece_bpe::{train_sp_bpe, tokenize_sentence, TrainConfig};

let sentences: Vec<String> = /* lista de frases */;
let cfg = TrainConfig { vocab_size: 1000, character_coverage: 0.9995, min_pair_count: 2 };
let model = train_sp_bpe(&sentences, &cfg);
let tokens = tokenize_sentence("olá mundo", &model);
// Exemplo: ["▁olá", "▁mundo"]
```

## Benchmark e CLI

```bash
# Benchmark com o corpus padrão
cargo run -p sentencepiece_bpe --release

# Personalizar cobertura e vocabulário
cargo run -p sentencepiece_bpe --release -- --vocab-size 500 --coverage 0.999

# Tokenizar uma sentença (espaços são preservados como ▁)
cargo run -p sentencepiece_bpe --release -- --vocab-size 1000 --tokenize "aprendizado de máquina"

# Salvar vocabulário
cargo run -p sentencepiece_bpe --release -- --vocab-size 500 --dump-vocab /tmp/sp_vocab.txt
```

### Flags

| Flag | Default | Descrição |
|---|---|---|
| `--file <CAMINHO>` | `/usr/share/dict/brazilian` | Arquivo (uma sentença por linha) |
| `--max-lines <N>` | (sem limite) | Limita as primeiras N linhas |
| `--vocab-size <V>` | 1000 | Tamanho alvo do vocabulário |
| `--coverage <F>` | 0.9995 | Fração de chars a cobrir (0–1) |
| `--min-pair <K>` | 2 | Contagem mínima para mesclar |
| `--runs <R>` | 3 | Repetições |
| `--dump-vocab <PATH>` | — | Salva vocabulário e sai |
| `--tokenize <SENTENÇA>` | — | Tokeniza e imprime |

## Estrutura de arquivos

```
sentencepiece_bpe/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs             — re-exports públicos
    ├── pretokenize.rs     — normalize_sentence (espaços → ▁)
    ├── pair_stats.rs      — PairStats, pack_pair, unpack_pair
    ├── trainer.rs         — train_sp_bpe, tokenize_sentence
    └── bin/
        └── sp_train.rs    — CLI
```

## Qualidade e compressão

Com `vocab=500`, SP-BPE alcança **4,90 tokens/palavra** — praticamente igual ao BPE clássico (4,70) e muito melhor que WordPiece (10,40) na mesma configuração.

A vantagem do SP-BPE sobre o BPE clássico não está na compressão bruta, mas no **significado dos tokens**:

| Palavra | BPE | SP-BPE |
|---|---|---|
| computador | `["comp", "u", "t", "ador"]` | `["▁comp", "ut", "ador"]` |
| programação | `["pro", "gra", "m", "ação"]` | `["▁pro", "g", "ram", "ação"]` |
| representação | `["re", "pres", "ent", "ação"]` | `["▁rep", "res", "ent", "ação"]` |

O prefixo `▁` marca início de palavra: `"▁comp"` é sempre o começo de uma palavra, não um fragmento interior. Isso é informação estrutural que o BPE clássico não possui.

**Limitação:** o filtro `character_coverage` elimina caracteres raros (ex: `"k"` em português), gerando `<unk>`. Para corpora multilíngues, use `--coverage 1.0` e aceite o vocab maior.

Ver [RESULTS.md](../RESULTS.md) para comparação completa.

## Testes

```bash
cargo test -p sentencepiece_bpe
```
