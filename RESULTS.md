# RESULTS.md — Comparação dos Quatro Tokenizadores

## 1. Configuração experimental

| Item | Valor |
|---|---|
| Plataforma | WSL2 2, Linux 6.6.87 |
| CPU | AMD Ryzen 7 5700X 8-Core (3,40 GHz) |
| RAM disponível | 10 GB (WSL2) |
| Corpus padrão | `/usr/share/dict/brazilian` (~276 k palavras) |
| Build | `cargo build --release` (lto=fat, codegen-units=1, panic=abort) |
| Medição | `std::time::Instant` — mínimo de 3 runs (Unigram: 1 run) |

### Limitações desta medição

Estes números são **indicativos, não controlados**. As diferenças em relação a um benchmark rigoroso (como o usado no `PMI-BPE-Comparison`) são:

| Técnica | PMI-BPE-Comparison | Este projeto |
|---|:---:|:---:|
| CPU pinning (`taskset -c N`) | sim | não |
| Governor de frequência (`performance`) | sim | não |
| Ferramenta de benchmark (`hyperfine`) | sim | não (`Instant::now()`) |
| Warmup dedicado antes de medir | sim (3 warmups) | não (runs sequenciais) |
| Corpus e config idênticos entre algoritmos | sim (mesmo algoritmo) | **não** (ver abaixo) |

**Por que os corpus e configs diferem:**

- BPE e WordPiece recebem **listas de palavras** (uma por linha) — o corpus `brazilian` encaixa naturalmente.
- SP-BPE e Unigram recebem **frases** — o mesmo arquivo é tratado como uma frase por linha (cada palavra vira uma frase de uma palavra). Isso é funcionalmente equivalente para o corpus usado.
- **BPE usa `num_merges=25`** enquanto WordPiece/SP-BPE usam `vocab_size=1000`. São critérios de parada diferentes: 25 merges é muito menos trabalho que 975 (necessários para vocab=1000 partindo de ~50 chars únicos). A comparação de tempo entre BPE e WordPiece **não é justa** sem normalizar pelo mesmo critério.
- **Unigram usa apenas 10 k linhas** do corpus (não o corpus completo de 276 k), porque o loop EM+Viterbi no corpus completo levaria vários minutos. Os números de throughput não são comparáveis com os outros algoritmos.

---

## 2. Tempo de treinamento (benchmark controlado)

Resultados gerados por `bench/run.sh` com `hyperfine` + `taskset -c 2` + `warmup=2` + `runs=5`.
Config padronizada: `vocab=500`, `BPE_MERGES=450` (≈ vocab 500), corpus `/usr/share/dict/brazilian`.

### BPE / WordPiece / SP-BPE — corpus completo (276 k palavras)

| Command | Média | Min | Max | Relativo |
|:---|---:|---:|---:|---:|
| `BPE (merges=450)` | 5,778 s ± 0,013 s | 5,756 s | 5,789 s | **1,00** |
| `SP-BPE (vocab=500)` | 5,869 s ± 0,085 s | 5,797 s | 6,010 s | 1,02 ± 0,01 |
| `WordPiece (vocab=500)` | 5,974 s ± 0,096 s | 5,907 s | 6,139 s | 1,03 ± 0,02 |

**Resultado chave:** com o mesmo número de iterações (~450), os três algoritmos levam praticamente o mesmo tempo. A diferença de menos de 3% está dentro da variação do WSL2.

### Unigram — corpus reduzido (5 k palavras, em-iters=3)

| Command | Média | Min | Max |
|:---|---:|---:|---:|
| `Unigram (vocab=500, 5 k linhas)` | 633 ms ± 16 ms | 620 ms | 650 ms |

> O Unigram usa apenas 5 k linhas. Não é comparável diretamente com os outros.
> No corpus completo (276 k), o tempo sobe para ~30–60 min dependendo do vocab_size e em-iters.

### Como reproduzir

```bash
# Benchmark padrão (BPE + WordPiece + SP-BPE, vocab=500)
bench/run.sh

# Incluindo Unigram (mais lento, corpus reduzido)
UNIGRAM=1 bench/run.sh

# Personalizar
VOCAB=1000 BPE_MERGES=950 CPU=3 RUNS=10 bench/run.sh
```

### Por que os tempos são tão similares?

Com `vocab=500`, todos os três algoritmos baseados em merge (BPE, WordPiece, SP-BPE) executam aproximadamente **450 iterações** (500 tokens − ~50 chars iniciais únicos). O custo por iteração é O(N·L̄ + |P|) onde:
- N = número de palavras, L̄ = comprimento médio, |P| = número de pares distintos.

As diferenças de implementação (BPE só usa `pair_counts`; WordPiece também usa `token_counts`) são quase imperceptíveis quando |P| é o gargalo real. A paralelização com `rayon` nivela ainda mais o resultado.

**O Unigram é estruturalmente diferente:** em vez de N iterações de merge, executa ~7 rodadas de `EM × poda`. Cada rodada roda Viterbi em todas as frases (O(L²) por frase). Com corpus grande, o E-step domina — por isso o Unigram não escala igual aos outros.

---

## 3. Comparação de qualidade — tokens por palavra

Medição em 20 palavras de teste. Mesma config do benchmark de tempo:
`BPE merges=450`, `WordPiece/SP-BPE vocab=500`, `Unigram vocab=500 / 5k linhas`.

### Métrica principal: compressão (tokens por palavra)

Menos tokens = mais compressão = vocabulário mais eficiente.

| Algoritmo | Média tokens/palavra | Relativo ao BPE |
|---|:---:|:---:|
| **BPE** (merges=450) | **4,70** | 1,00× |
| **SP-BPE** (vocab=500) | **4,90** | 1,04× |
| **Unigram** (vocab=500, 5k linhas) | 8,05 | 1,71× |
| **WordPiece** (vocab=500) | 10,40 | 2,21× |

> Estes números revelam algo que o benchmark de tempo não mostra:
> **os algoritmos entregam qualidades muito diferentes com o mesmo tempo de treino.**

### Por que WordPiece fragmenta tanto com vocab=500?

O score `freq(ab) / (freq(a) × freq(b))` é muito seletivo: letras como "a", "e", "o" têm frequência marginal altíssima em português, então qualquer merge envolvendo essas letras tem score baixo e é descartado. Com vocab=500, pouquíssimos merges passam o critério — a maioria das palavras fica quase em nível de caractere.

**WordPiece precisa de vocabulários muito maiores para ser eficiente** — o BERT usa 30.000 tokens. Com vocab=500 ele é o pior entre os quatro.

### Por que Unigram fragmenta mais que BPE/SP-BPE?

Dois fatores:
1. **Corpus reduzido** (5k vs 276k linhas): o EM converge em menos substrings frequentes.
2. **Mecanismo diferente**: o Unigram prioriza probabilidades, não compressão. O `▁` separado consome 1 token por palavra antes mesmo do conteúdo. Com mais dados e iterações, a qualidade melhora.

### Tabela completa de segmentações

| Palavra | BPE | WP | SP-BPE | Unigram |
|---|:---:|:---:|:---:|:---:|
| aprendizado | 5 | 11 | 5 | 6 |
| tokenização | 5 | 9 | 6 | 8 |
| computador | 4 | 10 | 3 | 7 |
| linguagem | 5 | 9 | 5 | 6 |
| programação | 4 | 9 | 4 | 7 |
| processamento | 5 | 13 | 5 | 10 |
| representação | 4 | 11 | 4 | 8 |
| transformação | 3 | 11 | 4 | 10 |
| classificação | 5 | 11 | 4 | 10 |
| reconhecimento | 5 | 13 | 6 | 10 |
| desenvolvimento | 5 | 15 | 6 | 11 |
| implementação | 5 | 9 | 5 | 9 |
| generalização | 4 | 11 | 5 | 8 |
| regularização | 4 | 11 | 4 | 8 |
| otimização | 5 | 8 | 5 | 7 |
| segmentação | 4 | 9 | 4 | 8 |
| vocabulário | 6 | 11 | 6 | 6 |
| subpalavra | 5 | 8 | 5 | 8 |
| morfologia | 6 | 10 | 6 | 8 |
| frequência | 5 | 9 | 6 | 6 |
| **Média** | **4,70** | **10,40** | **4,90** | **8,05** |

### Exemplos de segmentação detalhados

**"transformação"** — melhor caso do BPE:

| Algoritmo | Tokens |
|---|---|
| BPE | `["trans", "form", "ação"]` ← 3 morfemas reais |
| SP-BPE | `["▁trans", "f", "orm", "ação"]` |
| Unigram | `["▁", "t", "r", "an", "s", "f", "or", "m", "aç", "ão"]` |
| WordPiece | `["t", "##r", "##a", "##n", "##s", "##f", "##o", "##r", "##m", "##a", "##ção"]` |

**"computador"** — melhor caso do SP-BPE:

| Algoritmo | Tokens |
|---|---|
| SP-BPE | `["▁comp", "ut", "ador"]` ← 3 tokens, prefixos reais |
| BPE | `["comp", "u", "t", "ador"]` |
| Unigram | `["▁", "co", "m", "p", "u", "t", "ador"]` |
| WordPiece | `["c", "##o", "##m", "##p", "##u", "##t", "##a", "##d", "##o", "##r"]` |

> SP-BPE captura "▁comp" como prefixo de início de palavra — algo que o BPE clássico não pode representar porque não distingue posição na frase.

---

## 4. O que os dados dizem na prática

Com **mesmo tempo de treino** (~5,8s no corpus completo):

| Algoritmo | Compressão | Morfologia | Fronteira de palavra | Vocab pequeno |
|---|:---:|:---:|:---:|:---:|
| BPE | ✓✓ | parcial | não | ✓✓ |
| SP-BPE | ✓✓ | parcial | sim (▁) | ✓✓ |
| WordPiece | ✗ (vocab=500) | selectivo | não | ✗ |
| Unigram | parcial | melhor com mais dados | sim (▁) | depende do EM |

**Conclusão:** para vocab pequeno (≤1000), BPE e SP-BPE são os mais eficientes. WordPiece só mostra sua vantagem com vocab grande (≥5000), onde o score seletivo produz tokens mais significativos do que BPE. Unigram tem potencial maior com corpus maior e mais iterações de EM.

---

## 5. Análise por algoritmo

### BPE
- Critério puramente estatístico (frequência de pares).
- Produz tokens que refletem coocorrência no corpus, não necessariamente morfologia.
- Rápido, determinístico, fácil de implementar.
- Tokens podem ser morfologicamente arbitrários: "ad" em "aprendizado" não é um morfema.

### WordPiece
- Score normalizado pelas frequências marginais reduz a influência de letras muito comuns.
- O prefixo `##` permite reconstrução exata da palavra original a partir dos tokens.
- Inferência greedy longest-match: eficiente, mas pode errar em palavras ambíguas.
- Com vocab=1000 e corpus grande, ainda fragmenta palavras longas — precisa de vocab maior.

### SentencePiece-BPE
- Preserva informação de início de palavra via `▁` sem precisar de delimitadores explícitos.
- Tokens como `"▁comp"` ou `"▁ap"` capturam prefixos de palavras.
- Robusto para múltiplos idiomas: não depende de tokenização por espaços.
- O filtro `character_coverage` elimina caracteres raros (ex: "k" em português), produzindo `<unk>`.

### Unigram
- Vocabulário probabilístico: cada token tem um `log_prob`.
- O ▁ pode aparecer como token separado (representa a fronteira de palavra).
- Produz segmentações mais interpretáveis ao longo do tempo (EM converge).
- Muito mais lento que os outros: Viterbi O(L²) × |vocab| por sentença por iteração.
- Único entre os quatro que permite calcular probabilidades de segmentações alternativas.

---

## 6. Quando usar cada um

| Situação | Recomendação |
|---|---|
| Velocidade de treinamento é crítica | BPE |
| Compatibilidade com BERT/HuggingFace | WordPiece |
| Multilíngue ou idiomas sem espaços | SentencePiece-BPE |
| Precisa de probabilidades de segmentação | Unigram |
| Corpus muito pequeno (<10 k palavras) | Unigram (EM funciona bem com pouco dado) |

---

## 7. Como reproduzir

```bash
# Compilar em release
cargo build --release --workspace

# BPE
cargo run -p bpe --release -- --merges 50 --runs 3

# WordPiece
cargo run -p wordpiece --release -- --vocab-size 1000 --runs 3

# SentencePiece-BPE
cargo run -p sentencepiece_bpe --release -- --vocab-size 1000 --runs 3

# Unigram (corpus reduzido para timing razoável)
cargo run -p unigram --release -- --max-lines 10000 --vocab-size 500 --em-iters 3 --runs 1

# Comparar tokenização de uma palavra
for algo in bpe wordpiece sentencepiece_bpe unigram; do
    cargo run -p $algo --release -- --tokenize "aprendizado" 2>/dev/null
done
```
