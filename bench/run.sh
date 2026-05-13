#!/usr/bin/env bash
# Benchmark comparativo dos 4 tokenizadores (BPE, WordPiece, SP-BPE, Unigram).
# Pré-requisitos: hyperfine, taskset.
# Opcional: sudo cpupower (pinning de frequência de CPU).
#
# Uso:
#   bench/run.sh                        # defaults
#   VOCAB=1000 CPU=3 bench/run.sh       # vocabulário maior, outro core
#   UNIGRAM=1 bench/run.sh              # inclui Unigram (mais lento)
#
# Variáveis de ambiente:
#   VOCAB           Tamanho alvo do vocabulário para WP, SP-BPE e Unigram (default: 500)
#   BPE_MERGES      Número de merges do BPE (default: 450 ≈ vocab 500)
#   FILE            Corpus (default: /usr/share/dict/brazilian)
#   CPU             Core para taskset (default: 2)
#   WARMUP          Aquecimentos do hyperfine (default: 2)
#   RUNS            Runs do hyperfine (default: 5)
#   UNIGRAM         Se "1", inclui Unigram no benchmark (default: 0)
#   UNIGRAM_LINES   Linhas do corpus usadas pelo Unigram (default: 5000)
#   UNIGRAM_EM      Iterações EM do Unigram (default: 3)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$ROOT/bench/results"

# ── Parâmetros ──────────────────────────────────────────────────────────────
VOCAB="${VOCAB:-500}"
BPE_MERGES="${BPE_MERGES:-450}"
FILE="${FILE:-/usr/share/dict/brazilian}"
CPU="${CPU:-2}"
WARMUP="${WARMUP:-2}"
RUNS="${RUNS:-5}"
UNIGRAM="${UNIGRAM:-0}"
UNIGRAM_LINES="${UNIGRAM_LINES:-5000}"
UNIGRAM_EM="${UNIGRAM_EM:-3}"

# ── Pré-requisitos ───────────────────────────────────────────────────────────
if ! command -v hyperfine >/dev/null 2>&1; then
  echo "erro: hyperfine não encontrado." >&2
  echo "  Instale com: cargo install hyperfine" >&2
  echo "  Ou:          apt install hyperfine" >&2
  exit 1
fi

if ! command -v taskset >/dev/null 2>&1; then
  echo "aviso: taskset não encontrado — sem CPU pinning (instale util-linux)" >&2
  TASKSET=""
else
  TASKSET="taskset -c $CPU"
fi

if [ ! -f "$FILE" ]; then
  echo "erro: corpus não encontrado: $FILE" >&2
  echo "  No Ubuntu/Debian: sudo apt install wbrazilian" >&2
  exit 1
fi

# ── Build ────────────────────────────────────────────────────────────────────
echo "== build =="
(cd "$ROOT" && cargo build --release --quiet)
echo "  ok"

# ── Governor de CPU (não-fatal) ──────────────────────────────────────────────
if command -v cpupower >/dev/null 2>&1; then
  sudo -n cpupower frequency-set -g performance >/dev/null 2>&1 \
    && echo "governor: performance (cpu $CPU)" \
    || echo "governor: sem permissão sudo — continuando sem ajuste"
fi

# ── Binários (caminhos absolutos) ────────────────────────────────────────────
BPE_BIN="$ROOT/target/release/bpe-train"
WP_BIN="$ROOT/target/release/wp-train"
SP_BIN="$ROOT/target/release/sp-train"
UNI_BIN="$ROOT/target/release/unigram-train"

# ── Configuração do benchmark ────────────────────────────────────────────────
echo ""
echo "== Tokenizers-RS benchmark =="
echo "  corpus:       $FILE"
echo "  vocab alvo:   $VOCAB (BPE: $BPE_MERGES merges)"
echo "  CPU:          $CPU  |  warmup: $WARMUP  |  runs: $RUNS"
echo "  Unigram:      $([ "$UNIGRAM" = "1" ] && echo "sim (${UNIGRAM_LINES} linhas, em=${UNIGRAM_EM})" || echo "não (use UNIGRAM=1 para incluir)")"
echo ""

mkdir -p "$RESULTS"

# ── Comandos hiperfine ────────────────────────────────────────────────────────
# Nota: cada binário recebe --runs 1 para que o hyperfine controle as iterações.
# O hyperfine mede wall time completo (startup + leitura de arquivo + treino).

CMD_BPE="$TASKSET $BPE_BIN --file $FILE --merges $BPE_MERGES --runs 1"
CMD_WP="$TASKSET $WP_BIN  --file $FILE --vocab-size $VOCAB --runs 1"
CMD_SP="$TASKSET $SP_BIN  --file $FILE --vocab-size $VOCAB --runs 1"
CMD_UNI="$TASKSET $UNI_BIN --file $FILE --max-lines $UNIGRAM_LINES --vocab-size $VOCAB --em-iters $UNIGRAM_EM --runs 1"

if [ "$UNIGRAM" = "1" ]; then
  hyperfine \
    --warmup  "$WARMUP" \
    --runs    "$RUNS" \
    --export-markdown "$RESULTS/bench.md" \
    --export-json     "$RESULTS/bench.json" \
    -n "BPE (merges=$BPE_MERGES)"               "$CMD_BPE" \
    -n "WordPiece (vocab=$VOCAB)"                "$CMD_WP"  \
    -n "SP-BPE (vocab=$VOCAB)"                   "$CMD_SP"  \
    -n "Unigram (vocab=$VOCAB, ${UNIGRAM_LINES} linhas)" "$CMD_UNI"
else
  hyperfine \
    --warmup  "$WARMUP" \
    --runs    "$RUNS" \
    --export-markdown "$RESULTS/bench.md" \
    --export-json     "$RESULTS/bench.json" \
    -n "BPE (merges=$BPE_MERGES)"  "$CMD_BPE" \
    -n "WordPiece (vocab=$VOCAB)"   "$CMD_WP"  \
    -n "SP-BPE (vocab=$VOCAB)"      "$CMD_SP"
fi

# ── Resultado ────────────────────────────────────────────────────────────────
echo ""
echo "Resultados salvos em:"
echo "  $RESULTS/bench.md"
echo "  $RESULTS/bench.json"
echo ""
cat "$RESULTS/bench.md"
