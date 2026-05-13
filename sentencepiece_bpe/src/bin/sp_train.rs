use sentencepiece_bpe::{tokenize_sentence, train_sp_bpe, TrainConfig};
use std::time::Instant;

const DEFAULT_WORDLIST: &str = "/usr/share/dict/brazilian";

fn usage() -> ! {
    eprintln!(
        "\
Uso: sp-train [opções]

Por omissão: --file {DEFAULT_WORDLIST}

Opções:
  --file <CAMINHO>       Arquivo de texto (UTF-8, uma sentença/palavra por linha)
  --max-lines <N>        Só as primeiras N linhas úteis
  --vocab-size <V>       Tamanho alvo do vocabulário (default: 1000)
  --coverage <F>         character_coverage [0..1] (default: 0.9995)
  --min-pair <K>         min_pair_count (default: 2)
  --runs <R>             Repetições do treino (default: 3)
  --dump-vocab <PATH>    Salva vocabulário (um token por linha) e sai
  --tokenize <SENTENÇA>  Tokeniza uma sentença e imprime os tokens
  --help, -h

Diferença para BPE: opera em frases inteiras, substitui espaços por ▁.

Exemplo:
  cargo run --release -- --vocab-size 500
  cargo run --release -- --tokenize \"tokenização é importante\"
"
    );
    std::process::exit(0);
}

fn arg_usize(args: &[String], key: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn arg_u64(args: &[String], key: &str, default: u64) -> u64 {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn arg_f64(args: &[String], key: &str, default: f64) -> f64 {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn arg_opt_usize(args: &[String], key: &str) -> Option<usize> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

fn arg_str(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn load_sentences(path: &str, max_lines: Option<usize>) -> std::io::Result<Vec<String>> {
    let text = std::fs::read_to_string(path)?;
    let mut out: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect();
    if let Some(m) = max_lines {
        out.truncate(m);
    }
    Ok(out)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        usage();
    }

    let path = arg_str(&args, "--file").unwrap_or_else(|| DEFAULT_WORDLIST.to_string());
    let max_lines = arg_opt_usize(&args, "--max-lines");
    let vocab_size = arg_usize(&args, "--vocab-size", 1000);
    let character_coverage = arg_f64(&args, "--coverage", 0.9995);
    let min_pair_count = arg_u64(&args, "--min-pair", 2);
    let runs = arg_usize(&args, "--runs", 3).max(1);
    let dump_vocab = arg_str(&args, "--dump-vocab");
    let tokenize_arg = arg_str(&args, "--tokenize");

    let sentences = match load_sentences(&path, max_lines) {
        Ok(s) if !s.is_empty() => s,
        Ok(_) => {
            eprintln!("erro: ficheiro sem linhas válidas: {path}");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("erro ao ler {path}: {e}");
            std::process::exit(1);
        }
    };

    let cfg = TrainConfig { vocab_size, character_coverage, min_pair_count };

    if let Some(path_out) = dump_vocab {
        let model = train_sp_bpe(&sentences, &cfg);
        let out = model.vocab.join("\n") + "\n";
        if let Err(e) = std::fs::write(&path_out, out) {
            eprintln!("erro ao escrever {path_out}: {e}");
            std::process::exit(1);
        }
        return;
    }

    if let Some(sentence) = tokenize_arg {
        let model = train_sp_bpe(&sentences, &cfg);
        let tokens = tokenize_sentence(&sentence, &model);
        println!("SP-BPE(vocab={}): {:?}", model.vocab.len(), tokens);
        return;
    }

    let num_lines = sentences.len();
    let total_chars: usize = sentences.iter().map(|s| s.chars().count()).sum();

    println!("SentencePiece-BPE — benchmark");
    println!("------------------------------");
    println!("ficheiro:        {path}");
    println!("linhas:          {num_lines}");
    println!("chars total:     {total_chars}");
    println!("vocab_size alvo: {vocab_size}");
    println!("coverage:        {character_coverage}");
    println!("min_pair_count:  {min_pair_count}");
    println!("execuções:       {runs}");
    println!();

    let mut durations_ms: Vec<f64> = Vec::with_capacity(runs);
    let mut last_vocab_size = 0usize;

    for r in 0..runs {
        let t0 = Instant::now();
        let model = train_sp_bpe(&sentences, &cfg);
        let elapsed = t0.elapsed();
        last_vocab_size = model.vocab.len();
        durations_ms.push(elapsed.as_secs_f64() * 1000.0);
        println!(
            "  run {}: {:>10.2} ms   vocab: {}",
            r + 1,
            durations_ms[r],
            model.vocab.len()
        );
    }

    let min_ms = durations_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let max_ms = durations_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let avg_ms = durations_ms.iter().sum::<f64>() / runs as f64;

    println!();
    println!("Resumo:");
    println!("  tempo mínimo:  {min_ms:.2} ms");
    println!("  tempo médio:   {avg_ms:.2} ms");
    println!("  tempo máximo:  {max_ms:.2} ms");
    println!(
        "  throughput:    {:.0} linhas/s (médio)",
        num_lines as f64 / (avg_ms / 1000.0)
    );
    println!("  vocab final:   {last_vocab_size}");
}
