use std::time::Instant;
use unigram::{tokenize_sentence, train_unigram, TrainConfig};

const DEFAULT_WORDLIST: &str = "/usr/share/dict/brazilian";

fn usage() -> ! {
    eprintln!(
        "\
Uso: unigram-train [opções]

Por omissão: --file {DEFAULT_WORDLIST}

Opções:
  --file <CAMINHO>       Arquivo de texto (UTF-8, uma sentença/palavra por linha)
  --max-lines <N>        Só as primeiras N linhas úteis
  --vocab-size <V>       Tamanho alvo do vocabulário (default: 1000)
  --vocab-factor <F>     Fator do vocab inicial: target × factor (default: 10.0)
  --em-iters <E>         Iterações EM entre podas (default: 5)
  --prune-ratio <P>      Fração do vocab a remover por rodada (default: 0.2)
  --runs <R>             Repetições do treino (default: 1)
  --dump-vocab <PATH>    Salva vocabulário com log_probs e sai
  --tokenize <SENTENÇA>  Tokeniza uma sentença e imprime os tokens
  --help, -h

Nota: Unigram é o mais lento dos quatro — recomenda-se --vocab-size ≤ 2000.

Exemplo:
  cargo run --release -- --vocab-size 500 --runs 1
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
    let target_vocab_size = arg_usize(&args, "--vocab-size", 1000);
    let initial_vocab_factor = arg_f64(&args, "--vocab-factor", 10.0);
    let em_iterations = arg_usize(&args, "--em-iters", 5);
    let prune_ratio = arg_f64(&args, "--prune-ratio", 0.2);
    let runs = arg_usize(&args, "--runs", 1).max(1);
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

    let cfg = TrainConfig {
        target_vocab_size,
        initial_vocab_factor,
        em_iterations,
        prune_ratio,
        character_coverage: 0.9995,
    };

    if let Some(path_out) = dump_vocab {
        let model = train_unigram(&sentences, &cfg);
        let mut lines: Vec<String> = model
            .vocab
            .iter()
            .zip(model.log_probs.iter())
            .map(|(tok, lp)| format!("{tok}\t{lp:.6}"))
            .collect();
        lines.sort();
        let out = lines.join("\n") + "\n";
        if let Err(e) = std::fs::write(&path_out, out) {
            eprintln!("erro ao escrever {path_out}: {e}");
            std::process::exit(1);
        }
        return;
    }

    if let Some(sentence) = tokenize_arg {
        let model = train_unigram(&sentences, &cfg);
        let tokens = tokenize_sentence(&sentence, &model);
        println!("Unigram(vocab={}): {:?}", model.vocab.len(), tokens);
        return;
    }

    let num_lines = sentences.len();
    let total_chars: usize = sentences.iter().map(|s| s.chars().count()).sum();
    let initial_vocab_size = (target_vocab_size as f64 * initial_vocab_factor) as usize;

    println!("Unigram LM — benchmark");
    println!("----------------------");
    println!("ficheiro:         {path}");
    println!("linhas:           {num_lines}");
    println!("chars total:      {total_chars}");
    println!("vocab alvo:       {target_vocab_size}");
    println!("vocab inicial:    ~{initial_vocab_size}");
    println!("em_iterations:    {em_iterations}");
    println!("prune_ratio:      {prune_ratio}");
    println!("execuções:        {runs}");
    println!();

    let mut durations_ms: Vec<f64> = Vec::with_capacity(runs);
    let mut last_vocab_size = 0usize;

    for r in 0..runs {
        let t0 = Instant::now();
        let model = train_unigram(&sentences, &cfg);
        let elapsed = t0.elapsed();
        last_vocab_size = model.vocab.len();
        durations_ms.push(elapsed.as_secs_f64() * 1000.0);
        println!(
            "  run {}: {:>10.2} ms   vocab final: {}",
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
