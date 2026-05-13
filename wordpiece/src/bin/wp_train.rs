use std::time::Instant;
use wordpiece::{token_frequencies, tokenize_word, train_wordpiece, TrainConfig};

const DEFAULT_WORDLIST: &str = "/usr/share/dict/brazilian";

fn usage() -> ! {
    eprintln!(
        "\
Uso: wp-train [opções]

Por omissão: --file {DEFAULT_WORDLIST}

Opções:
  --file <CAMINHO>      Lista de palavras (UTF-8, uma por linha; # inicia comentário)
  --max-words <N>       Só as primeiras N entradas úteis
  --vocab-size <V>      Tamanho alvo do vocabulário (default: 1000)
  --min-pair <K>        min_pair_count (default: 2)
  --runs <R>            Repetições do treino (default: 3)
  --dump-vocab <PATH>   Salva vocabulário (um token por linha) e sai
  --tokenize <PALAVRA>  Tokeniza uma palavra e imprime os tokens
  --top-tokens <N>      Imprime os N tokens mais frequentes após treino
  --help, -h

Exemplo:
  cargo run --release -- --vocab-size 500 --runs 3
  cargo run --release -- --tokenize tokenização
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

fn load_corpus(path: &str, max_words: Option<usize>) -> std::io::Result<Vec<String>> {
    let text = std::fs::read_to_string(path)?;
    let mut out: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| l.split_whitespace().next())
        .map(str::to_string)
        .collect();
    if let Some(m) = max_words {
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
    let max_words = arg_opt_usize(&args, "--max-words");
    let vocab_size = arg_usize(&args, "--vocab-size", 1000);
    let min_pair_count = arg_u64(&args, "--min-pair", 2);
    let runs = arg_usize(&args, "--runs", 3).max(1);
    let dump_vocab = arg_str(&args, "--dump-vocab");
    let tokenize_word_arg = arg_str(&args, "--tokenize");
    let top_tokens = arg_opt_usize(&args, "--top-tokens");

    let corpus = match load_corpus(&path, max_words) {
        Ok(c) if !c.is_empty() => c,
        Ok(_) => {
            eprintln!("erro: ficheiro sem palavras válidas: {path}");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("erro ao ler {path}: {e}");
            std::process::exit(1);
        }
    };

    let cfg = TrainConfig { vocab_size, min_pair_count };

    if let Some(path_out) = dump_vocab {
        let model = train_wordpiece(&corpus, &cfg);
        let out = model.vocab.join("\n") + "\n";
        if let Err(e) = std::fs::write(&path_out, out) {
            eprintln!("erro ao escrever {path_out}: {e}");
            std::process::exit(1);
        }
        return;
    }

    if let Some(word) = tokenize_word_arg {
        let model = train_wordpiece(&corpus, &cfg);
        let tokens = tokenize_word(&word, &model);
        println!("WordPiece(vocab={}): {:?}", model.vocab.len(), tokens);
        return;
    }

    let num_words = corpus.len();
    let total_chars: usize = corpus.iter().map(|w| w.chars().count()).sum();
    let avg_len = total_chars as f64 / num_words as f64;

    println!("WordPiece — benchmark");
    println!("---------------------");
    println!("ficheiro:        {path}");
    println!("palavras:        {num_words}");
    println!("chars total:     {total_chars}");
    println!("média chars/pal: {avg_len:.1}");
    println!("vocab_size alvo: {vocab_size}");
    println!("min_pair_count:  {min_pair_count}");
    println!("execuções:       {runs}");
    println!();

    let mut durations_ms: Vec<f64> = Vec::with_capacity(runs);
    let mut last_vocab_size = 0usize;

    for r in 0..runs {
        let t0 = Instant::now();
        let model = train_wordpiece(&corpus, &cfg);
        let elapsed = t0.elapsed();
        last_vocab_size = model.vocab.len();
        durations_ms.push(elapsed.as_secs_f64() * 1000.0);
        println!(
            "  run {}: {:>10.2} ms   vocab: {}",
            r + 1,
            durations_ms[r],
            model.vocab.len()
        );

        if r + 1 == runs {
            if let Some(n) = top_tokens {
                let mut freqs: Vec<(String, u64)> = token_frequencies(&corpus, &model)
                    .into_iter()
                    .collect();
                freqs.sort_by(|a, b| b.1.cmp(&a.1));
                println!();
                println!("Top {} tokens por frequência:", n);
                for (tok, freq) in freqs.iter().take(n) {
                    println!("  {:>8}  {}", freq, tok);
                }
            }
        }
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
        "  throughput:    {:.0} palavras/s (médio)",
        num_words as f64 / (avg_ms / 1000.0)
    );
    println!("  vocab final:   {last_vocab_size}");
}
