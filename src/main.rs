use std::{fs, path::PathBuf};

use clap::{Parser, ValueEnum};
use statmm::synthetic::{
    AlgoKind, SyntheticConfig, load_fixture_csv, run_synthetic_market_maker, write_records_csv,
    write_summary_json,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum AlgoArg {
    Baseline,
    Obi,
}

impl From<AlgoArg> for AlgoKind {
    fn from(value: AlgoArg) -> Self {
        match value {
            AlgoArg::Baseline => AlgoKind::Baseline,
            AlgoArg::Obi => AlgoKind::Obi,
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Run the deterministic synthetic market-making research fixture")]
struct Args {
    #[arg(long)]
    fixture: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, value_enum, default_value_t = AlgoArg::Baseline)]
    algo: AlgoArg,
    #[arg(long, default_value_t = 1.0)]
    tick_size: f64,
    #[arg(long, default_value_t = 1.0)]
    order_qty: f64,
    #[arg(long, default_value_t = 2.0)]
    max_inventory: f64,
    #[arg(long, default_value_t = 1.0)]
    half_spread: f64,
    #[arg(long, default_value_t = 0.25)]
    inventory_skew: f64,
    #[arg(long, default_value_t = 1_000_000_000_i64)]
    entry_latency_ns: i64,
    #[arg(long, default_value_t = -0.00005)]
    maker_fee: f64,
    #[arg(long, default_value_t = 2_usize)]
    signal_levels: usize,
    #[arg(long, default_value_t = 3_usize)]
    signal_window: usize,
    #[arg(long, default_value_t = 0.5)]
    alpha_scale: f64,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let events = load_fixture_csv(&args.fixture)?;
    let config = SyntheticConfig {
        tick_size: args.tick_size,
        order_qty: args.order_qty,
        max_inventory: args.max_inventory,
        half_spread: args.half_spread,
        inventory_skew: args.inventory_skew,
        entry_latency_ns: args.entry_latency_ns,
        maker_fee: args.maker_fee,
        signal_levels: args.signal_levels,
        signal_window: args.signal_window,
        alpha_scale: args.alpha_scale,
        algo: args.algo.into(),
    };
    let result = run_synthetic_market_maker(&config, &events)?;
    fs::create_dir_all(&args.output_dir)
        .map_err(|error| format!("failed to create {}: {error}", args.output_dir.display()))?;

    let stem = result.summary.algo.as_str();
    let csv_path = args.output_dir.join(format!("{stem}.csv"));
    let summary_path = args.output_dir.join(format!("{stem}_summary.json"));
    write_records_csv(&csv_path, &result)?;
    write_summary_json(&summary_path, &result.summary)?;

    println!(
        "{} final_mtm={:.6} inventory={:.6} fills={} csv={} summary={}",
        stem,
        result.summary.final_mark_to_market,
        result.summary.final_inventory,
        result.summary.fills,
        csv_path.display(),
        summary_path.display()
    );
    Ok(())
}
