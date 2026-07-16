use std::path::PathBuf;

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
struct Args {
    #[arg(long, default_value = "fixtures/synthetic_l2.csv")]
    fixture: PathBuf,
    #[arg(long, default_value = "out/example_args")]
    output_dir: PathBuf,
    #[arg(long, value_enum, default_value_t = AlgoArg::Obi)]
    algo: AlgoArg,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir)
        .map_err(|error| format!("failed to create {}: {error}", args.output_dir.display()))?;

    let config = SyntheticConfig {
        algo: args.algo.into(),
        ..SyntheticConfig::default()
    };
    let result = run_synthetic_market_maker(&config, &load_fixture_csv(&args.fixture)?)?;
    let stem = result.summary.algo.as_str();
    write_records_csv(&args.output_dir.join(format!("{stem}.csv")), &result)?;
    write_summary_json(
        &args.output_dir.join(format!("{stem}_summary.json")),
        &result.summary,
    )?;
    Ok(())
}
