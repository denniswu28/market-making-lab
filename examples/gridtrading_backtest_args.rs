use std::path::PathBuf;

use clap::Parser;
use statmm::synthetic::{
    AlgoKind, SyntheticConfig, load_fixture_csv, run_synthetic_market_maker, write_records_csv,
    write_summary_json,
};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value = "fixtures/synthetic_l2.csv")]
    fixture: PathBuf,
    #[arg(long, default_value = "out/example_args")]
    output_dir: PathBuf,
    #[arg(long)]
    name: Option<String>,
    #[arg(long)]
    output_path: Option<PathBuf>,
    #[arg(long, default_value_t = 1.0)]
    tick_size: f64,
    #[arg(long)]
    lot_size: Option<f64>,
    #[arg(long, allow_hyphen_values = true, default_value_t = -0.00005)]
    maker_fee: f64,
    #[arg(long)]
    taker_fee: Option<f64>,
    #[arg(long)]
    queue_power: Option<f64>,
    #[arg(long)]
    relative_half_spread: Option<f64>,
    #[arg(long)]
    relative_grid_interval: Option<f64>,
    #[arg(long)]
    grid_num: Option<usize>,
    #[arg(long, default_value_t = 1.0)]
    order_qty: f64,
    #[arg(long)]
    max_position: Option<f64>,
    #[arg(long)]
    skew: Option<f64>,
    #[arg(long)]
    elapse_ns: Option<i64>,
    #[arg(long)]
    record_every: Option<usize>,
    #[arg(long, default_value = "obi")]
    algo: String,
    #[arg(long)]
    transform: Option<String>,
    #[arg(long)]
    min_grid_step: Option<f64>,
    #[arg(long)]
    initial_snapshot: Option<PathBuf>,
    #[arg(long)]
    window: Option<usize>,
    #[arg(long)]
    ema_alpha: Option<f64>,
    #[arg(long)]
    look_depth_pct: Option<f64>,
    #[arg(long, default_value_t = 0.5)]
    alpha_scale: f64,
    #[arg(long)]
    normalize: bool,
    #[arg(long)]
    vamp_depth_pct: Option<f64>,
    #[arg(long)]
    target_qty_per_side: Option<f64>,
    #[arg(long)]
    glft_vol_window: Option<usize>,
    #[arg(long)]
    glft_vol_scale: Option<f64>,
    #[arg(long, num_args = 1..)]
    data_files: Vec<PathBuf>,
    #[arg(long, num_args = 0..)]
    latency_files: Vec<PathBuf>,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let fixture = args.data_files.first().unwrap_or(&args.fixture);
    let output_dir = args.output_path.as_ref().unwrap_or(&args.output_dir);
    let algo = if args.algo == "baseline" {
        AlgoKind::Baseline
    } else {
        AlgoKind::Obi
    };
    let _ = (
        args.lot_size,
        args.taker_fee,
        args.queue_power,
        args.relative_half_spread,
        args.relative_grid_interval,
        args.grid_num,
        args.skew,
        args.record_every,
        args.transform,
        args.min_grid_step,
        args.initial_snapshot,
        args.window,
        args.ema_alpha,
        args.look_depth_pct,
        args.normalize,
        args.vamp_depth_pct,
        args.target_qty_per_side,
        args.glft_vol_window,
        args.glft_vol_scale,
        args.latency_files,
    );
    std::fs::create_dir_all(output_dir)
        .map_err(|error| format!("failed to create {}: {error}", output_dir.display()))?;

    let config = SyntheticConfig {
        tick_size: args.tick_size,
        order_qty: args.order_qty,
        max_inventory: args.max_position.unwrap_or(2.0),
        entry_latency_ns: args.elapse_ns.unwrap_or(1_000_000_000),
        maker_fee: args.maker_fee,
        alpha_scale: args.alpha_scale,
        algo,
        ..SyntheticConfig::default()
    };
    let result = run_synthetic_market_maker(&config, &load_fixture_csv(fixture)?)?;
    let stem = args
        .name
        .unwrap_or_else(|| result.summary.algo.as_str().to_string());
    write_records_csv(&output_dir.join(format!("{stem}.csv")), &result)?;
    write_summary_json(
        &output_dir.join(format!("{stem}_summary.json")),
        &result.summary,
    )?;
    Ok(())
}
