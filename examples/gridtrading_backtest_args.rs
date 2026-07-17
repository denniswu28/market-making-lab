use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use hftbacktest::{
    backtest::{
        Backtest, DataSource, ExchangeKind, L2AssetBuilder,
        assettype::LinearAsset,
        data::read_npz_file,
        models::{
            CommonFees, IntpOrderLatency, PowerProbQueueFunc3, ProbQueueModel, TradingValueFeeModel,
        },
        recorder::BacktestRecorder,
    },
    prelude::{ApplySnapshot, Bot, HashMapMarketDepth},
};
use statmm::algo::{
    Transform, grid_obi_static_alpha, grid_vamp_effective_fair, grid_vamp_fair,
    grid_weighted_depth_fair, gridtrading_with_timing,
};

#[derive(Clone, Debug, ValueEnum)]
enum Algorithm {
    Baseline,
    ObiStaticAlpha,
    Vamp,
    VampEffective,
    WeightedDepth,
}

impl Algorithm {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::ObiStaticAlpha => "obi-static-alpha",
            Self::Vamp => "vamp",
            Self::VampEffective => "vamp-effective",
            Self::WeightedDepth => "weighted-depth",
        }
    }
}

#[derive(Clone, Debug, ValueEnum)]
enum TransformKind {
    None,
    Sma,
    Ema,
    Zscore,
}

impl TransformKind {
    fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Sma => "sma",
            Self::Ema => "ema",
            Self::Zscore => "zscore",
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Experimental offline HftBacktest grid search; live execution is not supported.")]
struct Args {
    #[arg(long)]
    name: String,
    #[arg(long)]
    output_path: PathBuf,
    #[arg(long, num_args = 1..)]
    data_files: Vec<PathBuf>,
    #[arg(long, num_args = 1..)]
    latency_files: Vec<PathBuf>,
    #[arg(long)]
    initial_snapshot: Option<PathBuf>,
    #[arg(long)]
    tick_size: f64,
    #[arg(long)]
    lot_size: f64,
    #[arg(long)]
    relative_half_spread: f64,
    #[arg(long)]
    relative_grid_interval: f64,
    #[arg(long)]
    skew: f64,
    #[arg(long)]
    grid_num: usize,
    #[arg(long)]
    min_grid_step: Option<f64>,
    #[arg(long)]
    order_qty: f64,
    #[arg(long)]
    max_position: f64,
    #[arg(long, allow_hyphen_values = true, default_value_t = -0.00005)]
    maker_fee: f64,
    #[arg(long, default_value_t = 0.0007)]
    taker_fee: f64,
    #[arg(long, default_value_t = 3.0)]
    queue_power: f64,
    #[arg(long, default_value_t = 100_000_000)]
    elapse_ns: i64,
    #[arg(long, default_value_t = 10)]
    record_every: usize,
    #[arg(long, value_enum)]
    algo: Algorithm,
    #[arg(long, value_enum, default_value_t = TransformKind::None)]
    transform: TransformKind,
    #[arg(long, default_value_t = 300)]
    window: usize,
    #[arg(long, default_value_t = 0.1)]
    ema_alpha: f64,
    #[arg(long, default_value_t = 0.02)]
    look_depth_pct: f64,
    #[arg(long)]
    normalize: bool,
    #[arg(long, default_value_t = 50.0)]
    alpha_scale: f64,
    #[arg(long, default_value_t = 0.02)]
    vamp_depth_pct: f64,
    #[arg(long, default_value_t = 500.0)]
    target_qty_per_side: f64,
}

fn transform(args: &Args) -> Result<Transform, String> {
    match args.transform {
        TransformKind::None => Ok(Transform::None),
        TransformKind::Sma if args.window > 0 => Ok(Transform::SMA {
            window: args.window,
        }),
        TransformKind::Ema if (0.0..=1.0).contains(&args.ema_alpha) && args.ema_alpha > 0.0 => {
            Ok(Transform::EMA {
                alpha: args.ema_alpha,
            })
        }
        TransformKind::Zscore if args.window > 0 => Ok(Transform::ZScore {
            window: args.window,
        }),
        TransformKind::Sma | TransformKind::Zscore => {
            Err("--window must be positive for sma and zscore transforms".to_string())
        }
        TransformKind::Ema => Err("--ema-alpha must be in (0, 1] for ema transforms".to_string()),
    }
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    if args
        .data_files
        .iter()
        .any(|path| path.extension().is_some_and(|ext| ext == "csv"))
    {
        return Err(
            "CSV fixtures are only supported by the offline synthetic examples; this research CLI requires HftBacktest .npz files"
                .to_string(),
        );
    }
    if args.record_every == 0 || args.elapse_ns <= 0 {
        return Err("--record-every and --elapse-ns must be positive".to_string());
    }
    if args.tick_size <= 0.0 || args.lot_size <= 0.0 || args.order_qty <= 0.0 {
        return Err("--tick-size, --lot-size, and --order-qty must be positive".to_string());
    }
    let transform = transform(&args)?;
    std::fs::create_dir_all(&args.output_path)
        .map_err(|error| format!("failed to create {}: {error}", args.output_path.display()))?;

    let data = args
        .data_files
        .iter()
        .map(|file| DataSource::File(file.display().to_string()))
        .collect();
    let asset = L2AssetBuilder::new()
        .data(data)
        .latency_model(IntpOrderLatency::new(
            args.latency_files
                .iter()
                .map(|file| DataSource::File(file.display().to_string()))
                .collect(),
            0,
        ))
        .asset_type(LinearAsset::new(1.0))
        .fee_model(TradingValueFeeModel::new(CommonFees::new(
            args.maker_fee,
            args.taker_fee,
        )))
        .exchange(ExchangeKind::NoPartialFillExchange)
        .queue_model(ProbQueueModel::new(PowerProbQueueFunc3::new(
            args.queue_power,
        )))
        .depth({
            let snapshot = args.initial_snapshot.clone();
            move || {
                let mut depth = HashMapMarketDepth::new(args.tick_size, args.lot_size);
                if let Some(file) = &snapshot {
                    depth.apply_snapshot(
                        &read_npz_file(file.to_str().expect("UTF-8 snapshot path"), "data")
                            .expect("valid HftBacktest snapshot"),
                    );
                }
                depth
            }
        })
        .build()
        .map_err(|error| format!("failed to build HftBacktest asset: {error:?}"))?;
    let mut hbt: Backtest<HashMapMarketDepth> = Backtest::builder()
        .add_asset(asset)
        .build()
        .map_err(|error| format!("failed to build HftBacktest: {error:?}"))?;
    let mut recorder = BacktestRecorder::new(&hbt);
    let min_grid_step = args.min_grid_step.unwrap_or(args.tick_size);
    let algorithm_name = args.algo.as_str();
    let transform_name = args.transform.as_str();

    let executed_strategy = match args.algo {
        Algorithm::Baseline => {
            gridtrading_with_timing(
                &mut hbt,
                &mut recorder,
                args.relative_half_spread,
                args.relative_grid_interval,
                args.grid_num,
                min_grid_step,
                args.skew,
                args.order_qty,
                args.max_position,
                args.elapse_ns,
                args.record_every,
            )
            .map_err(|error| format!("HftBacktest strategy failed at {error}"))?;
            "baseline"
        }
        Algorithm::ObiStaticAlpha => {
            grid_obi_static_alpha(
                &mut hbt,
                &mut recorder,
                args.relative_half_spread,
                args.relative_grid_interval,
                args.grid_num,
                min_grid_step,
                args.skew,
                args.order_qty,
                args.max_position,
                args.look_depth_pct,
                args.normalize,
                args.alpha_scale,
                transform,
                args.elapse_ns,
                args.record_every,
            )
            .map_err(|error| format!("HftBacktest strategy failed at {error}"))?;
            "obi-static-alpha"
        }
        Algorithm::Vamp => {
            grid_vamp_fair(
                &mut hbt,
                &mut recorder,
                args.relative_half_spread,
                args.relative_grid_interval,
                args.grid_num,
                min_grid_step,
                args.skew,
                args.order_qty,
                args.max_position,
                args.vamp_depth_pct,
                transform,
                args.alpha_scale,
                args.elapse_ns,
                args.record_every,
            )
            .map_err(|error| format!("HftBacktest strategy failed at {error}"))?;
            "vamp"
        }
        Algorithm::VampEffective => {
            grid_vamp_effective_fair(
                &mut hbt,
                &mut recorder,
                args.relative_half_spread,
                args.relative_grid_interval,
                args.grid_num,
                min_grid_step,
                args.skew,
                args.order_qty,
                args.max_position,
                args.vamp_depth_pct,
                transform,
                args.alpha_scale,
                args.elapse_ns,
                args.record_every,
            )
            .map_err(|error| format!("HftBacktest strategy failed at {error}"))?;
            "vamp-effective"
        }
        Algorithm::WeightedDepth => {
            grid_weighted_depth_fair(
                &mut hbt,
                &mut recorder,
                args.relative_half_spread,
                args.relative_grid_interval,
                args.grid_num,
                min_grid_step,
                args.skew,
                args.order_qty,
                args.max_position,
                args.target_qty_per_side,
                transform,
                args.alpha_scale,
                args.elapse_ns,
                args.record_every,
            )
            .map_err(|error| format!("HftBacktest strategy failed at {error}"))?;
            "weighted-depth"
        }
    };
    hbt.close()
        .map_err(|error| format!("failed to close HftBacktest: {error:?}"))?;
    recorder
        .to_csv(&args.name, &args.output_path)
        .map_err(|error| format!("failed to write recorder output: {error:?}"))?;
    let manifest_path = args
        .output_path
        .join(format!("{}_run_manifest.json", args.name));
    let manifest = format!(
        concat!(
            "{{\n",
            "  \"algorithm\": \"{}\",\n",
            "  \"transform\": \"{}\",\n",
            "  \"executed_strategy\": \"{}\",\n",
            "  \"elapse_ns\": {},\n",
            "  \"record_every\": {}\n",
            "}}\n"
        ),
        algorithm_name, transform_name, executed_strategy, args.elapse_ns, args.record_every,
    );
    std::fs::write(&manifest_path, manifest)
        .map_err(|error| format!("failed to write {}: {error}", manifest_path.display()))?;
    Ok(())
}
