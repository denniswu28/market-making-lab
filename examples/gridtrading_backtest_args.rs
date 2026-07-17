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
    #[arg(long)]
    window: Option<usize>,
    #[arg(long)]
    ema_alpha: Option<f64>,
    #[arg(long)]
    look_depth_pct: Option<f64>,
    #[arg(long)]
    normalize: bool,
    #[arg(long)]
    alpha_scale: Option<f64>,
    #[arg(long)]
    vamp_depth_pct: Option<f64>,
    #[arg(long)]
    target_qty_per_side: Option<f64>,
}

fn transform(args: &Args) -> Result<(Transform, String), String> {
    match args.transform {
        TransformKind::None if args.window.is_none() && args.ema_alpha.is_none() => {
            Ok((Transform::None, "{}".to_string()))
        }
        TransformKind::None => {
            Err("--window and --ema-alpha are not applicable to transform none".to_string())
        }
        TransformKind::Sma | TransformKind::Zscore if args.ema_alpha.is_some() => {
            Err("--ema-alpha is only applicable to the ema transform".to_string())
        }
        TransformKind::Sma | TransformKind::Zscore => {
            let window = args.window.unwrap_or(300);
            if window == 0 {
                return Err("--window must be positive for sma and zscore transforms".to_string());
            }
            let parameters = format!("{{\"window\":{window}}}");
            if matches!(&args.transform, TransformKind::Sma) {
                Ok((Transform::SMA { window }, parameters))
            } else {
                Ok((Transform::ZScore { window }, parameters))
            }
        }
        TransformKind::Ema if args.window.is_some() => {
            Err("--window is only applicable to sma and zscore transforms".to_string())
        }
        TransformKind::Ema => {
            let alpha = args.ema_alpha.unwrap_or(0.1);
            if !(0.0..=1.0).contains(&alpha) || alpha == 0.0 {
                return Err("--ema-alpha must be in (0, 1] for ema transforms".to_string());
            }
            Ok((
                Transform::EMA { alpha },
                format!("{{\"ema_alpha\":{alpha}}}"),
            ))
        }
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
    match &args.algo {
        Algorithm::Baseline
            if !matches!(&args.transform, TransformKind::None)
                || args.look_depth_pct.is_some()
                || args.normalize
                || args.alpha_scale.is_some()
                || args.vamp_depth_pct.is_some()
                || args.target_qty_per_side.is_some() =>
        {
            return Err(
                "baseline requires --transform none and accepts no signal parameters".to_string(),
            );
        }
        Algorithm::ObiStaticAlpha
            if args.vamp_depth_pct.is_some() || args.target_qty_per_side.is_some() =>
        {
            return Err(
                "obi-static-alpha does not accept VAMP or weighted-depth parameters".to_string(),
            );
        }
        Algorithm::Vamp | Algorithm::VampEffective
            if args.look_depth_pct.is_some()
                || args.normalize
                || args.target_qty_per_side.is_some() =>
        {
            return Err(
                "VAMP algorithms accept only --vamp-depth-pct and --alpha-scale".to_string(),
            );
        }
        Algorithm::WeightedDepth
            if args.look_depth_pct.is_some() || args.normalize || args.vamp_depth_pct.is_some() =>
        {
            return Err(
                "weighted-depth accepts only --target-qty-per-side and --alpha-scale".to_string(),
            );
        }
        _ => {}
    }
    let alpha_scale = args.alpha_scale.unwrap_or(50.0);
    let look_depth_pct = args.look_depth_pct.unwrap_or(0.02);
    let vamp_depth_pct = args.vamp_depth_pct.unwrap_or(0.02);
    let target_qty_per_side = args.target_qty_per_side.unwrap_or(500.0);
    if !alpha_scale.is_finite()
        || !look_depth_pct.is_finite()
        || look_depth_pct <= 0.0
        || !vamp_depth_pct.is_finite()
        || vamp_depth_pct <= 0.0
        || !target_qty_per_side.is_finite()
        || target_qty_per_side <= 0.0
    {
        return Err(
            "algorithm numeric parameters must be finite and depths/quantities positive"
                .to_string(),
        );
    }
    let (transform, transform_parameters) = transform(&args)?;
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
    let (executed_strategy, executed_transform, executed_parameters) = match args.algo {
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
            ("baseline", "none", "{}".to_string())
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
                look_depth_pct,
                args.normalize,
                alpha_scale,
                transform,
                args.elapse_ns,
                args.record_every,
            )
            .map_err(|error| format!("HftBacktest strategy failed at {error}"))?;
            (
                "obi-static-alpha",
                args.transform.as_str(),
                format!(
                    "{{\"look_depth_pct\":{},\"normalize\":{},\"alpha_scale\":{}}}",
                    look_depth_pct, args.normalize, alpha_scale
                ),
            )
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
                vamp_depth_pct,
                transform,
                alpha_scale,
                args.elapse_ns,
                args.record_every,
            )
            .map_err(|error| format!("HftBacktest strategy failed at {error}"))?;
            (
                "vamp",
                args.transform.as_str(),
                format!(
                    "{{\"vamp_depth_pct\":{},\"alpha_scale\":{}}}",
                    vamp_depth_pct, alpha_scale
                ),
            )
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
                vamp_depth_pct,
                transform,
                alpha_scale,
                args.elapse_ns,
                args.record_every,
            )
            .map_err(|error| format!("HftBacktest strategy failed at {error}"))?;
            (
                "vamp-effective",
                args.transform.as_str(),
                format!(
                    "{{\"vamp_depth_pct\":{},\"alpha_scale\":{}}}",
                    vamp_depth_pct, alpha_scale
                ),
            )
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
                target_qty_per_side,
                transform,
                alpha_scale,
                args.elapse_ns,
                args.record_every,
            )
            .map_err(|error| format!("HftBacktest strategy failed at {error}"))?;
            (
                "weighted-depth",
                args.transform.as_str(),
                format!(
                    "{{\"target_qty_per_side\":{},\"alpha_scale\":{}}}",
                    target_qty_per_side, alpha_scale
                ),
            )
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
            "  \"executed_parameters\": {},\n",
            "  \"transform_parameters\": {},\n",
            "  \"elapse_ns\": {},\n",
            "  \"record_every\": {}\n",
            "}}\n"
        ),
        algorithm_name,
        executed_transform,
        executed_strategy,
        executed_parameters,
        transform_parameters,
        args.elapse_ns,
        args.record_every,
    );
    std::fs::write(&manifest_path, manifest)
        .map_err(|error| format!("failed to write {}: {error}", manifest_path.display()))?;
    Ok(())
}
