use clap::{Parser, ValueEnum, ArgAction};
use hftbacktest::{
    backtest::{
        assettype::LinearAsset,
        data::{read_npz_file, DataSource},
        models::{
            CommonFees, IntpOrderLatency, PowerProbQueueFunc3, ProbQueueModel,
            TradingValueFeeModel,
        },
        recorder::BacktestRecorder,
        Backtest, ExchangeKind, L2AssetBuilder,
    }, depth::MarketDepth, prelude::{ApplySnapshot, Bot, HashMapMarketDepth}
};
use hftbacktest::depth::{INVALID_MIN, INVALID_MAX};
use statmm::algo::{
    grid_obi_static_alpha, grid_vamp_effective_fair, grid_vamp_fair, grid_weighted_depth_fair, grid_glft_simplified,
    Transform,
};
use tracing_subscriber::{fmt, EnvFilter};
use tracing::{trace, debug, info, warn, error};

#[derive(Debug, Clone, ValueEnum)]
enum AlgoKind {
    /// Static Order Book Imbalance -> alpha -> price tilt
    ObiStaticAlpha,
    /// VAMP (volume-adjusted mid) as fair price
    Vamp,
    /// Weighted-depth price (fixed qty per side) as fair price
    WeightedDepth,
    /// Effective-VAMP (side-weighted prices) as fair price
    VampEffective,
    /// GLFT-style simplified: volatility-widened half-spread + mid transform
    GlftSimple,
}

#[derive(Debug, Clone, ValueEnum)]
enum TransformKind {
    None,
    /// Simple moving average over `--window`
    Sma,
    /// Exponential moving average with `--ema-alpha`
    Ema,
    /// Z-score standardization over `--window`
    Zscore,
}

#[derive(Parser, Debug)]
#[command(about = "HFT grid backtest with selectable alpha models", long_about=None)]
struct Args {
    // ---------- I/O ----------
    #[arg(long)]
    name: String,
    #[arg(long)]
    output_path: String,
    #[arg(long, num_args = 1..)]
    data_files: Vec<String>,
    #[arg(long, num_args = 0..)]
    latency_files: Vec<String>,
    #[arg(long)]
    initial_snapshot: Option<String>,

    // ---------- instrument ----------
    #[arg(long)]
    tick_size: f64,
    #[arg(long)]
    lot_size: f64,

    // ---------- fees / queue ----------
    #[arg(long, default_value_t = -0.00005)]
    maker_fee: f64,
    #[arg(long, default_value_t = 0.0007)]
    taker_fee: f64,
    #[arg(long, default_value_t = 3.0)]
    queue_power: f64,

    // ---------- grid params ----------
    #[arg(long)]
    relative_half_spread: f64,
    #[arg(long)]
    relative_grid_interval: f64,
    #[arg(long)]
    grid_num: usize,
    #[arg(long)]
    order_qty: f64,
    #[arg(long)]
    max_position: f64,
    #[arg(long)]
    skew: f64,
    /// defaults to tick_size if not provided
    #[arg(long)]
    min_grid_step: Option<f64>,

    // ---------- time control ----------
    /// elapse interval in ns (default 1s)
    #[arg(long, default_value_t = 1_000_000_000_i64)]
    elapse_ns: i64,
    /// record every N steps (default 1)
    #[arg(long, default_value_t = 1_usize)]
    record_every: usize,

    // ---------- model selection ----------
    #[arg(long, value_enum, default_value_t = AlgoKind::ObiStaticAlpha)]
    algo: AlgoKind,
    #[arg(long, value_enum, default_value_t = TransformKind::Zscore)]
    transform: TransformKind,
    /// for SMA/Z-score
    #[arg(long)]
    window: Option<usize>,
    /// for EMA
    #[arg(long)]
    ema_alpha: Option<f64>,

    // ---------- per-algo knobs ----------
    /// OBI: +/- depth percentage around mid (e.g. 0.025 -> 2.5%)
    #[arg(long)]
    look_depth_pct: Option<f64>,
    /// OBI: use (B-A)/(B+A) instead of raw B-A
    #[arg(long, action = ArgAction::SetTrue)]
    normalize: bool,
    /// OBI: scale alpha -> price (c1)
    #[arg(long)]
    alpha_scale: Option<f64>,

    /// VAMP / Effective-VAMP: +/- depth percentage (e.g. 0.02)
    #[arg(long)]
    vamp_depth_pct: Option<f64>,

    /// Weighted-depth: target qty per side
    #[arg(long)]
    target_qty_per_side: Option<f64>,

    /// GLFT-simple: rolling return std window (ticks)
    #[arg(long, default_value_t = 600_usize)]
    glft_vol_window: usize,
    /// GLFT-simple: additive widening coefficient (rhs_eff = rhs + vol_scale * sigma)
    #[arg(long, default_value_t = 0.0)]
    glft_vol_scale: f64,
}

fn prepare_backtest(
    latency_files: Vec<String>,
    data_files: Vec<String>,
    initial_snapshot: Option<String>,
    tick_size: f64,
    lot_size: f64,
    maker_fee: f64,
    taker_fee: f64,
    queue_power: f64,
) -> Backtest<HashMapMarketDepth> {
    let latency_model = IntpOrderLatency::new(
        latency_files
            .into_iter()
            .map(DataSource::File)
            .collect::<Vec<_>>(),
        0,
    );
    let asset_type = LinearAsset::new(1.0);
    let queue_model = ProbQueueModel::new(PowerProbQueueFunc3::new(queue_power));

    let hbt = Backtest::builder()
        .add_asset(
            L2AssetBuilder::new()
                .data(
                    data_files
                        .iter()
                        .map(|file| DataSource::File(file.clone()))
                        .collect(),
                )
                .latency_model(latency_model)
                .asset_type(asset_type)
                .fee_model(TradingValueFeeModel::new(CommonFees::new(maker_fee, taker_fee)))
                .exchange(ExchangeKind::NoPartialFillExchange)
                .queue_model(queue_model)
                .depth(move || {
                    let mut depth = HashMapMarketDepth::new(tick_size, lot_size);
                    if let Some(file) = initial_snapshot.as_ref() {
                        if let Ok(npz) = read_npz_file(file, "data") {
                            depth.apply_snapshot(&npz);
                            warn!("Applied initial snapshot from {}", file);
                        }
                    }
                    depth
                })
                .build()
                .unwrap(),
        )
        .build()
        .unwrap();
    hbt
}

fn build_transform(kind: TransformKind, window: Option<usize>, ema_alpha: Option<f64>) -> Transform {
    match kind {
        TransformKind::None => Transform::None,
        TransformKind::Sma => Transform::SMA {
            window: window.unwrap_or(300),
        },
        TransformKind::Ema => Transform::EMA {
            alpha: ema_alpha.unwrap_or(0.1),
        },
        TransformKind::Zscore => Transform::ZScore {
            window: window.unwrap_or(1800),
        },
    }
}

fn main() {
    let _ = fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_target(true)
        .with_level(true)
        .try_init();
    let args = Args::parse();
    warn!(?args, "CLI args parsed");
    let min_grid_step = args.min_grid_step.unwrap_or(args.tick_size);

    let mut hbt = prepare_backtest(
        args.latency_files,
        args.data_files,
        args.initial_snapshot.clone(),
        args.tick_size,
        args.lot_size,
        args.maker_fee,
        args.taker_fee,
        args.queue_power,
    );

    let mut recorder = BacktestRecorder::new(&hbt);
    let transform = build_transform(args.transform, args.window, args.ema_alpha);

    match args.algo {
        AlgoKind::ObiStaticAlpha => {
            let depth = args
                .look_depth_pct
                .expect("--look-depth-pct is required for --algo obi-static-alpha");
            let c1 = args
                .alpha_scale
                .expect("--alpha-scale is required for --algo obi-static-alpha");
            grid_obi_static_alpha::<HashMapMarketDepth, _, _>(
                &mut hbt,
                &mut recorder,
                args.relative_half_spread,
                args.relative_grid_interval,
                args.grid_num,
                min_grid_step,
                args.skew,
                args.order_qty,
                args.max_position,
                depth,
                args.normalize,
                c1,
                transform,
                args.elapse_ns,
                args.record_every,
            )
            .unwrap();
        }
        AlgoKind::Vamp => {
            let vamp_depth = args
                .vamp_depth_pct
                .expect("--vamp-depth-pct is required for --algo vamp");
            grid_vamp_fair::<HashMapMarketDepth, _, _>(
                &mut hbt,
                &mut recorder,
                args.relative_half_spread,
                args.relative_grid_interval,
                args.grid_num,
                min_grid_step,
                args.skew,
                args.order_qty,
                args.max_position,
                vamp_depth,
                transform,
                0.0,
                args.elapse_ns,
                args.record_every,
            )
            .unwrap();
        }
        AlgoKind::WeightedDepth => {
            let tgt = args
                .target_qty_per_side
                .expect("--target-qty-per-side is required for --algo weighted-depth");
            grid_weighted_depth_fair::<HashMapMarketDepth, _, _>(
                &mut hbt,
                &mut recorder,
                args.relative_half_spread,
                args.relative_grid_interval,
                args.grid_num,
                min_grid_step,
                args.skew,
                args.order_qty,
                args.max_position,
                tgt,
                transform,
                0.0,
                args.elapse_ns,
                args.record_every,
            )
            .unwrap();
        }
        AlgoKind::VampEffective => {
            let vamp_depth = args
                .vamp_depth_pct
                .expect("--vamp-depth-pct is required for --algo vamp-effective");
            grid_vamp_effective_fair::<HashMapMarketDepth, _, _>(
                &mut hbt,
                &mut recorder,
                args.relative_half_spread,
                args.relative_grid_interval,
                args.grid_num,
                min_grid_step,
                args.skew,
                args.order_qty,
                args.max_position,
                vamp_depth,
                transform,
                0.0,
                args.elapse_ns,
                args.record_every,
            )
            .unwrap();
        }
        AlgoKind::GlftSimple => {
            // For ZScore transform, alpha_scale is used as z->price multiplier; default to 0 if not set.
            let z_k = args.alpha_scale.unwrap_or(0.0);
            grid_glft_simplified::<HashMapMarketDepth, _, _>(
                &mut hbt,
                &mut recorder,
                args.relative_half_spread,
                args.relative_grid_interval,
                args.grid_num,
                min_grid_step,
                args.skew,
                args.order_qty,
                args.max_position,
                args.glft_vol_window,
                args.glft_vol_scale,
                transform,
                z_k,
                args.elapse_ns,
                args.record_every,
            )
            .unwrap();
        }
    }

    hbt.close().unwrap();
    recorder.to_csv(&args.name, &args.output_path).unwrap();
}