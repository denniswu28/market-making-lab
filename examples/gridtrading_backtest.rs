use statmm::algo::{grid_obi_static_alpha, Transform};
use hftbacktest::{
    backtest::{
        Backtest, ExchangeKind, L2AssetBuilder, assettype::LinearAsset, data::DataSource,
        models::{CommonFees, IntpOrderLatency, PowerProbQueueFunc3, ProbQueueModel, TradingValueFeeModel},
        recorder::BacktestRecorder,
    },
    prelude::{ApplySnapshot, Bot, HashMapMarketDepth},
};

fn prepare_backtest() -> Backtest<HashMapMarketDepth> {
    let base = "/data/tmp/tardis_bn_hft";
    let exch = "binance-futures";
    let sym  = "SOLUSDT";
    let date_from = 20240501;
    let date_to   = 20240531;

    let latency_data: Vec<_> = (date_from..=date_to)
        .map(|d| DataSource::File(format!("{base}/latency/{exch}/{sym}/latency_{d}.npz")))
        .collect();

    let data: Vec<_> = (date_from..=date_to)
        .map(|d| DataSource::File(format!("{base}/data/{exch}/{sym}/{sym}_{d}.npz")))
        .collect();

    let latency_model = IntpOrderLatency::new(latency_data, 0);
    let asset_type = LinearAsset::new(1.0);
    let queue_model = ProbQueueModel::new(PowerProbQueueFunc3::new(3.0));

    // If you have an SOD/EOD snapshot, set it here; otherwise you can skip apply_snapshot.
    // Example SOD snapshot path (optional):
    let sod_path = format!("{base}/data/{exch}/{sym}/{sym}_20240430_SOD.npz");

    let hbt = Backtest::builder()
        .add_asset(
            L2AssetBuilder::new()
                .data(data)
                .latency_model(latency_model)
                .asset_type(asset_type)
                .fee_model(TradingValueFeeModel::new(CommonFees::new(-0.00005, 0.0007)))
                .exchange(ExchangeKind::NoPartialFillExchange)
                .queue_model(queue_model)
                .depth({
                    let sod_path = sod_path.clone();
                    move || {
                        let mut depth = HashMapMarketDepth::new(0.01, 0.001);
                        if let Ok(npz) = hftbacktest::backtest::data::read_npz_file(&sod_path, "data") {
                            depth.apply_snapshot(&npz);
                        }
                        depth
                    }
                })
                .build()
                .unwrap(),
        )
        .build()
        .unwrap();
    hbt
}

fn main() {
    tracing_subscriber::fmt::init();

    // Grid/position knobs (same semantics as the original example)
    let relative_half_spread   = 0.0005;
    let relative_grid_interval = 0.0005;
    let grid_num               = 10;
    let min_grid_step          = 0.01;     // MUST be a multiple of tick_size
    let skew                   = relative_half_spread / grid_num as f64;
    let order_qty              = 1.0;
    let max_position_lots      = 10.0;     // measured in 'order_qty' lots

    let mut hbt = prepare_backtest();
    let mut recorder = BacktestRecorder::new(&hbt);

    // ------------------ CHOOSE ONE ALGO ------------------
    // 1) Static OBI alpha (normalized) with Z-score (window ~1h if 1s steps)
    grid_obi_static_alpha::<HashMapMarketDepth, _, _>(
        &mut hbt,
        &mut recorder,
        relative_half_spread,
        relative_grid_interval,
        grid_num,
        min_grid_step,
        skew,
        order_qty,
        max_position_lots,
        0.025, // look_depth_pct (+/-2.5% from mid)
        true,  // normalize (B-A)/(B+A)
        160.0, // alpha->price scale ("c1")
        Transform::ZScore { window: 3600 },
        1_000_000_000, // elapse 1s
        1,             // record every step
    ).unwrap();

    // // 2) VAMP_N as fair price (EMA smoothing):
    // use algo::{grid_vamp_fair, Transform};
    // grid_vamp_fair::<HashMapMarketDepth, _, _>(&mut hbt, &mut recorder,
    //     relative_half_spread, relative_grid_interval, grid_num, min_grid_step, skew,
    //     order_qty, max_position_lots,
    //     0.01,                            // +/-1% depth band
    //     Transform::EMA { alpha: 0.1 },   // smooth price
    //     0.0,                             // z->alpha scale (unused for EMA)
    //     1_000_000_000, 1).unwrap();

    // // 3) Weighted-Depth as fair (fixed qty per side) + Z-score -> alpha:
    // use algo::{grid_weighted_depth_fair, Transform};
    // grid_weighted_depth_fair::<HashMapMarketDepth, _, _>(&mut hbt, &mut recorder,
    //     relative_half_spread, relative_grid_interval, grid_num, min_grid_step, skew,
    //     order_qty, max_position_lots,
    //     500.0,                           // target_qty_per_side
    //     Transform::ZScore { window: 1800 }, // 30 min z
    //     50.0,                            // z->alpha scale
    //     1_000_000_000, 1).unwrap();

    // // 4) Effective VAMP (weighted side prices) with SMA smoothing:
    // use algo::{grid_vamp_effective_fair, Transform};
    // grid_vamp_effective_fair::<HashMapMarketDepth, _, _>(&mut hbt, &mut recorder,
    //     relative_half_spread, relative_grid_interval, grid_num, min_grid_step, skew,
    //     order_qty, max_position_lots,
    //     0.02,                            // +/-2%
    //     Transform::SMA { window: 300 },  // 5 min SMA at 1s steps
    //     0.0,
    //     1_000_000_000, 1).unwrap();

    hbt.close().unwrap();
    recorder.to_csv("gridtrading", ".").unwrap();
}