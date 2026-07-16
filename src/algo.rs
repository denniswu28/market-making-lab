#![allow(
    clippy::doc_overindented_list_items,
    clippy::empty_line_after_doc_comments,
    clippy::let_and_return,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::upper_case_acronyms
)]

use hftbacktest::depth::{INVALID_MAX, INVALID_MIN};
use hftbacktest::prelude::*;
use std::collections::HashMap;
use tracing::{debug, trace};
/// ---------------------------
/// Rolling utilities
/// ---------------------------
#[derive(Clone)]
struct RollingSMA {
    n: usize,
    buf: Vec<f64>,
    head: usize,
    len: usize,
    sum: f64,
}
impl RollingSMA {
    fn new(n: usize) -> Self {
        Self {
            n,
            buf: vec![0.0; n.max(1)],
            head: 0,
            len: 0,
            sum: 0.0,
        }
    }
    fn update(&mut self, x: f64) -> f64 {
        if self.n == 0 {
            return x;
        }
        if self.len < self.n {
            self.buf[self.head] = x;
            self.sum += x;
            self.head = (self.head + 1) % self.n;
            self.len += 1;
        } else {
            let old = self.buf[self.head];
            self.buf[self.head] = x;
            self.sum += x - old;
            self.head = (self.head + 1) % self.n;
        }
        self.sum / self.len as f64
    }
}

#[derive(Clone)]
struct RollingZ {
    n: usize,
    buf: Vec<f64>,
    head: usize,
    len: usize,
    sum: f64,
    sum2: f64,
    eps: f64,
}
impl RollingZ {
    fn new(n: usize) -> Self {
        Self {
            n,
            buf: vec![0.0; n.max(1)],
            head: 0,
            len: 0,
            sum: 0.0,
            sum2: 0.0,
            eps: 1e-12,
        }
    }
    fn update(&mut self, x: f64) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        if self.len < self.n {
            self.buf[self.head] = x;
            self.sum += x;
            self.sum2 += x * x;
            self.head = (self.head + 1) % self.n;
            self.len += 1;
        } else {
            let old = self.buf[self.head];
            self.buf[self.head] = x;
            self.sum += x - old;
            self.sum2 += x * x - old * old;
            self.head = (self.head + 1) % self.n;
        }
        let mean = self.sum / self.len as f64;
        let var = (self.sum2 / self.len as f64) - mean * mean;
        let std = var.max(0.0).sqrt();
        (x - mean) / (std + self.eps)
    }
}

#[derive(Clone)]
struct RollingEMA {
    alpha: f64,
    has: bool,
    v: f64,
}
impl RollingEMA {
    fn new(alpha: f64) -> Self {
        Self {
            alpha,
            has: false,
            v: 0.0,
        }
    }
    fn update(&mut self, x: f64) -> f64 {
        if !self.has {
            self.v = x;
            self.has = true;
        } else {
            self.v = self.alpha * x + (1.0 - self.alpha) * self.v;
        }
        self.v
    }
}

#[derive(Clone)]
struct RollingStd {
    n: usize,
    buf: Vec<f64>,
    head: usize,
    len: usize,
    sum: f64,
    sum2: f64,
}
impl RollingStd {
    fn new(n: usize) -> Self {
        Self {
            n: n.max(1),
            buf: vec![0.0; n.max(1)],
            head: 0,
            len: 0,
            sum: 0.0,
            sum2: 0.0,
        }
    }
    fn push(&mut self, x: f64) {
        if self.len < self.n {
            self.buf[self.head] = x;
            self.sum += x;
            self.sum2 += x * x;
            self.head = (self.head + 1) % self.n;
            self.len += 1;
        } else {
            let old = self.buf[self.head];
            self.buf[self.head] = x;
            self.sum += x - old;
            self.sum2 += x * x - old * old;
            self.head = (self.head + 1) % self.n;
        }
    }
    fn std(&self) -> f64 {
        if self.len == 0 {
            return f64::NAN;
        }
        let m = self.sum / self.len as f64;
        let v = (self.sum2 / self.len as f64) - m * m;
        v.max(0.0).sqrt()
    }
}

/// Time-series transform selection (applied to alpha OR price depending on algo)
pub enum Transform {
    None,
    SMA { window: usize },
    EMA { alpha: f64 },
    ZScore { window: usize },
}
impl Transform {
    fn to_state(&self) -> TransformState {
        match *self {
            Transform::None => TransformState::None,
            Transform::SMA { window } => TransformState::SMA(RollingSMA::new(window)),
            Transform::EMA { alpha } => TransformState::EMA(RollingEMA::new(alpha)),
            Transform::ZScore { window } => TransformState::Z(RollingZ::new(window)),
        }
    }
}
enum TransformState {
    None,
    SMA(RollingSMA),
    EMA(RollingEMA),
    Z(RollingZ),
}
impl TransformState {
    fn apply(&mut self, x: f64) -> f64 {
        match self {
            TransformState::None => x,
            TransformState::SMA(s) => s.update(x),
            TransformState::EMA(e) => e.update(x),
            TransformState::Z(z) => z.update(x),
        }
    }
}

/// ---------------------------
/// Depth helpers
/// ---------------------------

/// Sum qty from best ask upward to (inclusive) up_to_tick.
fn sum_ask_qty_up_to<MD: MarketDepth>(depth: &MD, best_ask_tick: i64, up_to_tick: i64) -> f64 {
    if best_ask_tick == INVALID_MAX || up_to_tick < best_ask_tick {
        return 0.0;
    }
    let mut s = 0.0;
    let mut t = best_ask_tick;
    while t <= up_to_tick {
        s += depth.ask_qty_at_tick(t);
        t += 1;
    }
    s
}

/// Sum qty from best bid downward to (inclusive) down_to_tick.
fn sum_bid_qty_down_to<MD: MarketDepth>(depth: &MD, best_bid_tick: i64, down_to_tick: i64) -> f64 {
    if best_bid_tick == INVALID_MIN || down_to_tick > best_bid_tick {
        return 0.0;
    }
    let mut s = 0.0;
    let mut t = best_bid_tick;
    while t >= down_to_tick {
        s += depth.bid_qty_at_tick(t);
        t -= 1;
    }
    s
}

/// Collect (price, qty) levels within +/- pct of mid on each side (tight, symmetric).
fn collect_levels_by_percent<MD: MarketDepth>(
    depth: &MD,
    pct: f64,
) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
    let ts = depth.tick_size() as f64;
    let bb = depth.best_bid();
    let ba = depth.best_ask();
    if bb.is_nan() || ba.is_nan() {
        return (vec![], vec![]);
    }
    let mid = 0.5 * (bb + ba);
    let bid_floor_tick = ((mid * (1.0 - pct)) / ts).floor() as i64;
    let ask_ceil_tick = ((mid * (1.0 + pct)) / ts).ceil() as i64;

    let mut bids = Vec::new();
    let mut t = depth.best_bid_tick();
    while t >= bid_floor_tick {
        let q = depth.bid_qty_at_tick(t);
        if q > 0.0 {
            bids.push((t as f64 * ts, q));
        }
        t -= 1;
    }
    let mut asks = Vec::new();
    let mut u = depth.best_ask_tick();
    while u <= ask_ceil_tick {
        let q = depth.ask_qty_at_tick(u);
        if q > 0.0 {
            asks.push((u as f64 * ts, q));
        }
        u += 1;
    }
    (bids, asks)
}

/// Collect side until target_qty is reached (allow partial on boundary).
fn collect_side_until_qty<MD: MarketDepth>(depth: &MD, side: Side, target_qty: f64) -> (f64, f64) {
    let ts = depth.tick_size() as f64;
    let mut acc_px_qty = 0.0;
    let mut acc_qty = 0.0;
    match side {
        Side::Buy => {
            let mut t = depth.best_bid_tick();
            while t != INVALID_MIN && acc_qty < target_qty {
                let q = depth.bid_qty_at_tick(t);
                if q > 0.0 {
                    let px = t as f64 * ts;
                    let take = f64::min(q, target_qty - acc_qty);
                    acc_px_qty += px * take;
                    acc_qty += take;
                }
                t -= 1;
            }
        }
        Side::Sell => {
            let mut t = depth.best_ask_tick();
            while t != INVALID_MAX && acc_qty < target_qty {
                let q = depth.ask_qty_at_tick(t);
                if q > 0.0 {
                    let px = t as f64 * ts;
                    let take = f64::min(q, target_qty - acc_qty);
                    acc_px_qty += px * take;
                    acc_qty += take;
                }
                t += 1;
            }
        }
        _ => {}
    }
    (acc_px_qty, acc_qty) // (sum px*qty, sum qty)
}

/// ---------------------------
/// Common grid quote update
/// ---------------------------

fn update_grid<I, MD>(
    hbt: &mut I,
    forecast_mid_price: f64,
    relative_half_spread: f64,
    relative_grid_interval: f64,
    min_grid_step: f64,
    skew: f64,
    order_qty: f64,
    max_position: f64,
    grid_num: usize,
) -> Result<(), i64>
where
    MD: MarketDepth,
    I: Bot<MD>,
{
    let tick_size = hbt.depth(0).tick_size() as f64;
    let min_grid_step = (min_grid_step / tick_size).round() * tick_size;

    let depth = hbt.depth(0);
    let position = hbt.position(0);

    if depth.best_bid_tick() == INVALID_MIN || depth.best_ask_tick() == INVALID_MAX {
        return Ok(()); // no BBO yet
    }
    // inventory skew in *relative* space; match template
    let normalized_position = position / order_qty;
    let rel_bid_depth = relative_half_spread + skew * normalized_position;
    let rel_ask_depth = relative_half_spread - skew * normalized_position;

    let mut bid_price = (forecast_mid_price * (1.0 - rel_bid_depth)).min(depth.best_bid() as f64);
    let mut ask_price = (forecast_mid_price * (1.0 + rel_ask_depth)).max(depth.best_ask() as f64);

    let grid_interval = ((forecast_mid_price * relative_grid_interval / min_grid_step).round()
        * min_grid_step)
        .max(min_grid_step);
    bid_price = (bid_price / grid_interval).floor() * grid_interval;
    ask_price = (ask_price / grid_interval).ceil() * grid_interval;
    let lot = hbt.depth(0).lot_size() as f64;
    let tick_order_qty = ((order_qty / lot).round_ties_even()) * lot;

    debug!(
        mid = forecast_mid_price,
        rel_half = relative_half_spread,
        rel_grid = relative_grid_interval,
        min_grid_step,
        skew,
        order_qty = tick_order_qty,
        max_position,
        grid_num,
        "grid update inputs"
    );

    hbt.clear_inactive_orders(Some(0));

    // BUY side
    {
        let orders = hbt.orders(0);
        let mut new_bid = HashMap::new();
        if position < max_position && bid_price.is_finite() {
            for _ in 0..grid_num {
                let tid = (bid_price / tick_size).round() as u64;
                new_bid.insert(tid, bid_price);
                bid_price -= grid_interval;
            }
        }
        let cancels: Vec<u64> = orders
            .values()
            .filter(|o| {
                o.side == Side::Buy && o.cancellable() && !new_bid.contains_key(&o.order_id)
            })
            .map(|o| o.order_id)
            .collect();
        let posts: Vec<(u64, f64)> = new_bid
            .into_iter()
            .filter(|(id, _)| !orders.contains_key(id))
            .collect();
        for id in cancels {
            debug!(side = "buy", order_id = id, "cancel BUY");
            let _ = hbt.cancel(0, id, false);
        }
        for (id, px) in posts {
            debug!(
                side = "buy",
                order_id = id,
                price = px,
                qty = tick_order_qty,
                "post BUY"
            );
            let _ = hftbacktest::prelude::Bot::submit_buy_order(
                hbt,
                0,
                id,
                px,
                tick_order_qty,
                TimeInForce::GTX,
                OrdType::Limit,
                false,
            );
        }
    }
    // SELL side
    {
        let orders = hbt.orders(0);
        let mut new_ask = HashMap::new();
        if position > -max_position && ask_price.is_finite() {
            for _ in 0..grid_num {
                let tid = (ask_price / tick_size).round() as u64;
                new_ask.insert(tid, ask_price);
                ask_price += grid_interval;
            }
        }
        let cancels: Vec<u64> = orders
            .values()
            .filter(|o| {
                o.side == Side::Sell && o.cancellable() && !new_ask.contains_key(&o.order_id)
            })
            .map(|o| o.order_id)
            .collect();
        let posts: Vec<(u64, f64)> = new_ask
            .into_iter()
            .filter(|(id, _)| !orders.contains_key(id))
            .collect();
        for id in cancels {
            debug!(side = "sell", order_id = id, "cancel SELL");
            let _ = hbt.cancel(0, id, false);
        }
        for (id, px) in posts {
            debug!(
                side = "sell",
                order_id = id,
                price = px,
                qty = tick_order_qty,
                "post SELL"
            );
            let _ = hftbacktest::prelude::Bot::submit_sell_order(
                hbt,
                0,
                id,
                px,
                tick_order_qty,
                TimeInForce::GTX,
                OrdType::Limit,
                false,
            );
        }
    }
    Ok(())
}

/// Tick loop wrapper: step elapse_ns, record every `record_every` steps, call `fair_price_fn` to compute forecast mid.
fn run_loop<I, R, MD, F>(
    hbt: &mut I,
    recorder: &mut R,
    elapse_ns: i64,
    record_every: usize,
    mut fair_price_fn: F,
    quote_args: (&f64, &f64, &usize, &f64, &f64, &f64, &f64),
) -> Result<(), i64>
where
    MD: MarketDepth,
    I: Bot<MD>,
    <I as Bot<MD>>::Error: std::fmt::Debug,
    R: Recorder,
    <R as Recorder>::Error: std::fmt::Debug,
    F: FnMut(&I) -> f64,
{
    let (rel_half, rel_grid, grid_num, min_step, skew, order_qty, max_pos_qty) = quote_args;
    let mut k = 0usize;
    while ElapseResult::Ok == hbt.elapse(elapse_ns).unwrap() {
        k += 1;
        trace!(k = k, ts = hbt.current_timestamp(), "loop");
        if k.is_multiple_of(record_every) {
            recorder.record(hbt).unwrap();
        }
        let forecast_mid = fair_price_fn(hbt);
        update_grid::<I, MD>(
            hbt,
            forecast_mid,
            *rel_half,
            *rel_grid,
            *min_step,
            *skew,
            *order_qty,
            *max_pos_qty,
            *grid_num,
        )?;
    }
    Ok(())
}

/// --------------------------------------------
/// 1) Static OBI as alpha (price modification)
/// --------------------------------------------
#[allow(clippy::too_many_arguments)]
pub fn grid_obi_static_alpha<MD, I, R>(
    hbt: &mut I,
    recorder: &mut R,
    relative_half_spread: f64,
    relative_grid_interval: f64,
    grid_num: usize,
    min_grid_step: f64,
    skew: f64,
    order_qty: f64,
    max_position_qty: f64,
    // --- alpha knobs ---
    look_depth_pct: f64,     // e.g. 0.025 => +/-2.5%
    normalize: bool,         // true => (B-A)/(B+A), false => (B-A)
    alpha_scale: f64,        // c1 in your notebook
    ts_transform: Transform, // e.g. ZScore{window:3600}, SMA{..}, EMA{..}, None
    elapse_ns: i64,          // step
    record_every: usize,
) -> Result<(), i64>
where
    MD: MarketDepth,
    I: Bot<MD>,
    <I as Bot<MD>>::Error: std::fmt::Debug,
    R: Recorder,
    <R as Recorder>::Error: std::fmt::Debug,
{
    let mut tf = ts_transform.to_state();
    run_loop::<I, R, MD, _>(
        hbt,
        recorder,
        elapse_ns,
        record_every,
        move |bot| {
            let d = bot.depth(0);
            let bb = d.best_bid();
            let ba = d.best_ask();
            if bb.is_nan() || ba.is_nan() {
                trace!("no BBO yet; skipping");
                return f64::NAN;
            }
            let mid = 0.5 * (bb + ba) as f64;
            trace!(best_bid = bb, best_ask = ba, mid, "BBO");
            // compute static OBI within +/- look_depth_pct of mid
            let ts = d.tick_size() as f64;
            let best_bid_tick = d.best_bid_tick();
            let best_ask_tick = d.best_ask_tick();
            if best_bid_tick == INVALID_MIN || best_ask_tick == INVALID_MAX {
                return mid;
            }
            let low_tick = ((mid * (1.0 - look_depth_pct)) / ts).floor() as i64;
            let high_tick = ((mid * (1.0 + look_depth_pct)) / ts).ceil() as i64;

            let sum_bid = sum_bid_qty_down_to(d, best_bid_tick, low_tick);
            let sum_ask = sum_ask_qty_up_to(d, best_ask_tick, high_tick);
            let raw = sum_bid - sum_ask;
            let alpha = if normalize {
                let denom = (sum_bid + sum_ask).max(1e-12);
                raw / denom
            } else {
                raw
            };

            let alpha_std = tf.apply(alpha);
            let fair = mid + alpha_scale * alpha_std;
            fair
        },
        (
            &relative_half_spread,
            &relative_grid_interval,
            &grid_num,
            &min_grid_step,
            &skew,
            &order_qty,
            &max_position_qty,
        ),
    )
}

/// ----------------------------------------------------
/// 2) VAMP_N (volume adjusted mid) as fair price
///     N defined by +/- pct from mid
/// ----------------------------------------------------
#[allow(clippy::too_many_arguments)]
pub fn grid_vamp_fair<MD, I, R>(
    hbt: &mut I,
    recorder: &mut R,
    relative_half_spread: f64,
    relative_grid_interval: f64,
    grid_num: usize,
    min_grid_step: f64,
    skew: f64,
    order_qty: f64,
    max_position_qty: f64,
    depth_pct: f64,             // e.g. 0.01 => 1% bands
    price_transform: Transform, // apply to the VAMP price series (SMA/EMA/ZScore/None)
    z_as_alpha_scale: f64,      // when Transform::ZScore, interpret z as alpha and do mid + k*z
    elapse_ns: i64,
    record_every: usize,
) -> Result<(), i64>
where
    MD: MarketDepth,
    I: Bot<MD>,
    <I as Bot<MD>>::Error: std::fmt::Debug,
    R: Recorder,
    <R as Recorder>::Error: std::fmt::Debug,
{
    let mut tf = price_transform.to_state();
    run_loop::<I, R, MD, _>(
        hbt,
        recorder,
        elapse_ns,
        record_every,
        move |bot| {
            let d = bot.depth(0);
            let bb = d.best_bid();
            let ba = d.best_ask();
            if bb.is_nan() || ba.is_nan() {
                return f64::NAN;
            }
            let mid = 0.5 * (bb + ba) as f64;

            let (bids, asks) = collect_levels_by_percent(d, depth_pct);
            let k = bids.len().min(asks.len());
            if k == 0 {
                return mid;
            }

            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..k {
                let (pb, qb) = bids[i];
                let (pa, qa) = asks[i];
                num += pb * qa + pa * qb;
                den += qb + qa;
            }
            let vamp = if den > 0.0 { num / den } else { mid };

            // transform semantics:
            // - SMA/EMA: transform(vamp) is the fair price
            // - ZScore: treat z as alpha on top of mid, with scale
            let fair = match price_transform {
                Transform::ZScore { .. } => {
                    let z = tf.apply(vamp);
                    mid + z_as_alpha_scale * z
                }
                _ => tf.apply(vamp),
            };
            fair
        },
        (
            &relative_half_spread,
            &relative_grid_interval,
            &grid_num,
            &min_grid_step,
            &skew,
            &order_qty,
            &max_position_qty,
        ),
    )
}

/// ----------------------------------------------------
/// 3) Weighted-Depth Order Book Price as fair price
///     N defined by *fixed* target_qty per side
/// ----------------------------------------------------
#[allow(clippy::too_many_arguments)]
pub fn grid_weighted_depth_fair<MD, I, R>(
    hbt: &mut I,
    recorder: &mut R,
    relative_half_spread: f64,
    relative_grid_interval: f64,
    grid_num: usize,
    min_grid_step: f64,
    skew: f64,
    order_qty: f64,
    max_position_qty: f64,
    target_qty_per_side: f64,   // e.g. 500 contracts on each side
    price_transform: Transform, // SMA/EMA/ZScore/None on the price or z→alpha
    z_as_alpha_scale: f64,
    elapse_ns: i64,
    record_every: usize,
) -> Result<(), i64>
where
    MD: MarketDepth,
    I: Bot<MD>,
    <I as Bot<MD>>::Error: std::fmt::Debug,
    R: Recorder,
    <R as Recorder>::Error: std::fmt::Debug,
{
    let mut tf = price_transform.to_state();
    run_loop::<I, R, MD, _>(
        hbt,
        recorder,
        elapse_ns,
        record_every,
        move |bot| {
            let d = bot.depth(0);
            let bb = d.best_bid();
            let ba = d.best_ask();
            if bb.is_nan() || ba.is_nan() {
                return f64::NAN;
            }
            let mid = 0.5 * (bb + ba) as f64;

            let (sum_pbqb, sum_qb) = collect_side_until_qty(d, Side::Buy, target_qty_per_side);
            let (sum_paqa, sum_qa) = collect_side_until_qty(d, Side::Sell, target_qty_per_side);
            let den = sum_qb + sum_qa;
            let wdp = if den > 0.0 {
                (sum_pbqb + sum_paqa) / den
            } else {
                mid
            };

            let fair = match price_transform {
                Transform::ZScore { .. } => {
                    let z = tf.apply(wdp);
                    mid + z_as_alpha_scale * z
                }
                _ => tf.apply(wdp),
            };
            fair
        },
        (
            &relative_half_spread,
            &relative_grid_interval,
            &grid_num,
            &min_grid_step,
            &skew,
            &order_qty,
            &max_position_qty,
        ),
    )
}

/// ---------------------------------------------------------------
/// 4) VAMP with side-effective (weighted) prices as fair price
///     Effective side prices by +/- pct from mid (VAMP-style range)
/// ---------------------------------------------------------------
#[allow(clippy::too_many_arguments)]
pub fn grid_vamp_effective_fair<MD, I, R>(
    hbt: &mut I,
    recorder: &mut R,
    relative_half_spread: f64,
    relative_grid_interval: f64,
    grid_num: usize,
    min_grid_step: f64,
    skew: f64,
    order_qty: f64,
    max_position_qty: f64,
    depth_pct: f64,
    price_transform: Transform,
    z_as_alpha_scale: f64,
    elapse_ns: i64,
    record_every: usize,
) -> Result<(), i64>
where
    MD: MarketDepth,
    I: Bot<MD>,
    <I as Bot<MD>>::Error: std::fmt::Debug,
    R: Recorder,
    <R as Recorder>::Error: std::fmt::Debug,
{
    let mut tf = price_transform.to_state();
    run_loop::<I, R, MD, _>(
        hbt,
        recorder,
        elapse_ns,
        record_every,
        move |bot| {
            let d = bot.depth(0);
            let bb = d.best_bid();
            let ba = d.best_ask();
            if bb.is_nan() || ba.is_nan() {
                return f64::NAN;
            }
            let mid = 0.5 * (bb + ba) as f64;

            // within pct band, compute effective side prices
            let (bids, asks) = collect_levels_by_percent(d, depth_pct);
            let (mut sum_pbqb, mut sum_qb) = (0.0, 0.0);
            for (pb, qb) in &bids {
                sum_pbqb += pb * qb;
                sum_qb += qb;
            }
            let (mut sum_paqa, mut sum_qa) = (0.0, 0.0);
            for (pa, qa) in &asks {
                sum_paqa += pa * qa;
                sum_qa += qa;
            }

            if sum_qb <= 0.0 || sum_qa <= 0.0 {
                return mid;
            }
            let p_eff_bid = sum_pbqb / sum_qb;
            let p_eff_ask = sum_paqa / sum_qa;

            // VAMP using effective side prices and total side sizes.
            let vamp_eff = (p_eff_bid * sum_qa + p_eff_ask * sum_qb) / (sum_qb + sum_qa);

            let fair = match price_transform {
                Transform::ZScore { .. } => {
                    let z = tf.apply(vamp_eff);
                    mid + z_as_alpha_scale * z
                }
                _ => tf.apply(vamp_eff),
            };
            fair
        },
        (
            &relative_half_spread,
            &relative_grid_interval,
            &grid_num,
            &min_grid_step,
            &skew,
            &order_qty,
            &max_position_qty,
        ),
    )
}

// #[allow(clippy::too_many_arguments)]
// pub fn grid_glft_simplified<MD, I, R>(
//     hbt: &mut I,
//     recorder: &mut R,
//     base_relative_half_spread: f64,
//     _relative_grid_interval: f64,   // ignored; we tie grid interval to RHS dynamically
//     grid_num: usize,
//     min_grid_step: f64,
//     skew: f64,
//     order_qty: f64,                 // baseline qty → derive baseline notional from first mid
//     max_position_qty: f64,
//     // GLFT-like knobs
//     vol_window: usize,              // rolling window length in *steps*
//     vol_scale: f64,                 // ticks-per-sigma multiplier (tutorial’s vol_to_half_spread)
//     _price_transform: Transform,    // kept for API parity (not used; tutorial uses microprice)
//     _z_as_alpha_scale: f64,         // kept for API parity
//     elapse_ns: i64,
//     record_every: usize,
//     vol_refresh_ns: i64,            // NEW: refresh cadence for σ updates
// ) -> Result<(), i64>
// where
//     MD: MarketDepth,
//     I: Bot<MD>,
//     <I as Bot<MD>>::Error: std::fmt::Debug,
//     R: Recorder,
//     <R as Recorder>::Error: std::fmt::Debug,
// {
//     let mut rstd = RollingStd::new(vol_window);

//     let tick_size = hbt.depth(0).tick_size() as f64;
//     let lot_size  = hbt.depth(0).lot_size()  as f64;

//     let step_ns = elapse_ns.max(1);
//     let refresh_steps = ((vol_refresh_ns.max(step_ns)) / step_ns) as usize;

//     // sqrt(minutes) factor: with a vol_window of W steps, each step elapse_ns long
//     // window_minutes = W * elapse_ns / 60e9 → multiplier = sqrt(window_minutes)
//     let window_minutes = (vol_window as f64) * (elapse_ns as f64) / 60_000_000_000.0;
//     let vol_time_mod = window_minutes.max(0.0).sqrt();

//     let mut prev_mid_tick: Option<f64> = None;
//     let mut samples: usize = 0;
//     let mut sigma_tick: f64 = 0.0;           // start conservative
//     let mut baseline_notional: Option<f64> = None;

//     // Optional initial record once BBO is ready
//     if hbt.depth(0).best_bid_tick() != INVALID_MIN && hbt.depth(0).best_ask_tick() != INVALID_MAX {
//         recorder.record(hbt).unwrap();
//     }

//     let mut k = 0usize;
//     while ElapseResult::Ok == hbt.elapse(elapse_ns).unwrap() {
//         k += 1;
//         if k % record_every == 0 { recorder.record(hbt).unwrap(); }

//         let d = hbt.depth(0);
//         let bb  = d.best_bid();
//         let ba  = d.best_ask();
//         let bbq = d.best_bid_qty();
//         let baq = d.best_ask_qty();

//         if bb.is_nan() || ba.is_nan() { continue; } // wait for BBO

//         let mid = 0.5 * (bb + ba) as f64;
//         let fair = (bb as f64 * baq + ba as f64 * bbq) / (baq + bbq);

//         // establish baseline notional from the passed-in order_qty on first tick with mid
//         if baseline_notional.is_none() {
//             baseline_notional = Some(order_qty.max(lot_size) * mid);
//         }

//         // rolling sigma in ticks
//         let mid_tick = mid / tick_size;
//         if let Some(pm) = prev_mid_tick {
//             let d_tick = mid_tick - pm;
//             let cur_std = rstd.update(d_tick); // <-- do NOT double-update
//             samples += 1;

//             if samples >= vol_window && (k % refresh_steps == 0) {
//                 sigma_tick = cur_std * vol_time_mod;
//             }
//         }
//         prev_mid_tick = Some(mid_tick);

//         // half-spread in *ticks* and then → *relative*
//         // tutorial: half_spread_tick = sigma_tick * vol_scale
//         let half_spread_tick = (sigma_tick.max(0.0)) * vol_scale;
//         let rhs_vol_rel = ((half_spread_tick * tick_size) / fair).max(0.0);
//         // additive widening with configured base
//         let rhs_eff = (base_relative_half_spread + rhs_vol_rel).max(0.0);
//         // tie the grid interval to half-spread (tutorial behavior)
//         let rgi_eff = rhs_eff;

//         // dynamic order size to keep baseline notional approximately constant
//         let oq_notional = baseline_notional.unwrap_or(order_qty.max(lot_size) * mid);
//         let mut order_qty_dyn = (oq_notional / fair / lot_size).round();
//         let order_qty_dyn = order_qty_dyn * lot_size;

//         // let grids_cap = (max_position_qty / order_qty.max(lot_size)).max(0.0);
//         let max_notional_position = 1000.0;

//         let max_position_qty_dyn = (max_notional_position / mid).max(lot_size);

//         let skew_tut = skew; // read from CLI/config (set to 1.0 for parity)
//         let skew_eff = if max_notional_position > 0.0 {
//             skew_tut * (order_qty_dyn * mid / max_notional_position)
//         } else {
//             0.0
//         };

//         // run the grid
//         update_grid::<I, MD>(
//             hbt,
//             fair,
//             rhs_eff,
//             rgi_eff,         // keep tied to RHS (tutorial behavior)
//             min_grid_step,
//             skew_eff,        // <— dynamic skew matching tutorial math
//             order_qty_dyn,   // dynamic $-anchored order size
//             max_position_qty_dyn, // <— dynamic quantity cap matching tutorial notional cap
//             grid_num,
//         )?;
//     }
//     Ok(())
// }

#[allow(clippy::too_many_arguments)]
pub fn grid_glft_simplified<MD, I, R>(
    hbt: &mut I,
    recorder: &mut R,
    // tutorial parameters
    vol_to_half_spread: f64, // "vol_to_half_spread" (scale)
    min_grid_step: f64,      // price units
    grid_num: usize,
    skew: f64,
    max_position_qty: f64, // hard position cap in *qty*
    // implementation knobs
    vol_window_ticks: usize,       // typically 6000 for 10 minutes @100ms
    order_value_usd: f64,          // $100 in the tutorial
    max_notional_cap: Option<f64>, // optional fixed notional cap; if None use qty cap
    // time control
    elapse_ns: i64,      // 100_000_000 in tutorial (100ms)
    record_every: usize, // 10 in tutorial (record every 1s)
) -> Result<(), i64>
where
    MD: MarketDepth,
    I: Bot<MD>,
    <I as Bot<MD>>::Error: std::fmt::Debug,
    R: Recorder,
    <R as Recorder>::Error: std::fmt::Debug,
{
    let asset = 0usize;
    let tick_size = hbt.depth(asset).tick_size() as f64;
    let lot_size = hbt.depth(asset).lot_size() as f64;

    // tutorial updates vol every 5s → derive from elapse_ns
    let five_sec = 5_000_000_000_i64;
    let vol_update_every = ((five_sec / elapse_ns).max(1)) as usize;

    let mut t: usize = 0;
    let mut prev_mid_tick: f64 = f64::NAN;
    let mut vol_tick_std: f64 = f64::NAN; // std of mid_tick change * sqrt(10)

    let mut roll = RollingStd::new(vol_window_ticks);

    while ElapseResult::Ok == hbt.elapse(elapse_ns).unwrap() {
        t += 1;

        // always clear inactive with a short, exclusive borrow
        hbt.clear_inactive_orders(Some(asset));

        // -------- READ-ONLY BLOCK (depth, prices, vol, build next grids) --------
        // Collect everything we need without mutating hbt, then *drop* borrows.
        let (maybe_new_bid, maybe_new_ask, order_qty) = {
            let d = hbt.depth(asset);
            let bb = d.best_bid();
            let ba = d.best_ask();
            if bb.is_nan() || ba.is_nan() {
                if t.is_multiple_of(record_every) {
                    recorder.record(hbt).unwrap();
                }
                continue;
            }

            let bbq = d.best_bid_qty();
            let baq = d.best_ask_qty();
            if bbq <= 0.0 || baq <= 0.0 {
                if t.is_multiple_of(record_every) {
                    recorder.record(hbt).unwrap();
                }
                continue;
            }

            // micro & mid
            let mid = 0.5 * (bb + ba) as f64;
            let mp = (bb as f64 * baq + ba as f64 * bbq) / (baq + bbq);

            // volatility from mid tick deltas
            let mid_tick = mid / tick_size;
            if prev_mid_tick.is_finite() {
                roll.push(mid_tick - prev_mid_tick);
                if t.is_multiple_of(vol_update_every) && roll.len >= vol_window_ticks {
                    vol_tick_std = roll.std() * 10f64.sqrt();
                }
            }
            prev_mid_tick = mid_tick;

            // normalized position (note: *no* borrow of depth here)
            let pos_qty = hbt.position(asset); // standalone read; fine inside this scope
            let notional = pos_qty * mid;
            let norm_pos = if let Some(max_notional) = max_notional_cap {
                if max_notional > 0.0 {
                    (notional / max_notional).clamp(-1.0, 1.0)
                } else {
                    0.0
                }
            } else if max_position_qty > 0.0 {
                (pos_qty / max_position_qty).clamp(-1.0, 1.0)
            } else {
                0.0
            };

            // depths in ticks from vol
            let half_spread_tick = vol_tick_std * vol_to_half_spread;
            let bid_depth_tick = half_spread_tick * (1.0 + skew * norm_pos);
            let ask_depth_tick = half_spread_tick * (1.0 - skew * norm_pos);

            // forecast mid = microprice
            let fair = mp;

            // $100 order size, rounded to lot
            let mut oq = (order_value_usd / mid / lot_size).round();
            if oq < 1.0 {
                oq = 1.0;
            }
            let order_qty = oq * lot_size;

            // clamp to BBO
            let mut bid_px = (fair - bid_depth_tick * tick_size).min(bb as f64);
            let mut ask_px = (fair + ask_depth_tick * tick_size).max(ba as f64);

            // grid interval
            let grid_interval = {
                let raw = (half_spread_tick * tick_size) / min_grid_step;
                (raw.round() * min_grid_step).max(min_grid_step)
            };

            // align to grid
            if bid_px.is_finite() && grid_interval.is_finite() {
                bid_px = (bid_px / grid_interval).floor() * grid_interval;
            }
            if ask_px.is_finite() && grid_interval.is_finite() {
                ask_px = (ask_px / grid_interval).ceil() * grid_interval;
            }

            // build target grids as plain HashMaps (no borrow from hbt)
            let mut new_bid: HashMap<u64, f64> = HashMap::new();
            if norm_pos < 1.0 && bid_px.is_finite() {
                let mut px = bid_px;
                for _ in 0..grid_num {
                    let tid = (px / tick_size).round() as u64;
                    new_bid.insert(tid, px);
                    px -= grid_interval;
                }
            }

            let mut new_ask: HashMap<u64, f64> = HashMap::new();
            if norm_pos > -1.0 && ask_px.is_finite() {
                let mut px = ask_px;
                for _ in 0..grid_num {
                    let tid = (px / tick_size).round() as u64;
                    new_ask.insert(tid, px);
                    px += grid_interval;
                }
            }

            (Some(new_bid), Some(new_ask), order_qty)
        }; // <-- 'd' (and any read-only borrows) drop here

        let new_bid = maybe_new_bid.unwrap();
        let new_ask = maybe_new_ask.unwrap();

        // -------- READ ORDERS (immutable), decide cancels/posts, then drop borrow --------
        let (cancels, post_bids, post_asks) = {
            let orders = hbt.orders(asset); // immutable borrow limited to this block
            // cancels
            let mut cancels: Vec<u64> = Vec::new();
            for o in orders.values() {
                if o.cancellable() {
                    let keep = match o.side {
                        Side::Buy => new_bid.contains_key(&o.order_id),
                        Side::Sell => new_ask.contains_key(&o.order_id),
                        _ => true,
                    };
                    if !keep {
                        cancels.push(o.order_id);
                    }
                }
            }
            // posts (only where not already working)
            let mut post_bids: Vec<(u64, f64)> = Vec::new();
            for (id, px) in new_bid.iter() {
                if !orders.contains_key(id) {
                    post_bids.push((*id, *px));
                }
            }
            let mut post_asks: Vec<(u64, f64)> = Vec::new();
            for (id, px) in new_ask.iter() {
                if !orders.contains_key(id) {
                    post_asks.push((*id, *px));
                }
            }
            (cancels, post_bids, post_asks)
        }; // <-- 'orders' borrow ends here

        // -------- MUTATIONS (safe: no immutable borrows alive) --------
        for id in cancels {
            let _ = hbt.cancel(asset, id, false);
        }
        for (id, px) in post_bids {
            let _ = Bot::submit_buy_order(
                hbt,
                asset,
                id,
                px,
                order_qty,
                TimeInForce::GTX,
                OrdType::Limit,
                false,
            );
        }
        for (id, px) in post_asks {
            let _ = Bot::submit_sell_order(
                hbt,
                asset,
                id,
                px,
                order_qty,
                TimeInForce::GTX,
                OrdType::Limit,
                false,
            );
        }

        // record after updates
        if t.is_multiple_of(record_every) {
            recorder.record(hbt).unwrap();
        }
    }

    Ok(())
}

/// ------------------------------------------------------------
/// Your original no-alpha grid for completeness (unchanged)
/// ------------------------------------------------------------
pub fn gridtrading<MD, I, R>(
    hbt: &mut I,
    recorder: &mut R,
    relative_half_spread: f64,
    relative_grid_interval: f64,
    grid_num: usize,
    min_grid_step: f64,
    skew: f64,
    order_qty: f64,
    max_position: f64,
) -> Result<(), i64>
where
    MD: MarketDepth,
    I: Bot<MD>,
    <I as Bot<MD>>::Error: std::fmt::Debug,
    R: Recorder,
    <R as Recorder>::Error: std::fmt::Debug,
{
    run_loop::<I, R, MD, _>(
        hbt,
        recorder,
        100_000_000, // 100ms
        10,          // record every 1s
        move |bot| {
            let d = bot.depth(0);
            let mid = 0.5 * (d.best_bid() + d.best_ask()) as f64;
            mid
        },
        (
            &relative_half_spread,
            &relative_grid_interval,
            &grid_num,
            &min_grid_step,
            &skew,
            &order_qty,
            &max_position,
        ),
    )
}
