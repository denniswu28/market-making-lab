use std::collections::HashMap;
use hftbacktest::prelude::*;
use hftbacktest::depth::{INVALID_MIN, INVALID_MAX};
use tracing::{trace, debug, info, warn, error};
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
        Self { n, buf: vec![0.0; n.max(1)], head: 0, len: 0, sum: 0.0 }
    }
    fn update(&mut self, x: f64) -> f64 {
        if self.n == 0 { return x; }
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
        Self { n, buf: vec![0.0; n.max(1)], head: 0, len: 0, sum: 0.0, sum2: 0.0, eps: 1e-12 }
    }
    fn update(&mut self, x: f64) -> f64 {
        if self.n == 0 { return 0.0; }
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
    fn new(alpha: f64) -> Self { Self { alpha, has: false, v: 0.0 } }
    fn update(&mut self, x: f64) -> f64 {
        if !self.has { self.v = x; self.has = true; }
        else { self.v = self.alpha * x + (1.0 - self.alpha) * self.v; }
        self.v
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
            TransformState::Z(z)   => z.update(x),
        }
    }
}

/// ---------------------------
/// Depth helpers
/// ---------------------------

fn clamp_ticks(lb: i64, ub: i64, t: i64) -> i64 {
    t.max(lb).min(ub)
}

/// Sum qty from best ask upward to (inclusive) up_to_tick.
fn sum_ask_qty_up_to<MD: MarketDepth>(depth: &MD, best_ask_tick: i64, up_to_tick: i64) -> f64 {
    if best_ask_tick == INVALID_MAX || up_to_tick < best_ask_tick { return 0.0; }
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
    if best_bid_tick == INVALID_MIN || down_to_tick > best_bid_tick { return 0.0; }
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
    if bb.is_nan() || ba.is_nan() { return (vec![], vec![]); }
    let mid = 0.5 * (bb + ba);
    let bid_floor_tick = ((mid * (1.0 - pct)) / ts).floor() as i64;
    let ask_ceil_tick  = ((mid * (1.0 + pct)) / ts).ceil() as i64;

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
fn collect_side_until_qty<MD: MarketDepth>(
    depth: &MD,
    side: Side,
    target_qty: f64,
) -> (f64, f64) {
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
    let mid = 0.5 * (depth.best_bid() + depth.best_ask()) as f64;

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
            .filter(|o| o.side == Side::Buy && o.cancellable() && !new_bid.contains_key(&o.order_id))
            .map(|o| o.order_id)
            .collect();
        let posts: Vec<(u64, f64)> = new_bid
            .into_iter()
            .filter(|(id, _)| !orders.contains_key(id))
            .collect();
        for id in cancels { 
            debug!(side="buy", order_id=id, "cancel BUY");
            let _ = hbt.cancel(0, id, false); 
        }
        for (id, px) in posts {
            debug!(side="buy", order_id=id, price=px, qty=tick_order_qty, "post BUY");
            let _ = hftbacktest::prelude::Bot::submit_buy_order(
                hbt, 0, id, px, tick_order_qty, TimeInForce::GTX, OrdType::Limit, false
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
            .filter(|o| o.side == Side::Sell && o.cancellable() && !new_ask.contains_key(&o.order_id))
            .map(|o| o.order_id)
            .collect();
        let posts: Vec<(u64, f64)> = new_ask
            .into_iter()
            .filter(|(id, _)| !orders.contains_key(id))
            .collect();
        for id in cancels {
            debug!(side="sell", order_id=id, "cancel SELL");
            let _ = hbt.cancel(0, id, false); 
        }
        for (id, px) in posts {
            debug!(side="sell", order_id=id, price=px, qty=tick_order_qty, "post SELL");
            let _ = hftbacktest::prelude::Bot::submit_sell_order(
                hbt, 0, id, px, tick_order_qty, TimeInForce::GTX, OrdType::Limit, false
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
    quote_args: (&f64, &f64, &usize, &f64, &f64, &f64, &f64)
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
        trace!(k=k,ts=hbt.current_timestamp(),"loop");
        if k % record_every == 0 { recorder.record(hbt).unwrap(); }
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
    look_depth_pct: f64,        // e.g. 0.025 => +/-2.5%
    normalize: bool,            // true => (B-A)/(B+A), false => (B-A)
    alpha_scale: f64,           // c1 in your notebook
    ts_transform: Transform,    // e.g. ZScore{window:3600}, SMA{..}, EMA{..}, None
    elapse_ns: i64,             // step
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
            trace!(best_bid=bb, best_ask=ba, mid, "BBO");
            // compute static OBI within +/- look_depth_pct of mid
            let ts = d.tick_size() as f64;
            let best_bid_tick = d.best_bid_tick();
            let best_ask_tick = d.best_ask_tick();
            if best_bid_tick == INVALID_MIN || best_ask_tick == INVALID_MAX { return mid; }
            let low_tick = ((mid * (1.0 - look_depth_pct)) / ts).floor() as i64;
            let high_tick = ((mid * (1.0 + look_depth_pct)) / ts).ceil() as i64;

            let sum_bid = sum_bid_qty_down_to(d, best_bid_tick, low_tick);
            let sum_ask = sum_ask_qty_up_to(d, best_ask_tick, high_tick);
            let raw = sum_bid - sum_ask;
            let alpha = if normalize {
                let denom = (sum_bid + sum_ask).max(1e-12);
                raw / denom
            } else { raw };

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
            if bb.is_nan() || ba.is_nan() { return f64::NAN; }
            let mid = 0.5 * (bb + ba) as f64;

            let (bids, asks) = collect_levels_by_percent(d, depth_pct);
            let k = bids.len().min(asks.len());
            if k == 0 { return mid; }

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
            if bb.is_nan() || ba.is_nan() { return f64::NAN; }
            let mid = 0.5 * (bb + ba) as f64;

            let (sum_pbqb, sum_qb) = collect_side_until_qty(d, Side::Buy,  target_qty_per_side);
            let (sum_paqa, sum_qa) = collect_side_until_qty(d, Side::Sell, target_qty_per_side);
            let den = sum_qb + sum_qa;
            let wdp = if den > 0.0 { (sum_pbqb + sum_paqa) / den } else { mid };

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
            if bb.is_nan() || ba.is_nan() { return f64::NAN; }
            let mid = 0.5 * (bb + ba) as f64;

            // within pct band, compute effective side prices
            let (bids, asks) = collect_levels_by_percent(d, depth_pct);
            let (mut sum_pbqb, mut sum_qb) = (0.0, 0.0);
            for (pb, qb) in &bids { sum_pbqb += pb * qb; sum_qb += qb; }
            let (mut sum_paqa, mut sum_qa) = (0.0, 0.0);
            for (pa, qa) in &asks { sum_paqa += pa * qa; sum_qa += qa; }

            if sum_qb <= 0.0 || sum_qa <= 0.0 { return mid; }
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

#[derive(Clone)]
struct RollingStd {
    n: usize,
    buf: Vec<f64>,
    head: usize,
    len: usize,
    sum: f64,
    sum2: f64,
    eps: f64,
}
impl RollingStd {
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
        if self.n == 0 { return 0.0; }
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
        var.max(0.0).sqrt()
    }
}

// --- NEW: GLFT-simplified algo ----------------------------------------------------
#[allow(clippy::too_many_arguments)]
pub fn grid_glft_simplified<MD, I, R>(
    hbt: &mut I,
    recorder: &mut R,
    base_relative_half_spread: f64,
    relative_grid_interval: f64,
    grid_num: usize,
    min_grid_step: f64,
    skew: f64,
    order_qty: f64,
    max_position_qty: f64,
    // GLFT-like knobs
    vol_window: usize,        // e.g. 600 (seconds-worth of ticks if elapse_ns=1s)
    vol_scale: f64,           // additive widening: rhs_eff = base_rhs + vol_scale * sigma
    price_transform: Transform,     // None | SMA(window) | EMA(alpha) | ZScore(window)
    z_as_alpha_scale: f64,    // if Transform::ZScore, mid + k * z
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
    let mut rstd = RollingStd::new(vol_window);
    let mut prev_mid: Option<f64> = None;

    // Optional initial record once BBO is ready (helps zero baseline)
    if hbt.depth(0).best_bid_tick() != INVALID_MIN && hbt.depth(0).best_ask_tick() != INVALID_MAX {
        recorder.record(hbt).unwrap();
    }

    let mut k = 0usize;
    while ElapseResult::Ok == hbt.elapse(elapse_ns).unwrap() {
        k += 1;
        trace!(k=k, ts=hbt.current_timestamp(), "glft loop");

        if k % record_every == 0 {
            recorder.record(hbt).unwrap();
        }

        let d = hbt.depth(0);
        let bb = d.best_bid();
        let ba = d.best_ask();
        if bb.is_nan() || ba.is_nan() {
            continue; // wait for BBO
        }
        let mid = 0.5 * (bb + ba) as f64;

        // rolling return std (simple pct return)
        let ret = if let Some(pm) = prev_mid {
            if pm != 0.0 { (mid / pm) - 1.0 } else { 0.0 }
        } else { 0.0 };
        prev_mid = Some(mid);
        let sigma = rstd.update(ret);

        // fair price via transform
        let fair = match price_transform {
            Transform::ZScore { .. } => {
                let z = tf.apply(mid);
                mid + z_as_alpha_scale * z
            }
            _ => tf.apply(mid),
        };

        // dynamic half-spread (GLFT-style widening by volatility)
        let rhs_eff = (base_relative_half_spread + vol_scale * sigma).max(0.0);

        // drive the grid
        update_grid::<I, MD>(
            hbt,
            fair,
            rhs_eff,
            relative_grid_interval,
            min_grid_step,
            skew,
            order_qty,
            max_position_qty,
            grid_num,
        )?;
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
        100_000_000,   // 100ms
        10,            // record every 1s
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
