use std::{collections::VecDeque, fmt::Write as _, fs, path::Path};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Level {
    pub price: f64,
    pub qty: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticEvent {
    pub exch_ts: i64,
    pub local_ts: i64,
    pub bids: Vec<Level>,
    pub asks: Vec<Level>,
}

impl SyntheticEvent {
    pub fn best_bid(&self) -> Level {
        self.bids[0]
    }

    pub fn best_ask(&self) -> Level {
        self.asks[0]
    }

    pub fn mid_price(&self) -> f64 {
        (self.best_bid().price + self.best_ask().price) * 0.5
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgoKind {
    Baseline,
    Obi,
}

impl AlgoKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Obi => "obi",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SyntheticConfig {
    pub tick_size: f64,
    pub order_qty: f64,
    pub max_inventory: f64,
    pub half_spread: f64,
    pub inventory_skew: f64,
    pub entry_latency_ns: i64,
    pub maker_fee: f64,
    pub signal_levels: usize,
    pub signal_window: usize,
    pub alpha_scale: f64,
    pub algo: AlgoKind,
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            tick_size: 1.0,
            order_qty: 1.0,
            max_inventory: 2.0,
            half_spread: 1.0,
            inventory_skew: 0.25,
            entry_latency_ns: 1_000_000_000,
            maker_fee: -0.00005,
            signal_levels: 2,
            signal_window: 3,
            alpha_scale: 0.5,
            algo: AlgoKind::Baseline,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Quote {
    side: Side,
    price: f64,
    qty: f64,
}

#[derive(Debug, Clone)]
struct PendingRefresh {
    ack_ts: i64,
    bid: Option<Quote>,
    ask: Option<Quote>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StepRecord {
    pub exch_ts: i64,
    pub local_ts: i64,
    pub mid_price: f64,
    pub fair_price: f64,
    pub signal: f64,
    pub bid_quote: Option<f64>,
    pub ask_quote: Option<f64>,
    pub inventory: f64,
    pub cash: f64,
    pub fees: f64,
    pub fill_count: usize,
    pub warmed_up: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunSummary {
    pub algo: AlgoKind,
    pub final_mid_price: f64,
    pub final_inventory: f64,
    pub final_cash: f64,
    pub final_mark_to_market: f64,
    pub total_fees: f64,
    pub fills: usize,
    pub placements: usize,
    pub cancellations: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunResult {
    pub records: Vec<StepRecord>,
    pub summary: RunSummary,
}

#[derive(Debug, Clone)]
struct SignalState {
    window: usize,
    values: VecDeque<f64>,
}

impl SignalState {
    fn new(window: usize) -> Self {
        Self {
            window: window.max(1),
            values: VecDeque::new(),
        }
    }

    fn update(&mut self, value: f64) -> (f64, bool) {
        self.values.push_back(value);
        if self.values.len() > self.window {
            self.values.pop_front();
        }
        if self.values.len() < self.window {
            return (0.0, false);
        }
        let len = self.values.len() as f64;
        let mean = self.values.iter().sum::<f64>() / len;
        let var = self
            .values
            .iter()
            .map(|sample| {
                let diff = sample - mean;
                diff * diff
            })
            .sum::<f64>()
            / len;
        let std = var.sqrt();
        if std <= f64::EPSILON {
            (0.0, true)
        } else {
            ((value - mean) / std, true)
        }
    }
}

pub fn load_fixture_csv(path: &Path) -> Result<Vec<SyntheticEvent>, String> {
    let content = fs::read_to_string(path)
        .map_err(|error| format!("failed to read fixture {}: {error}", path.display()))?;
    let mut events = Vec::new();
    for (index, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if index == 0 && line.starts_with("exch_ts,") {
            continue;
        }
        let columns: Vec<_> = line.split(',').map(str::trim).collect();
        if columns.len() != 10 {
            return Err(format!(
                "fixture {} line {} expected 10 columns, found {}",
                path.display(),
                index + 1,
                columns.len()
            ));
        }
        let parse_i64 = |value: &str, name: &str| {
            value
                .parse::<i64>()
                .map_err(|error| format!("invalid {name} '{value}' on line {}: {error}", index + 1))
        };
        let parse_f64 = |value: &str, name: &str| {
            value
                .parse::<f64>()
                .map_err(|error| format!("invalid {name} '{value}' on line {}: {error}", index + 1))
        };
        let exch_ts = parse_i64(columns[0], "exch_ts")?;
        let local_ts = parse_i64(columns[1], "local_ts")?;
        let bids = vec![
            Level {
                price: parse_f64(columns[2], "bid_px_1")?,
                qty: parse_f64(columns[3], "bid_qty_1")?,
            },
            Level {
                price: parse_f64(columns[4], "bid_px_2")?,
                qty: parse_f64(columns[5], "bid_qty_2")?,
            },
        ];
        let asks = vec![
            Level {
                price: parse_f64(columns[6], "ask_px_1")?,
                qty: parse_f64(columns[7], "ask_qty_1")?,
            },
            Level {
                price: parse_f64(columns[8], "ask_px_2")?,
                qty: parse_f64(columns[9], "ask_qty_2")?,
            },
        ];
        events.push(SyntheticEvent {
            exch_ts,
            local_ts,
            bids,
            asks,
        });
    }
    if events.is_empty() {
        return Err(format!(
            "fixture {} did not contain any events",
            path.display()
        ));
    }
    Ok(events)
}

pub fn run_synthetic_market_maker(
    config: &SyntheticConfig,
    events: &[SyntheticEvent],
) -> Result<RunResult, String> {
    if events.is_empty() {
        return Err("at least one synthetic event is required".into());
    }

    let mut ordered = events.to_vec();
    ordered.sort_by_key(|event| (event.local_ts, event.exch_ts));

    let mut working_bid: Option<Quote> = None;
    let mut working_ask: Option<Quote> = None;
    let mut pending: VecDeque<PendingRefresh> = VecDeque::new();
    let mut signal_state = SignalState::new(config.signal_window);
    let mut records = Vec::with_capacity(ordered.len());
    let mut inventory = 0.0;
    let mut cash = 0.0;
    let mut fees = 0.0;
    let mut fills = 0usize;
    let mut placements = 0usize;
    let mut cancellations = 0usize;

    for event in &ordered {
        while let Some(next_refresh) = pending.front() {
            if next_refresh.ack_ts > event.local_ts {
                break;
            }
            let refresh = pending.pop_front().expect("pending refresh exists");
            apply_refresh(
                refresh,
                &mut working_bid,
                &mut working_ask,
                &mut placements,
                &mut cancellations,
            );
        }

        if let Some(quote) = working_bid
            && event.best_ask().price <= quote.price
        {
            let trade_notional = quote.price * quote.qty;
            let trade_fee = trade_notional * config.maker_fee;
            inventory += quote.qty;
            cash -= trade_notional + trade_fee;
            fees += trade_fee;
            fills += 1;
            working_bid = None;
        }
        if let Some(quote) = working_ask
            && event.best_bid().price >= quote.price
        {
            let trade_notional = quote.price * quote.qty;
            let trade_fee = trade_notional * config.maker_fee;
            inventory -= quote.qty;
            cash += trade_notional - trade_fee;
            fees += trade_fee;
            fills += 1;
            working_ask = None;
        }

        let mid_price = event.mid_price();
        let raw_imbalance = normalized_imbalance(event, config.signal_levels);
        let (standardized_signal, warmed_up) = signal_state.update(raw_imbalance);
        let signal = match config.algo {
            AlgoKind::Baseline => 0.0,
            AlgoKind::Obi if warmed_up => standardized_signal,
            AlgoKind::Obi => 0.0,
        };
        let fair_price = mid_price + config.alpha_scale * signal;
        let desired = desired_quotes(config, event, fair_price, inventory);
        let reference = pending
            .back()
            .map(|refresh| (refresh.bid, refresh.ask))
            .unwrap_or((working_bid, working_ask));
        if desired != reference {
            pending.push_back(PendingRefresh {
                ack_ts: event.local_ts + config.entry_latency_ns,
                bid: desired.0,
                ask: desired.1,
            });
        }

        records.push(StepRecord {
            exch_ts: event.exch_ts,
            local_ts: event.local_ts,
            mid_price,
            fair_price,
            signal,
            bid_quote: working_bid.map(|quote| quote.price),
            ask_quote: working_ask.map(|quote| quote.price),
            inventory,
            cash,
            fees,
            fill_count: fills,
            warmed_up,
        });
    }

    let final_mid_price = ordered
        .last()
        .map(SyntheticEvent::mid_price)
        .ok_or_else(|| "missing final event".to_string())?;

    Ok(RunResult {
        records,
        summary: RunSummary {
            algo: config.algo,
            final_mid_price,
            final_inventory: inventory,
            final_cash: cash,
            final_mark_to_market: cash + inventory * final_mid_price,
            total_fees: fees,
            fills,
            placements,
            cancellations,
        },
    })
}

fn normalized_imbalance(event: &SyntheticEvent, levels: usize) -> f64 {
    let use_levels = levels.max(1).min(event.bids.len()).min(event.asks.len());
    let bid_qty = event
        .bids
        .iter()
        .take(use_levels)
        .map(|level| level.qty)
        .sum::<f64>();
    let ask_qty = event
        .asks
        .iter()
        .take(use_levels)
        .map(|level| level.qty)
        .sum::<f64>();
    let denom = bid_qty + ask_qty;
    if denom <= f64::EPSILON {
        0.0
    } else {
        (bid_qty - ask_qty) / denom
    }
}

fn desired_quotes(
    config: &SyntheticConfig,
    event: &SyntheticEvent,
    fair_price: f64,
    inventory: f64,
) -> (Option<Quote>, Option<Quote>) {
    let normalized_inventory = inventory / config.order_qty;
    let reservation_price = fair_price - config.inventory_skew * normalized_inventory;
    let best_bid = event.best_bid().price;
    let best_ask = event.best_ask().price;

    let bid_price = round_down(
        (reservation_price - config.half_spread).min(best_bid),
        config.tick_size,
    );
    let ask_price = round_up(
        (reservation_price + config.half_spread).max(best_ask),
        config.tick_size,
    );

    let bid = (inventory < config.max_inventory).then_some(Quote {
        side: Side::Buy,
        price: bid_price,
        qty: config.order_qty,
    });
    let ask = (inventory > -config.max_inventory).then_some(Quote {
        side: Side::Sell,
        price: ask_price,
        qty: config.order_qty,
    });
    (bid, ask)
}

fn apply_refresh(
    refresh: PendingRefresh,
    working_bid: &mut Option<Quote>,
    working_ask: &mut Option<Quote>,
    placements: &mut usize,
    cancellations: &mut usize,
) {
    apply_side_refresh(refresh.bid, working_bid, placements, cancellations);
    apply_side_refresh(refresh.ask, working_ask, placements, cancellations);
}

fn apply_side_refresh(
    desired: Option<Quote>,
    working: &mut Option<Quote>,
    placements: &mut usize,
    cancellations: &mut usize,
) {
    if *working != desired {
        if working.is_some() {
            *cancellations += 1;
        }
        if desired.is_some() {
            *placements += 1;
        }
        *working = desired;
    }
}

fn round_down(price: f64, tick_size: f64) -> f64 {
    (price / tick_size).floor() * tick_size
}

fn round_up(price: f64, tick_size: f64) -> f64 {
    (price / tick_size).ceil() * tick_size
}

pub fn write_records_csv(path: &Path, result: &RunResult) -> Result<(), String> {
    let mut output = String::from(
        "algo,exch_ts,local_ts,mid_price,fair_price,signal,bid_quote,ask_quote,inventory,cash,fees,fill_count,warmed_up\n",
    );
    for record in &result.records {
        let bid_quote = record
            .bid_quote
            .map(|value| value.to_string())
            .unwrap_or_default();
        let ask_quote = record
            .ask_quote
            .map(|value| value.to_string())
            .unwrap_or_default();
        writeln!(
            output,
            "{},{},{},{:.6},{:.6},{:.6},{},{},{:.6},{:.6},{:.6},{},{}",
            result.summary.algo.as_str(),
            record.exch_ts,
            record.local_ts,
            record.mid_price,
            record.fair_price,
            record.signal,
            bid_quote,
            ask_quote,
            record.inventory,
            record.cash,
            record.fees,
            record.fill_count,
            record.warmed_up,
        )
        .expect("writing to a String should not fail");
    }
    fs::write(path, output).map_err(|error| format!("failed to write {}: {error}", path.display()))
}

pub fn write_summary_json(path: &Path, summary: &RunSummary) -> Result<(), String> {
    let payload = format!(
        concat!(
            "{{\n",
            "  \"algo\": \"{}\",\n",
            "  \"final_mid_price\": {:.6},\n",
            "  \"final_inventory\": {:.6},\n",
            "  \"final_cash\": {:.6},\n",
            "  \"final_mark_to_market\": {:.6},\n",
            "  \"total_fees\": {:.6},\n",
            "  \"fills\": {},\n",
            "  \"placements\": {},\n",
            "  \"cancellations\": {}\n",
            "}}\n"
        ),
        summary.algo.as_str(),
        summary.final_mid_price,
        summary.final_inventory,
        summary.final_cash,
        summary.final_mark_to_market,
        summary.total_fees,
        summary.fills,
        summary.placements,
        summary.cancellations,
    );
    fs::write(path, payload).map_err(|error| format!("failed to write {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(
        exch_ts: i64,
        local_ts: i64,
        bid_px: f64,
        bid_qty: f64,
        ask_px: f64,
        ask_qty: f64,
    ) -> SyntheticEvent {
        SyntheticEvent {
            exch_ts,
            local_ts,
            bids: vec![
                Level {
                    price: bid_px,
                    qty: bid_qty,
                },
                Level {
                    price: bid_px - 1.0,
                    qty: bid_qty * 0.5,
                },
            ],
            asks: vec![
                Level {
                    price: ask_px,
                    qty: ask_qty,
                },
                Level {
                    price: ask_px + 1.0,
                    qty: ask_qty * 0.5,
                },
            ],
        }
    }

    #[test]
    fn sorts_events_by_local_then_exchange_timestamp() {
        let config = SyntheticConfig::default();
        let events = vec![
            event(2, 2, 99.0, 4.0, 101.0, 4.0),
            event(0, 1, 99.0, 4.0, 101.0, 4.0),
            event(1, 1, 99.0, 4.0, 101.0, 4.0),
        ];

        let result = run_synthetic_market_maker(&config, &events).expect("run should succeed");

        let observed: Vec<_> = result
            .records
            .iter()
            .map(|record| (record.local_ts, record.exch_ts))
            .collect();
        assert_eq!(observed, vec![(1, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn signal_warmup_uses_only_trailing_history() {
        let config = SyntheticConfig {
            algo: AlgoKind::Obi,
            signal_window: 3,
            ..SyntheticConfig::default()
        };
        let events = vec![
            event(0, 0, 99.0, 6.0, 101.0, 6.0),
            event(1, 1, 99.0, 9.0, 101.0, 3.0),
            event(2, 2, 99.0, 10.0, 101.0, 2.0),
        ];

        let result = run_synthetic_market_maker(&config, &events).expect("run should succeed");

        assert_eq!(result.records[0].signal, 0.0);
        assert_eq!(result.records[1].signal, 0.0);
        assert!(result.records[2].signal > 0.0);
        assert!(!result.records[1].warmed_up);
        assert!(result.records[2].warmed_up);
    }

    #[test]
    fn entry_latency_delays_quote_activation() {
        let config = SyntheticConfig {
            entry_latency_ns: 2,
            ..SyntheticConfig::default()
        };
        let events = vec![
            event(0, 0, 99.0, 4.0, 101.0, 4.0),
            event(1, 1, 100.0, 4.0, 102.0, 4.0),
            event(2, 2, 100.0, 4.0, 102.0, 4.0),
        ];

        let result = run_synthetic_market_maker(&config, &events).expect("run should succeed");

        assert_eq!(result.records[0].bid_quote, None);
        assert_eq!(result.records[1].bid_quote, None);
        assert_eq!(result.records[2].bid_quote, Some(99.0));
    }

    #[test]
    fn inventory_limit_turns_off_further_bids_after_fill() {
        let config = SyntheticConfig {
            max_inventory: 1.0,
            entry_latency_ns: 0,
            ..SyntheticConfig::default()
        };
        let events = vec![
            event(0, 0, 99.0, 6.0, 101.0, 6.0),
            event(1, 1, 99.0, 6.0, 99.0, 1.0),
            event(2, 2, 99.0, 6.0, 101.0, 6.0),
        ];

        let result = run_synthetic_market_maker(&config, &events).expect("run should succeed");

        assert_eq!(result.records[1].inventory, 1.0);
        assert_eq!(result.records[2].bid_quote, None);
        assert_eq!(result.summary.cancellations, 1);
    }

    #[test]
    fn no_partial_fill_policy_uses_full_order_quantity() {
        let config = SyntheticConfig {
            order_qty: 2.0,
            entry_latency_ns: 0,
            ..SyntheticConfig::default()
        };
        let events = vec![
            event(0, 0, 99.0, 6.0, 101.0, 6.0),
            event(1, 1, 99.0, 6.0, 99.0, 0.1),
        ];

        let result = run_synthetic_market_maker(&config, &events).expect("run should succeed");

        assert_eq!(result.records[1].inventory, 2.0);
        assert_eq!(result.summary.fills, 1);
    }

    #[test]
    fn maker_rebate_sign_flows_into_cash_and_mark_to_market() {
        let config = SyntheticConfig {
            entry_latency_ns: 0,
            maker_fee: -0.001,
            ..SyntheticConfig::default()
        };
        let events = vec![
            event(0, 0, 99.0, 6.0, 101.0, 6.0),
            event(1, 1, 99.0, 6.0, 99.0, 6.0),
            event(2, 2, 101.0, 6.0, 103.0, 6.0),
        ];

        let result = run_synthetic_market_maker(&config, &events).expect("run should succeed");

        assert!((result.summary.total_fees + 0.199).abs() < 1e-9);
        assert!(result.summary.final_cash > 0.0);
        assert!(result.summary.final_mark_to_market > 0.0);
    }

    #[test]
    fn quote_replacement_counts_cancellations_and_placements() {
        let config = SyntheticConfig {
            algo: AlgoKind::Obi,
            signal_window: 2,
            alpha_scale: 2.0,
            entry_latency_ns: 0,
            ..SyntheticConfig::default()
        };
        let events = vec![
            event(0, 0, 99.0, 6.0, 101.0, 6.0),
            event(1, 1, 99.0, 12.0, 101.0, 2.0),
            event(2, 2, 99.0, 2.0, 101.0, 12.0),
        ];

        let result = run_synthetic_market_maker(&config, &events).expect("run should succeed");

        assert!(result.summary.placements >= 2);
        assert!(result.summary.cancellations >= 1);
    }
}
