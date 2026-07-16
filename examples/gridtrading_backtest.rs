use std::path::Path;

use statmm::synthetic::{
    AlgoKind, SyntheticConfig, load_fixture_csv, run_synthetic_market_maker, write_records_csv,
    write_summary_json,
};

fn main() -> Result<(), String> {
    let fixture = Path::new("fixtures/synthetic_l2.csv");
    let output_dir = Path::new("out/example_baseline");
    std::fs::create_dir_all(output_dir)
        .map_err(|error| format!("failed to create {}: {error}", output_dir.display()))?;

    let config = SyntheticConfig {
        algo: AlgoKind::Baseline,
        ..SyntheticConfig::default()
    };
    let result = run_synthetic_market_maker(&config, &load_fixture_csv(fixture)?)?;
    write_records_csv(&output_dir.join("baseline.csv"), &result)?;
    write_summary_json(&output_dir.join("baseline_summary.json"), &result.summary)?;
    Ok(())
}
