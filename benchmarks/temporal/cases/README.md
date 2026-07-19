# Temporal backtest cases

Each `<case_id>.json` here is a **resolved real-world episode** with strict as-of separation
(see `benchmarks/temporal/harness.py` for the schema): known send/response timestamps, known
publication and exposure timing, known institutional stage durations, known filing/decision
dates, known launch/reaction timing, known event sequences.

**Status: EMPTY.** The harness, metrics (first-event error, CRPS, interval coverage, order
accuracy, censoring accuracy, missed/false event rates, trigger precision/recall, cost per
simulated day) and baseline arms (event-driven production vs periodic-scheduler ablation vs
fixed-delay vs single-cadence) are implemented and tested on synthetic smoke cases — but **no
sufficient real resolved corpus has been assembled yet, so temporal validation is incomplete
and temporal calibration is NOT claimed.** Adding a case means: collect the resolved episode,
verify every resolved timestamp postdates `as_of` (the harness enforces this), and never tune
the runtime against a locked case's outcome.
