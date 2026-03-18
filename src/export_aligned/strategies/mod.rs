//! Per-strategy label generation. Each strategy produces a `LabelingResult`
//! that is consumed by the unified export pipeline in `export_day_common()`.

pub(crate) mod opportunity;
pub(crate) mod regression;
pub(crate) mod tlob;
pub(crate) mod triple_barrier;
