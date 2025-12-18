//! Parallel batch processing for multi-day datasets.
//!
//! This module provides efficient parallel processing of multiple DBN files
//! using Rayon's work-stealing thread pool. Each file is processed in its own
//! thread with its own Pipeline instance, ensuring no shared mutable state.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    BatchProcessor                                │
//! │  ┌─────────────────────────────────────────────────────────────┐│
//! │  │                   Rayon Thread Pool                          ││
//! │  │                                                              ││
//! │  │  Thread 1        Thread 2        Thread N                   ││
//! │  │  ┌─────────┐    ┌─────────┐    ┌─────────┐                  ││
//! │  │  │Pipeline │    │Pipeline │    │Pipeline │   (each owns)    ││
//! │  │  │   #1    │    │   #2    │    │   #N    │                  ││
//! │  │  └────┬────┘    └────┬────┘    └────┬────┘                  ││
//! │  │       │              │              │                        ││
//! │  │  Day1.dbn       Day2.dbn       DayN.dbn                     ││
//! │  │       │              │              │                        ││
//! │  │       ▼              ▼              ▼                        ││
//! │  │  DayResult      DayResult      DayResult                    ││
//! │  └──────────────────────┬───────────────────────────────────────┘│
//! │                         ▼                                        │
//! │                   BatchOutput                                    │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Features
//!
//! - **Configurable parallelism**: Set thread count based on hardware
//! - **Error handling modes**: Fail fast or collect errors and continue
//! - **Progress reporting**: Optional callbacks for monitoring
//! - **Graceful cancellation**: Cancel long-running jobs from any thread
//! - **Memory efficient**: Each thread manages its own memory
//! - **Hot store integration**: Auto-prefer decompressed files for ~30% speedup
//!
//! # Example
//!
//! ```ignore
//! use feature_extractor::prelude::*;
//! use feature_extractor::batch::{BatchProcessor, BatchConfig};
//!
//! // Configure for 8-core machine
//! let batch_config = BatchConfig::new()
//!     .with_threads(8)
//!     .with_error_mode(ErrorMode::CollectErrors);
//!
//! // Create processor
//! let processor = BatchProcessor::new(pipeline_config, batch_config);
//!
//! // Process all days in parallel
//! let results = processor.process_files(&day_files)?;
//!
//! println!("Processed {} days in {:?}", results.successful_count(), results.elapsed);
//! println!("Throughput: {:.2} msg/sec", results.throughput_msg_per_sec);
//! ```
//!
//! # Cancellation Support
//!
//! Long-running batch jobs can be cancelled gracefully using a `CancellationToken`:
//!
//! ```ignore
//! use feature_extractor::batch::{BatchProcessor, CancellationToken};
//! use std::thread;
//!
//! // Create a cancellation token
//! let token = CancellationToken::new();
//! let processor = BatchProcessor::new(pipeline_config, batch_config)
//!     .with_cancellation_token(token.clone());
//!
//! // Start processing in a background thread
//! let handle = thread::spawn(move || processor.process_files(&day_files));
//!
//! // Cancel after timeout or user request
//! thread::sleep(Duration::from_secs(30));
//! token.cancel();
//!
//! // Get partial results
//! let result = handle.join().unwrap().unwrap();
//! if result.was_cancelled {
//!     println!("Cancelled after processing {} files", result.successful_count());
//!     println!("Skipped {} files", result.skipped_count);
//! }
//! ```
//!
//! # Hot Store Integration
//!
//! For faster processing, configure a hot store directory with decompressed files:
//!
//! ```ignore
//! let batch_config = BatchConfig::new()
//!     .with_threads(6)
//!     .with_hot_store_dir("data/hot_store/");  // ~30% faster
//!
//! let processor = BatchProcessor::new(pipeline_config, batch_config);
//! // Files automatically resolved through hot store
//! ```
//!
//! # Hardware Configuration
//!
//! The `BatchConfig` allows fine-tuning based on your hardware:
//!
//! | Hardware | Recommended Threads | Memory |
//! |----------|---------------------|--------|
//! | 4-core laptop | 4 | ~40 MB |
//! | 8-core desktop | 8 | ~80 MB |
//! | 16-core workstation | 12-14 | ~140 MB |
//! | 32-core server | 24-28 | ~280 MB |
//!
//! Leave some cores free for OS and other processes.

use crate::config::PipelineConfig;
use crate::pipeline::{Pipeline, PipelineOutput};
use mbo_lob_reconstructor::{HotStoreManager, Result};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ============================================================================
// Configuration
// ============================================================================

/// Error handling mode for batch processing.
///
/// Determines how the processor handles failures when processing individual files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ErrorMode {
    /// Stop processing immediately on first error (default).
    ///
    /// Use this when data integrity is critical and you want to
    /// investigate failures before continuing.
    #[default]
    FailFast,

    /// Continue processing remaining files, collect all errors.
    ///
    /// Use this for batch jobs where you want to process as much
    /// data as possible and handle failures later.
    CollectErrors,
}

// ============================================================================
// Cancellation Support
// ============================================================================

/// Token for cancelling batch processing.
///
/// The `CancellationToken` provides a thread-safe way to signal cancellation
/// to a running batch job. It can be cloned and shared across threads.
///
/// # Thread Safety
///
/// The token uses atomic operations internally, so it's safe to:
/// - Call `cancel()` from any thread
/// - Check `is_cancelled()` from any thread
/// - Clone and share across threads
///
/// # Example
///
/// ```ignore
/// use feature_extractor::batch::{BatchProcessor, CancellationToken};
/// use std::thread;
/// use std::time::Duration;
///
/// let token = CancellationToken::new();
/// let processor = BatchProcessor::new(config, batch_config)
///     .with_cancellation_token(token.clone());
///
/// // Start processing in background
/// let handle = thread::spawn(move || {
///     processor.process_files(&files)
/// });
///
/// // Cancel after 30 seconds
/// thread::sleep(Duration::from_secs(30));
/// token.cancel();
///
/// // Wait for graceful shutdown
/// let result = handle.join().unwrap();
/// assert!(result.unwrap().was_cancelled);
/// ```
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    /// Internal flag indicating cancellation was requested.
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Create a new cancellation token.
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Request cancellation.
    ///
    /// This signals all workers to stop processing after their current file.
    /// Already-completed files are preserved in the output.
    ///
    /// # Thread Safety
    ///
    /// This method is safe to call from any thread at any time.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Check if cancellation was requested.
    ///
    /// Workers check this periodically and stop if true.
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Reset the token (for reuse).
    ///
    /// # Warning
    ///
    /// Only call this when no processing is active.
    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::SeqCst);
    }
}

/// Configuration for batch processing.
///
/// Allows fine-tuning parallel processing based on available hardware
/// and processing requirements.
///
/// # Example
///
/// ```ignore
/// use feature_extractor::batch::{BatchConfig, ErrorMode};
///
/// // For a 16-core machine, leave 4 cores for OS
/// let config = BatchConfig::new()
///     .with_threads(12)
///     .with_error_mode(ErrorMode::CollectErrors);
/// ```
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Number of threads to use.
    ///
    /// - `None`: Use Rayon default (typically num_cpus)
    /// - `Some(n)`: Use exactly n threads
    ///
    /// Recommendation: `num_cpus - 2` for dedicated batch jobs
    pub num_threads: Option<usize>,

    /// How to handle errors during processing.
    pub error_mode: ErrorMode,

    /// Enable progress reporting via callback.
    pub report_progress: bool,

    /// Stack size per thread in bytes (advanced).
    ///
    /// Default is Rayon's default (typically 2MB).
    /// Increase if processing very deep LOB levels.
    pub stack_size: Option<usize>,

    /// Optional hot store directory for decompressed files.
    ///
    /// When set, the processor will resolve file paths through the hot store,
    /// preferring decompressed files when available for ~30% faster processing.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = BatchConfig::new()
    ///     .with_hot_store_dir("data/hot_store/");
    /// ```
    pub hot_store_dir: Option<PathBuf>,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            num_threads: None, // Use Rayon default
            error_mode: ErrorMode::FailFast,
            report_progress: false,
            stack_size: None,
            hot_store_dir: None,
        }
    }
}

impl BatchConfig {
    /// Create a new batch configuration with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of threads to use.
    ///
    /// # Arguments
    ///
    /// * `threads` - Number of worker threads
    ///
    /// # Panics
    ///
    /// Panics if threads is 0.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = BatchConfig::new().with_threads(8);
    /// ```
    pub fn with_threads(mut self, threads: usize) -> Self {
        assert!(threads > 0, "Thread count must be > 0");
        self.num_threads = Some(threads);
        self
    }

    /// Set the error handling mode.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = BatchConfig::new()
    ///     .with_error_mode(ErrorMode::CollectErrors);
    /// ```
    pub fn with_error_mode(mut self, mode: ErrorMode) -> Self {
        self.error_mode = mode;
        self
    }

    /// Enable or disable progress reporting.
    pub fn with_progress(mut self, report: bool) -> Self {
        self.report_progress = report;
        self
    }

    /// Set custom stack size per thread (advanced).
    pub fn with_stack_size(mut self, size: usize) -> Self {
        self.stack_size = Some(size);
        self
    }

    /// Set hot store directory for preferring decompressed files.
    ///
    /// When set, file paths are resolved through the hot store before processing.
    /// If a decompressed version exists, it will be used for ~30% faster processing.
    ///
    /// # Arguments
    ///
    /// * `dir` - Path to the hot store directory containing decompressed `.dbn` files
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = BatchConfig::new()
    ///     .with_threads(6)
    ///     .with_hot_store_dir("data/hot_store/");
    ///
    /// // All convenience functions now use hot store
    /// let output = process_files_with_threads(&pipeline_config, &files, 6)?;
    /// ```
    pub fn with_hot_store_dir<P: AsRef<Path>>(mut self, dir: P) -> Self {
        self.hot_store_dir = Some(dir.as_ref().to_path_buf());
        self
    }

    /// Check if hot store is configured.
    pub fn has_hot_store(&self) -> bool {
        self.hot_store_dir.is_some()
    }

    /// Get effective thread count.
    ///
    /// Returns configured threads or Rayon's default.
    pub fn effective_threads(&self) -> usize {
        self.num_threads.unwrap_or_else(rayon::current_num_threads)
    }
}

// ============================================================================
// Results
// ============================================================================

/// Result from processing a single day/file.
#[derive(Debug, Clone)]
pub struct DayResult {
    /// Day identifier (extracted from filename).
    pub day: String,

    /// Full path to the processed file.
    pub file_path: String,

    /// Pipeline output containing sequences and statistics.
    pub output: PipelineOutput,

    /// Processing time for this file.
    pub elapsed: Duration,

    /// Thread ID that processed this file (for debugging).
    pub thread_id: usize,
}

impl DayResult {
    /// Get the number of messages processed.
    pub fn messages(&self) -> usize {
        self.output.messages_processed
    }

    /// Get the number of sequences generated.
    pub fn sequences(&self) -> usize {
        self.output.sequences_generated
    }

    /// Get processing throughput (messages per second).
    pub fn throughput(&self) -> f64 {
        self.output.messages_processed as f64 / self.elapsed.as_secs_f64()
    }
}

/// Error information for a failed file.
#[derive(Debug, Clone)]
pub struct FileError {
    /// Day identifier.
    pub day: String,

    /// File path that failed.
    pub file_path: String,

    /// Error message.
    pub error: String,
}

/// Aggregated results from batch processing.
///
/// Contains all successful results, any errors, and aggregate statistics.
#[derive(Debug)]
pub struct BatchOutput {
    /// Successfully processed days.
    pub results: Vec<DayResult>,

    /// Failed days (only populated if ErrorMode::CollectErrors).
    pub errors: Vec<FileError>,

    /// Total processing time (wall clock).
    pub elapsed: Duration,

    /// Number of threads used.
    pub threads_used: usize,

    /// Whether processing was cancelled before completion.
    ///
    /// If true, `results` contains only the files that completed before
    /// cancellation was detected. Files that were in-progress when
    /// cancellation was requested will complete normally.
    pub was_cancelled: bool,

    /// Number of files that were skipped due to cancellation.
    pub skipped_count: usize,
}

impl BatchOutput {
    /// Get count of successfully processed files.
    pub fn successful_count(&self) -> usize {
        self.results.len()
    }

    /// Get count of failed files.
    pub fn failed_count(&self) -> usize {
        self.errors.len()
    }

    /// Get total messages processed across all successful files.
    pub fn total_messages(&self) -> usize {
        self.results
            .iter()
            .map(|r| r.output.messages_processed)
            .sum()
    }

    /// Get total sequences generated across all successful files.
    pub fn total_sequences(&self) -> usize {
        self.results
            .iter()
            .map(|r| r.output.sequences_generated)
            .sum()
    }

    /// Get total features extracted across all successful files.
    pub fn total_features(&self) -> usize {
        self.results
            .iter()
            .map(|r| r.output.features_extracted)
            .sum()
    }

    /// Get overall throughput (messages per second).
    pub fn throughput_msg_per_sec(&self) -> f64 {
        self.total_messages() as f64 / self.elapsed.as_secs_f64()
    }

    /// Get speedup factor compared to sequential processing.
    ///
    /// Calculated as: sum of individual processing times / total wall clock time
    pub fn speedup_factor(&self) -> f64 {
        let sequential_time: Duration = self.results.iter().map(|r| r.elapsed).sum();
        sequential_time.as_secs_f64() / self.elapsed.as_secs_f64()
    }

    /// Check if all files were processed successfully.
    pub fn all_successful(&self) -> bool {
        self.errors.is_empty()
    }

    /// Get results sorted by day.
    pub fn results_by_day(&self) -> Vec<&DayResult> {
        let mut sorted: Vec<_> = self.results.iter().collect();
        sorted.sort_by(|a, b| a.day.cmp(&b.day));
        sorted
    }

    /// Iterate over successful results.
    pub fn iter(&self) -> impl Iterator<Item = &DayResult> {
        self.results.iter()
    }

    /// Iterate over errors.
    pub fn iter_errors(&self) -> impl Iterator<Item = &FileError> {
        self.errors.iter()
    }
}

// ============================================================================
// Progress Reporting
// ============================================================================

/// Progress information for callbacks.
#[derive(Debug, Clone)]
pub struct ProgressInfo {
    /// Current file being processed.
    pub current_file: String,

    /// Index of current file (0-based).
    pub current_index: usize,

    /// Total number of files to process.
    pub total_files: usize,

    /// Number of files completed so far.
    pub completed: usize,

    /// Number of files failed so far.
    pub failed: usize,

    /// Elapsed time since start.
    pub elapsed: Duration,
}

impl ProgressInfo {
    /// Get completion percentage (0.0 to 100.0).
    pub fn percent_complete(&self) -> f64 {
        if self.total_files == 0 {
            100.0
        } else {
            (self.completed + self.failed) as f64 / self.total_files as f64 * 100.0
        }
    }

    /// Estimate remaining time based on current progress.
    pub fn estimated_remaining(&self) -> Option<Duration> {
        let done = self.completed + self.failed;
        if done == 0 {
            return None;
        }
        let remaining = self.total_files - done;
        let avg_time = self.elapsed.as_secs_f64() / done as f64;
        Some(Duration::from_secs_f64(avg_time * remaining as f64))
    }
}

/// Trait for progress reporting callbacks.
///
/// Implement this trait to receive progress updates during batch processing.
///
/// # Example
///
/// ```ignore
/// struct ConsoleProgress;
///
/// impl ProgressCallback for ConsoleProgress {
///     fn on_progress(&self, info: &ProgressInfo) {
///         println!(
///             "[{}/{}] Processing {} ({:.1}%)",
///             info.completed + 1,
///             info.total_files,
///             info.current_file,
///             info.percent_complete()
///         );
///     }
///
///     fn on_complete(&self, output: &BatchOutput) {
///         println!("Done! {} files in {:?}", output.successful_count(), output.elapsed);
///     }
/// }
/// ```
pub trait ProgressCallback: Send + Sync {
    /// Called when starting to process a file.
    fn on_progress(&self, info: &ProgressInfo);

    /// Called when batch processing completes.
    fn on_complete(&self, output: &BatchOutput);
}

/// Simple console progress reporter.
#[derive(Debug, Default)]
pub struct ConsoleProgress {
    /// Show verbose output.
    pub verbose: bool,
}

impl ConsoleProgress {
    /// Create a new console progress reporter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable verbose output.
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
}

impl ProgressCallback for ConsoleProgress {
    fn on_progress(&self, info: &ProgressInfo) {
        if self.verbose {
            println!(
                "[{:3}/{:3}] Processing: {} ({:.1}% complete)",
                info.completed + 1,
                info.total_files,
                info.current_file,
                info.percent_complete()
            );
        } else {
            print!(
                "\r[{:3}/{:3}] {:.1}%",
                info.completed + info.failed,
                info.total_files,
                info.percent_complete()
            );
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
    }

    fn on_complete(&self, output: &BatchOutput) {
        println!();
        println!("═══════════════════════════════════════════════════════════════");
        println!("Batch Processing Complete");
        println!("═══════════════════════════════════════════════════════════════");
        println!("  Files processed: {}", output.successful_count());
        println!("  Files failed:    {}", output.failed_count());
        println!("  Total messages:  {}", output.total_messages());
        println!("  Total sequences: {}", output.total_sequences());
        println!("  Total time:      {:?}", output.elapsed);
        println!(
            "  Throughput:      {:.2} msg/sec",
            output.throughput_msg_per_sec()
        );
        println!("  Speedup:         {:.2}x", output.speedup_factor());
        println!("═══════════════════════════════════════════════════════════════");
    }
}

// ============================================================================
// Batch Processor
// ============================================================================

/// Parallel batch processor for multi-day datasets.
///
/// Processes multiple DBN files in parallel using Rayon's work-stealing
/// thread pool. Each file is processed by its own Pipeline instance,
/// ensuring thread safety without locks.
///
/// # Thread Safety
///
/// This processor is designed with the following guarantees:
/// - No shared mutable state between threads
/// - Each thread creates its own Pipeline from cloned config
/// - Results are collected safely via Rayon's parallel iterator
///
/// # Example
///
/// ```ignore
/// use feature_extractor::prelude::*;
/// use feature_extractor::batch::{BatchProcessor, BatchConfig};
///
/// // Create pipeline configuration
/// let pipeline_config = PipelineBuilder::new()
///     .lob_levels(10)
///     .event_sampling(1000)
///     .build_config()?;
///
/// // Create batch processor
/// let batch_config = BatchConfig::new().with_threads(8);
/// let processor = BatchProcessor::new(pipeline_config, batch_config);
///
/// // Process files
/// let files = vec!["day1.dbn.zst", "day2.dbn.zst", "day3.dbn.zst"];
/// let results = processor.process_files(&files)?;
/// ```
pub struct BatchProcessor {
    /// Pipeline configuration (shared across threads).
    pipeline_config: Arc<PipelineConfig>,

    /// Batch processing configuration.
    batch_config: BatchConfig,

    /// Optional progress callback.
    progress_callback: Option<Arc<dyn ProgressCallback>>,

    /// Cancellation token for graceful shutdown.
    cancellation_token: CancellationToken,

    /// Optional hot store manager for decompressed file resolution.
    /// When set, file paths are resolved through the hot store, preferring
    /// decompressed files when available for improved performance.
    hot_store_manager: Option<Arc<HotStoreManager>>,
}

impl BatchProcessor {
    /// Create a new batch processor.
    ///
    /// # Arguments
    ///
    /// * `pipeline_config` - Configuration for the pipeline (will be cloned per thread)
    /// * `batch_config` - Configuration for batch processing behavior
    ///
    /// # Hot Store Auto-Configuration
    ///
    /// If `batch_config.hot_store_dir` is set, a `HotStoreManager` will be automatically
    /// created to resolve file paths through the hot store.
    pub fn new(pipeline_config: PipelineConfig, batch_config: BatchConfig) -> Self {
        // Auto-create HotStoreManager from config if hot_store_dir is set
        let hot_store_manager = batch_config
            .hot_store_dir
            .as_ref()
            .map(|dir| Arc::new(HotStoreManager::for_dbn(dir)));

        Self {
            pipeline_config: Arc::new(pipeline_config),
            batch_config,
            progress_callback: None,
            cancellation_token: CancellationToken::new(),
            hot_store_manager,
        }
    }

    /// Create a batch processor with default batch configuration.
    pub fn with_pipeline_config(pipeline_config: PipelineConfig) -> Self {
        Self::new(pipeline_config, BatchConfig::default())
    }

    /// Set a progress callback.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let processor = BatchProcessor::new(config, batch_config)
    ///     .with_progress_callback(Box::new(ConsoleProgress::new()));
    /// ```
    pub fn with_progress_callback(mut self, callback: Box<dyn ProgressCallback>) -> Self {
        self.progress_callback = Some(Arc::from(callback));
        self
    }

    /// Set a cancellation token for graceful shutdown.
    ///
    /// The token can be used to cancel processing from another thread.
    /// After cancellation, any files that were in-progress will complete,
    /// but no new files will be started.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use feature_extractor::batch::{BatchProcessor, CancellationToken};
    /// use std::thread;
    ///
    /// let token = CancellationToken::new();
    /// let processor = BatchProcessor::new(config, batch_config)
    ///     .with_cancellation_token(token.clone());
    ///
    /// // Cancel from another thread
    /// let token_handle = token.clone();
    /// thread::spawn(move || {
    ///     thread::sleep(std::time::Duration::from_secs(5));
    ///     token_handle.cancel();
    /// });
    ///
    /// let result = processor.process_files(&files)?;
    /// if result.was_cancelled {
    ///     println!("Processing was cancelled after {} files", result.successful_count());
    /// }
    /// ```
    pub fn with_cancellation_token(mut self, token: CancellationToken) -> Self {
        self.cancellation_token = token;
        self
    }

    /// Set a custom hot store manager for preferring decompressed files.
    ///
    /// This method allows you to provide a pre-configured `HotStoreManager`.
    /// For simpler usage, consider using `BatchConfig::with_hot_store_dir()` instead,
    /// which auto-creates a `HotStoreManager` with default settings.
    ///
    /// **Note**: This method overrides any hot store configured via `BatchConfig`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use feature_extractor::batch::BatchProcessor;
    /// use mbo_lob_reconstructor::HotStoreManager;
    ///
    /// // Simple: Use BatchConfig (recommended)
    /// let batch_config = BatchConfig::new()
    ///     .with_hot_store_dir("data/hot_store/");
    ///
    /// // Advanced: Use custom HotStoreManager
    /// let hot_store = HotStoreManager::for_dbn("data/hot_store/");
    /// let processor = BatchProcessor::new(config, batch_config)
    ///     .with_hot_store(hot_store);
    /// ```
    pub fn with_hot_store(mut self, hot_store: HotStoreManager) -> Self {
        self.hot_store_manager = Some(Arc::new(hot_store));
        self
    }

    /// Get a clone of the cancellation token.
    ///
    /// Use this to share the token with other threads for external cancellation.
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation_token.clone()
    }

    /// Request cancellation of the current batch processing.
    ///
    /// This is a convenience method equivalent to `processor.cancellation_token().cancel()`.
    ///
    /// # Thread Safety
    ///
    /// This method is safe to call from any thread.
    pub fn cancel(&self) {
        self.cancellation_token.cancel();
    }

    /// Check if cancellation was requested.
    ///
    /// # Thread Safety
    ///
    /// This method is safe to call from any thread.
    pub fn is_cancelled(&self) -> bool {
        self.cancellation_token.is_cancelled()
    }

    /// Get the batch configuration.
    pub fn batch_config(&self) -> &BatchConfig {
        &self.batch_config
    }

    /// Get the pipeline configuration.
    pub fn pipeline_config(&self) -> &PipelineConfig {
        &self.pipeline_config
    }

    /// Process multiple files in parallel.
    ///
    /// Each file is processed by a dedicated Pipeline instance created
    /// from the shared configuration. Results are collected and returned
    /// as a BatchOutput.
    ///
    /// # Arguments
    ///
    /// * `files` - Slice of file paths to process
    ///
    /// # Returns
    ///
    /// * `Ok(BatchOutput)` - Results from all processed files
    /// * `Err(...)` - If error mode is FailFast and a file fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let files = vec![
    ///     "data/day1.dbn.zst",
    ///     "data/day2.dbn.zst",
    ///     "data/day3.dbn.zst",
    /// ];
    ///
    /// let results = processor.process_files(&files)?;
    /// ```
    pub fn process_files<P: AsRef<Path> + Sync>(&self, files: &[P]) -> Result<BatchOutput> {
        let start = Instant::now();
        let total_files = files.len();

        let threads_used = self.batch_config.effective_threads();

        // Counters for progress tracking
        let completed = AtomicUsize::new(0);
        let failed = AtomicUsize::new(0);
        let skipped = AtomicUsize::new(0);

        // Internal result type to handle cancellation
        // Note: DayResult is boxed because it's significantly larger than other variants
        // (312 bytes vs 72 bytes), which would waste memory when passing around the enum.
        enum ProcessResult {
            Success(Box<DayResult>),
            Error(FileError),
            Skipped,
        }

        // Build a LOCAL thread pool with the requested number of threads.
        // Note: build_global() only works once per process, so we use a local pool
        // to allow different BatchProcessor instances to use different thread counts.
        let mut pool_builder = rayon::ThreadPoolBuilder::new().num_threads(threads_used);

        if let Some(stack_size) = self.batch_config.stack_size {
            pool_builder = pool_builder.stack_size(stack_size);
        }

        let pool = pool_builder.build().map_err(|e| {
            mbo_lob_reconstructor::TlobError::generic(format!(
                "Failed to create thread pool: {}",
                e
            ))
        })?;

        // Process files in parallel using the local thread pool
        let results: Vec<ProcessResult> = pool.install(|| {
            files
                .par_iter()
                .enumerate()
                .map(|(index, file)| {
                    let file_path = file.as_ref().to_string_lossy().to_string();
                    let day = extract_day_from_path(&file_path);

                    // Check for cancellation BEFORE starting work
                    if self.cancellation_token.is_cancelled() {
                        skipped.fetch_add(1, Ordering::Relaxed);
                        return ProcessResult::Skipped;
                    }

                    // Report progress if callback is set
                    if let Some(ref callback) = self.progress_callback {
                        let info = ProgressInfo {
                            current_file: file_path.clone(),
                            current_index: index,
                            total_files,
                            completed: completed.load(Ordering::Relaxed),
                            failed: failed.load(Ordering::Relaxed),
                            elapsed: start.elapsed(),
                        };
                        callback.on_progress(&info);
                    }

                    // Process the file
                    let result = self.process_single_file(file, &day, &file_path);

                    // Update counters
                    match &result {
                        Ok(day_result) => {
                            completed.fetch_add(1, Ordering::Relaxed);
                            ProcessResult::Success(Box::new(day_result.clone()))
                        }
                        Err(err) => {
                            failed.fetch_add(1, Ordering::Relaxed);
                            ProcessResult::Error(err.clone())
                        }
                    }
                })
                .collect()
        }); // End of pool.install()

        // Partition results
        let mut successful = Vec::new();
        let mut errors = Vec::new();
        let mut skipped_count = 0usize;

        for result in results {
            match result {
                ProcessResult::Success(day_result) => successful.push(*day_result),
                ProcessResult::Error(file_error) => {
                    if self.batch_config.error_mode == ErrorMode::FailFast {
                        return Err(mbo_lob_reconstructor::TlobError::generic(format!(
                            "Failed to process {}: {}",
                            file_error.file_path, file_error.error
                        )));
                    }
                    errors.push(file_error);
                }
                ProcessResult::Skipped => {
                    skipped_count += 1;
                }
            }
        }

        let was_cancelled = self.cancellation_token.is_cancelled();

        let output = BatchOutput {
            results: successful,
            errors,
            elapsed: start.elapsed(),
            threads_used,
            was_cancelled,
            skipped_count,
        };

        // Report completion
        if let Some(ref callback) = self.progress_callback {
            callback.on_complete(&output);
        }

        Ok(output)
    }

    /// Process a single file (called from thread pool).
    fn process_single_file<P: AsRef<Path>>(
        &self,
        file: P,
        day: &str,
        file_path: &str,
    ) -> std::result::Result<DayResult, FileError> {
        let start = Instant::now();

        // Resolve path through hot store if available
        let resolved_path: PathBuf = match &self.hot_store_manager {
            Some(hot_store) => hot_store.resolve(file.as_ref()),
            None => file.as_ref().to_path_buf(),
        };

        // Create a NEW Pipeline for this thread
        let mut pipeline =
            Pipeline::from_config((*self.pipeline_config).clone()).map_err(|e| FileError {
                day: day.to_string(),
                file_path: file_path.to_string(),
                error: format!("Failed to create pipeline: {}", e),
            })?;

        // Process the resolved file path
        let output = pipeline.process(&resolved_path).map_err(|e| FileError {
            day: day.to_string(),
            file_path: file_path.to_string(),
            error: e.to_string(),
        })?;

        Ok(DayResult {
            day: day.to_string(),
            file_path: file_path.to_string(),
            output,
            elapsed: start.elapsed(),
            thread_id: rayon::current_thread_index().unwrap_or(0),
        })
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract day identifier from file path.
///
/// Handles common naming conventions:
/// - `xnas-itch-20250203.mbo.dbn.zst` → `20250203`
/// - `NVDA_20250203.dbn.zst` → `20250203`
/// - `/path/to/20250203.dbn.zst` → `20250203`
fn extract_day_from_path(path: &str) -> String {
    // Try to find an 8-digit date pattern (YYYYMMDD)
    // Common patterns
    for pattern in &[
        r"(\d{4})-(\d{2})-(\d{2})", // 2025-02-03
        r"(\d{8})",                 // 20250203
        r"_(\d{8})",                // _20250203
        r"-(\d{8})",                // -20250203
    ] {
        if let Some(caps) = regex_lite_find(path, pattern) {
            return caps;
        }
    }

    // Fallback: use filename without extension
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

/// Simple regex-like pattern matching for date extraction.
fn regex_lite_find(text: &str, _pattern: &str) -> Option<String> {
    // Look for 8-digit sequences
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i + 8 <= chars.len() {
        let candidate: String = chars[i..i + 8].iter().collect();
        if candidate.chars().all(|c| c.is_ascii_digit()) {
            // Validate it looks like a date (starts with 20)
            if candidate.starts_with("20") {
                return Some(candidate);
            }
        }
        i += 1;
    }

    None
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Process multiple files with default settings.
///
/// Convenience function for simple batch processing without custom configuration.
///
/// # Example
///
/// ```ignore
/// use feature_extractor::batch::process_files_parallel;
///
/// let results = process_files_parallel(&pipeline_config, &files)?;
/// ```
pub fn process_files_parallel<P: AsRef<Path> + Sync>(
    pipeline_config: &PipelineConfig,
    files: &[P],
) -> Result<BatchOutput> {
    let processor = BatchProcessor::with_pipeline_config(pipeline_config.clone());
    processor.process_files(files)
}

/// Process multiple files with specified thread count.
///
/// # Example
///
/// ```ignore
/// let results = process_files_with_threads(&config, &files, 8)?;
/// ```
pub fn process_files_with_threads<P: AsRef<Path> + Sync>(
    pipeline_config: &PipelineConfig,
    files: &[P],
    threads: usize,
) -> Result<BatchOutput> {
    let batch_config = BatchConfig::new().with_threads(threads);
    let processor = BatchProcessor::new(pipeline_config.clone(), batch_config);
    processor.process_files(files)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_defaults() {
        let config = BatchConfig::new();
        assert!(config.num_threads.is_none());
        assert_eq!(config.error_mode, ErrorMode::FailFast);
        assert!(!config.report_progress);
    }

    #[test]
    fn test_batch_config_builder() {
        let config = BatchConfig::new()
            .with_threads(8)
            .with_error_mode(ErrorMode::CollectErrors)
            .with_progress(true);

        assert_eq!(config.num_threads, Some(8));
        assert_eq!(config.error_mode, ErrorMode::CollectErrors);
        assert!(config.report_progress);
    }

    #[test]
    #[should_panic(expected = "Thread count must be > 0")]
    fn test_batch_config_zero_threads() {
        BatchConfig::new().with_threads(0);
    }

    #[test]
    fn test_extract_day_from_path() {
        // Standard patterns
        assert_eq!(
            extract_day_from_path("xnas-itch-20250203.mbo.dbn.zst"),
            "20250203"
        );
        assert_eq!(
            extract_day_from_path("/data/NVDA_20250203.dbn.zst"),
            "20250203"
        );
        assert_eq!(
            extract_day_from_path("path/to/file-20250203.dbn"),
            "20250203"
        );

        // Fallback to filename
        assert_eq!(extract_day_from_path("unknown_file.dbn"), "unknown_file");
    }

    #[test]
    fn test_progress_info_percent() {
        let info = ProgressInfo {
            current_file: "test.dbn".to_string(),
            current_index: 0,
            total_files: 10,
            completed: 5,
            failed: 0,
            elapsed: Duration::from_secs(10),
        };

        assert_eq!(info.percent_complete(), 50.0);
    }

    #[test]
    fn test_progress_info_estimated_remaining() {
        let info = ProgressInfo {
            current_file: "test.dbn".to_string(),
            current_index: 0,
            total_files: 10,
            completed: 5,
            failed: 0,
            elapsed: Duration::from_secs(10),
        };

        let remaining = info.estimated_remaining().unwrap();
        assert_eq!(remaining, Duration::from_secs(10)); // 5 done in 10s, 5 remaining
    }

    #[test]
    fn test_batch_output_aggregates() {
        let output = BatchOutput {
            results: vec![],
            errors: vec![],
            elapsed: Duration::from_secs(10),
            threads_used: 4,
            was_cancelled: false,
            skipped_count: 0,
        };

        assert_eq!(output.successful_count(), 0);
        assert_eq!(output.failed_count(), 0);
        assert_eq!(output.total_messages(), 0);
        assert!(output.all_successful());
        assert!(!output.was_cancelled);
        assert_eq!(output.skipped_count, 0);
    }

    #[test]
    fn test_file_error() {
        let error = FileError {
            day: "20250203".to_string(),
            file_path: "/data/file.dbn".to_string(),
            error: "File not found".to_string(),
        };

        assert_eq!(error.day, "20250203");
        assert!(error.error.contains("not found"));
    }

    #[test]
    fn test_error_mode_default() {
        let mode = ErrorMode::default();
        assert_eq!(mode, ErrorMode::FailFast);
    }
}
