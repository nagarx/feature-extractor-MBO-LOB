//! Simple example demonstrating the Sequence Builder.
//!
//! This shows how to:
//! 1. Configure the sequence builder
//! 2. Push feature snapshots
//! 3. Generate sequences for transformer input
//! 4. Access sequence metadata
//!
//! Usage:
//! ```bash
//! cargo run --example sequence_builder_example
//! ```

use feature_extractor::sequence_builder::{SequenceBuilder, SequenceConfig};

fn main() {
    println!("🔬 Sequence Builder Example\n");

    // =========================================================================
    // 1. Basic Configuration
    // =========================================================================
    println!("1️⃣ Creating sequence builder with standard TLOB configuration:");
    println!("   - Window size: 100 snapshots");
    println!("   - Stride: 1 (maximum overlap)");
    println!("   - Features: 48 (LOB features)\n");

    let config = SequenceConfig::new(100, 1) // 100 snapshots, stride 1
        .with_feature_count(48) // LOB features
        .with_max_buffer_size(1000); // Keep 1000 snapshots max

    let mut builder = SequenceBuilder::with_config(config);

    // =========================================================================
    // 2. Pushing Feature Snapshots
    // =========================================================================
    println!("2️⃣ Simulating feature extraction pipeline:");
    println!("   Pushing 150 synthetic snapshots...\n");

    for i in 0..150 {
        // Simulate feature vector (48 features)
        let features: Vec<f64> = (0..48).map(|f| (i as f64 + f as f64) / 10.0).collect();

        // Timestamp: 1 second apart
        let timestamp = i * 1_000_000_000; // i seconds in nanoseconds

        builder.push(timestamp, features);
    }

    let (buf_len, total_pushed, _, utilization) = builder.statistics();
    println!("   ✓ Buffer status:");
    println!("     - Current length: {buf_len}");
    println!("     - Total pushed: {total_pushed}");
    println!("     - Utilization: {utilization:.1}%\n");

    // =========================================================================
    // 3. Building Sequences (Online Mode)
    // =========================================================================
    println!("3️⃣ Building sequences online (as data streams in):\n");

    let mut sequence_count = 0;
    while let Some(seq) = builder.try_build_sequence() {
        sequence_count += 1;

        if sequence_count <= 3 {
            println!("   Sequence #{sequence_count}:");
            println!("     - Length: {} snapshots", seq.length);
            println!("     - Start time: {} ns", seq.start_timestamp);
            println!("     - End time: {} ns", seq.end_timestamp);
            println!("     - Duration: {:.2} seconds", seq.duration_seconds());
            println!(
                "     - Shape: {} × {}",
                seq.features.len(),
                seq.features[0].len()
            );
            println!(
                "     - Sample interval: {:.3} seconds\n",
                seq.avg_sample_interval()
            );
        }
    }

    println!("   ✓ Total sequences generated (online): {sequence_count}\n");

    // =========================================================================
    // 4. Batch Processing Mode
    // =========================================================================
    println!("4️⃣ Batch mode (generate all sequences at once):\n");

    // Reset and add more data
    builder.reset();
    for i in 0..200 {
        let features: Vec<f64> = (0..48).map(|f| (i + f) as f64).collect();
        let timestamp = i * 500_000_000; // 0.5 seconds apart
        builder.push(timestamp, features);
    }

    // Generate all sequences with stride
    let all_sequences = builder.generate_all_sequences();

    println!("   ✓ Generated {} sequences in batch", all_sequences.len());

    if let Some(first_seq) = all_sequences.first() {
        println!("\n   First sequence details:");
        println!(
            "     - Duration: {:.2} seconds",
            first_seq.duration_seconds()
        );
        println!(
            "     - Avg interval: {:.3} seconds",
            first_seq.avg_sample_interval()
        );
    }

    if let Some(last_seq) = all_sequences.last() {
        println!("\n   Last sequence details:");
        println!(
            "     - Duration: {:.2} seconds",
            last_seq.duration_seconds()
        );
        println!(
            "     - Avg interval: {:.3} seconds",
            last_seq.avg_sample_interval()
        );
    }

    // =========================================================================
    // 5. Accessing Sequence Data
    // =========================================================================
    println!("\n5️⃣ Accessing sequence data:\n");

    if let Some(seq) = all_sequences.first() {
        // As 2D array (preferred for transformers)
        println!(
            "   2D format: {} snapshots × {} features",
            seq.features.len(),
            seq.features[0].len()
        );
        println!(
            "   First snapshot first 5 features: {:?}",
            &seq.features[0][0..5]
        );

        // As flattened vector (some models prefer this)
        let flat = seq.as_flat();
        println!("\n   Flattened: {} values", flat.len());
        println!("   First 5 values: {:?}", &flat[0..5]);
    }

    // =========================================================================
    // 6. Production Configuration Examples
    // =========================================================================
    println!("\n6️⃣ Production configuration examples:\n");

    // For training: larger stride to reduce overlap
    let _training_config = SequenceConfig::new(100, 20)
        .with_feature_count(48)
        .with_max_buffer_size(2000);
    println!("   Training: window=100, stride=20 (reduces redundancy)");

    // For inference: stride 1 for maximum responsiveness
    let _inference_config = SequenceConfig::new(100, 1)
        .with_feature_count(48)
        .with_max_buffer_size(150); // Smaller buffer for inference
    println!("   Inference: window=100, stride=1 (maximum responsiveness)");

    // For long-term prediction: larger window
    let _longterm_config = SequenceConfig::new(200, 10)
        .with_feature_count(48)
        .with_max_buffer_size(3000);
    println!("   Long-term: window=200, stride=10 (more context)");

    // With MBO features (84 total)
    let _mbo_config = SequenceConfig::new(100, 1)
        .with_feature_count(84) // 48 LOB + 36 MBO
        .with_max_buffer_size(1000);
    println!("   With MBO: feature_count=84 (48 LOB + 36 MBO)");

    println!("\n✅ Example complete!");
    println!("\n💡 Key takeaways:");
    println!("   • Sequences are built from sliding windows of features");
    println!("   • Stride controls overlap (stride=1 max overlap, stride=window no overlap)");
    println!("   • Buffer is bounded to prevent memory growth");
    println!("   • Supports both online (streaming) and batch modes");
    println!("   • Zero-copy efficient operations");
}
