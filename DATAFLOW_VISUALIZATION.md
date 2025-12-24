# Feature Extractor MBO-LOB: Dataflow Visualization

> **Purpose**: Visual documentation of the feature-extractor library's dataflow and architecture using Mermaid diagrams.  
> **Version**: 2.0 (Enhanced Technical Details)

---

## 1. High-Level Pipeline Dataflow

```mermaid
flowchart TB
    subgraph Input["📁 Input Layer"]
        DBN["DBN Files<br/>.dbn.zst (zstd compressed)<br/>~7M messages/day"]
        HotStore["Hot Store<br/>Decompressed .dbn<br/>~30% faster reads"]
    end

    subgraph External["🔗 External Dependency (mbo-lob-reconstructor)"]
        DbnLoader["DbnLoader::new(path)<br/>iter_messages() → Iterator"]
        LobRecon["LobReconstructor<br/>process_message(&amp;msg)<br/>BTreeMap + AHashMap"]
    end

    subgraph Pipeline["⚙️ Pipeline Core (pipeline.rs)"]
        direction TB
        PipelineBuilder["PipelineBuilder::new()<br/>.lob_levels(10)<br/>.with_derived_features()<br/>.window(100, 10)<br/>.build()"]
        PipelineConfig["PipelineConfig {<br/>  features: FeatureConfig<br/>  sequence: SequenceConfig<br/>  sampling: SamplingConfig<br/>}"]
        PipelineProc["Pipeline.process(path)<br/>Pipeline.process_messages(iter)<br/>Pipeline.process_source(src)"]
    end

    subgraph Sampling["📊 Sampling (preprocessing/)"]
        EventSampler["EventBasedSampler<br/>sample_interval: u64<br/>event_count: u64"]
        VolumeSampler["VolumeBasedSampler<br/>target_volume: 1000 shares<br/>min_time_interval_ns: 1ms"]
        AdaptiveSampler["AdaptiveVolumeThreshold<br/>min_mult: 0.5x<br/>max_mult: 2.0x<br/>Welford volatility"]
    end

    subgraph Features["🧮 Feature Extraction (features/)"]
        direction TB
        FeatExt["FeatureExtractor<br/>extract_into(&amp;lob, &amp;mut buf)<br/>extract_arc(&amp;lob) → Arc"]
        LOBFeat["LOB Features (40)<br/>ask_p[10] ask_v[10]<br/>bid_p[10] bid_v[10]<br/>prices: i64/1e9 → f64"]
        DerivedFeat["Derived Features (8)<br/>mid, spread, imbalance<br/>microprice, impact"]
        MBOFeat["MBO Features (36)<br/>MboAggregator<br/>~8MB per symbol<br/>3 timescale windows"]
        Signals["Trading Signals (14)<br/>OfiComputer streaming<br/>TimeRegime UTC→ET<br/>Safety gates"]
    end

    subgraph Sequences["📦 Sequence Building (sequence_builder/)"]
        SeqBuilder["SequenceBuilder<br/>VecDeque circular buffer<br/>push_arc() zero-copy<br/>try_build_sequence()"]
        MultiScale["MultiScaleWindow<br/>fast: 100 snapshots<br/>medium: 1000<br/>slow: 5000"]
        Sequence["Sequence {<br/>  features: Vec Arc Vec f64<br/>  timestamps: u64 ns<br/>  length: 100<br/>}"]
    end

    subgraph Labels["🏷️ Labeling (labeling/)"]
        TLOBLabel["TlobLabelGenerator<br/>h=50, k=10, θ=0.0008<br/>smoothed past vs future"]
        DeepLOBLabel["DeepLobLabelGenerator<br/>k=h (horizon bias)<br/>vs current or past avg"]
        MultiHorizon["MultiHorizonLabelGenerator<br/>horizons: [10,20,50,100]<br/>ThresholdStrategy enum"]
    end

    subgraph Export["💾 Export (export/)"]
        NumpyExp["NumpyExporter<br/>ndarray-npy crate<br/>float32 output"]
        BatchExp["BatchExporter<br/>per-day files<br/>labels + metadata"]
        TensorFmt["TensorFormatter<br/>Flat/DeepLOB/HLOB/Image<br/>FeatureMapping indices"]
        DatasetCfg["DatasetConfig<br/>TOML/JSON config<br/>symbol-agnostic"]
    end

    subgraph Output["📤 Output Artifacts"]
        NPY["*_sequences.npy<br/>(N, 100, F) float32<br/>~16KB per sequence"]
        JSON["*_metadata.json<br/>config, stats, version"]
        Manifest["dataset_manifest.json<br/>splits, horizons, params"]
    end

    %% Flow connections with data types
    DBN -->|"zstd decompress"| DbnLoader
    HotStore -->|"direct read"| DbnLoader
    DbnLoader -->|"MboMessage struct"| LobRecon
    LobRecon -->|"LobState: [i64;20]×4"| PipelineProc

    PipelineBuilder -->|"validate()"| PipelineConfig
    PipelineConfig --> PipelineProc

    PipelineProc -->|"every msg"| EventSampler
    PipelineProc -->|"msg.size"| VolumeSampler
    VolumeSampler -.->|"volatility feedback"| AdaptiveSampler

    EventSampler -->|"should_sample()=true"| FeatExt
    VolumeSampler -->|"should_sample()=true"| FeatExt

    FeatExt --> LOBFeat
    FeatExt --> DerivedFeat
    FeatExt --> MBOFeat
    FeatExt --> Signals

    LOBFeat -->|"indices 0-39"| SeqBuilder
    DerivedFeat -->|"indices 40-47"| SeqBuilder
    MBOFeat -->|"indices 48-83"| SeqBuilder
    Signals -->|"indices 84-97"| SeqBuilder

    SeqBuilder --> Sequence
    SeqBuilder -.->|"optional"| MultiScale
    MultiScale --> Sequence

    Sequence -->|"mid_prices Vec"| TLOBLabel
    Sequence -->|"mid_prices Vec"| DeepLOBLabel
    TLOBLabel --> MultiHorizon
    DeepLOBLabel --> MultiHorizon

    Sequence --> NumpyExp
    Sequence --> BatchExp
    Sequence --> TensorFmt
    MultiHorizon -->|"TrendLabel array"| BatchExp

    DatasetCfg -->|"to_pipeline_config()"| BatchExp

    NumpyExp --> NPY
    BatchExp --> NPY
    BatchExp --> JSON
    BatchExp --> Manifest
    TensorFmt --> NPY

    style Pipeline fill:#e1f5fe
    style Features fill:#fff3e0
    style Sequences fill:#e8f5e9
    style Labels fill:#fce4ec
    style Export fill:#f3e5f5
```

---

## 2. Module Architecture

```mermaid
flowchart LR
    subgraph lib["lib.rs (Public API)"]
        prelude["prelude.rs<br/>━━━━━━━━━━━━<br/>Pipeline, PipelineBuilder<br/>FeatureExtractor<br/>SequenceBuilder, Sequence<br/>LabelGenerator, TrendLabel<br/>TensorFormatter<br/>All normalizers<br/>mbo-lob-reconstructor types"]
    end

    subgraph core["Core Modules"]
        builder["builder.rs<br/>━━━━━━━━━━━━<br/>PipelineBuilder<br/>.from_preset(Preset)<br/>.lob_levels(n)<br/>.with_derived_features()<br/>.with_mbo_features()<br/>.with_trading_signals()<br/>.window(size, stride)<br/>.volume_sampling(n)<br/>.build() → Pipeline"]
        config["config.rs<br/>━━━━━━━━━━━━<br/>PipelineConfig<br/>SamplingConfig<br/>SamplingStrategy enum<br/>AdaptiveSamplingConfig<br/>MultiScaleConfig<br/>ExperimentMetadata"]
        pipeline["pipeline.rs<br/>━━━━━━━━━━━━<br/>Pipeline struct<br/>process(path) → Output<br/>process_messages(iter)<br/>process_source(src)<br/>reset() state clear"]
        batch["batch.rs<br/>━━━━━━━━━━━━<br/>BatchProcessor<br/>BatchConfig<br/>BatchOutput<br/>DayResult<br/>CancellationToken<br/>ErrorMode enum<br/>Rayon thread pool"]
        validation["validation.rs<br/>━━━━━━━━━━━━<br/>FeatureValidator<br/>ValidationConfig<br/>ValidationResult<br/>crossed/locked quotes<br/>NaN/Inf detection"]
    end

    subgraph features["features/"]
        feat_mod["mod.rs<br/>━━━━━━━━━━━━<br/>FeatureConfig {<br/>  lob_levels: 10<br/>  tick_size: 0.01<br/>  include_derived<br/>  include_mbo<br/>  include_signals<br/>}<br/>feature_count() → usize<br/>FeatureExtractor"]
        lob["lob_features.rs<br/>━━━━━━━━━━━━<br/>extract_raw_features()<br/>extract_normalized_features()<br/>price: i64/1e9 → f64<br/>size: u32 → f64"]
        derived["derived_features.rs<br/>━━━━━━━━━━━━<br/>compute_derived_features()<br/>compute_depth_features()<br/>Returns [f64; 8]"]
        mbo["mbo_features.rs<br/>━━━━━━━━━━━━<br/>MboAggregator<br/>MboWindow (3 scales)<br/>OrderInfo tracker<br/>MboEvent conversion<br/>~8MB memory/symbol"]
        signals_mod["signals.rs<br/>━━━━━━━━━━━━<br/>OfiComputer<br/>  update(&amp;lob)<br/>  sample_and_reset()<br/>  is_warm() ≥100<br/>TimeRegime enum<br/>compute_signals()<br/>SIGNAL_COUNT = 14"]
        order_flow["order_flow.rs<br/>━━━━━━━━━━━━<br/>compute_ofi()<br/>compute_mlofi()<br/>QueueImbalance<br/>TradeFlowImbalance"]
        fi2010["fi2010.rs<br/>━━━━━━━━━━━━<br/>FI2010_FEATURE_COUNT=80<br/>Time-insensitive (20)<br/>Time-sensitive (20)<br/>Depth features (40)"]
        market_impact["market_impact.rs<br/>━━━━━━━━━━━━<br/>estimate_slippage()<br/>compute_vwap()<br/>MarketImpactEstimator"]
    end

    subgraph preprocessing["preprocessing/"]
        prep_mod["mod.rs<br/>━━━━━━━━━━━━<br/>Re-exports all"]
        sampling["sampling.rs<br/>━━━━━━━━━━━━<br/>EventBasedSampler<br/>VolumeBasedSampler<br/>should_sample() → bool<br/>reset()"]
        normalization["normalization.rs<br/>━━━━━━━━━━━━<br/>Normalizer trait<br/>ZScoreNormalizer<br/>RollingZScoreNormalizer<br/>GlobalZScoreNormalizer<br/>BilinearNormalizer<br/>PercentageChangeNormalizer<br/>MinMaxNormalizer<br/>PerFeatureNormalizer"]
        adaptive["adaptive_sampling.rs<br/>━━━━━━━━━━━━<br/>AdaptiveVolumeThreshold<br/>min_multiplier: 0.5<br/>max_multiplier: 2.0<br/>Uses VolatilityEstimator"]
        volatility["volatility.rs<br/>━━━━━━━━━━━━<br/>VolatilityEstimator<br/>Welford online algo<br/>O(1) update<br/>Numerically stable"]
    end

    subgraph sequence_builder["sequence_builder/"]
        seq_mod["mod.rs<br/>━━━━━━━━━━━━<br/>pub type FeatureVec =<br/>  Arc&lt;Vec&lt;f64&gt;&gt;<br/>8-byte clone vs 672B"]
        seq_builder["builder.rs<br/>━━━━━━━━━━━━<br/>SequenceBuilder<br/>SequenceConfig {<br/>  window_size: 100<br/>  stride: 10<br/>  max_buffer: 1000<br/>  feature_count<br/>}<br/>push_arc() zero-alloc<br/>try_build_sequence()"]
        horizon_aware["horizon_aware.rs<br/>━━━━━━━━━━━━<br/>HorizonAwareBuilder<br/>Label-aware windowing<br/>Ensures valid labels"]
        multiscale["multiscale.rs<br/>━━━━━━━━━━━━<br/>MultiScaleWindow<br/>Fast: 100 (~2s)<br/>Medium: 1000 (~20s)<br/>Slow: 5000 (~100s)<br/>push_arc() → all scales<br/>try_build_all()"]
    end

    subgraph labeling["labeling/"]
        label_mod["mod.rs<br/>━━━━━━━━━━━━<br/>LabelConfig {<br/>  horizon: usize<br/>  smoothing_window<br/>  threshold: f64<br/>}<br/>TrendLabel enum<br/>  Down=-1, Stable=0, Up=1<br/>LabelStats"]
        tlob["tlob.rs<br/>━━━━━━━━━━━━<br/>TlobLabelGenerator<br/>add_prices(&amp;[f64])<br/>generate_labels()<br/>compute_stats()<br/>Decoupled h/k params"]
        deeplob["deeplob.rs<br/>━━━━━━━━━━━━<br/>DeepLobLabelGenerator<br/>DeepLobMethod enum<br/>  VsCurrentPrice<br/>  VsPastAverage<br/>k = h (horizon bias)"]
        multi_horizon["multi_horizon.rs<br/>━━━━━━━━━━━━<br/>MultiHorizonLabelGenerator<br/>MultiHorizonConfig<br/>ThresholdStrategy {<br/>  Fixed(f64)<br/>  RollingSpread{...}<br/>  Quantile{...}<br/>}<br/>FI-2010/DeepLOB presets"]
    end

    subgraph schema["schema/"]
        schema_mod["mod.rs<br/>━━━━━━━━━━━━<br/>SCHEMA_VERSION<br/>get_feature_schema()"]
        feature_def["feature_def.rs<br/>━━━━━━━━━━━━<br/>FeatureDef struct<br/>FeatureCategory enum<br/>FeatureSchema<br/>Feature metadata"]
        presets["presets.rs<br/>━━━━━━━━━━━━<br/>Preset enum {<br/>  DeepLOB, TLOB<br/>  FI2010, TransLOB<br/>  LiT, Minimal, Full<br/>}"]
    end

    subgraph export["export/"]
        export_mod["mod.rs<br/>━━━━━━━━━━━━<br/>NumpyExporter<br/>BatchExporter<br/>Re-exports all"]
        tensor_format["tensor_format.rs<br/>━━━━━━━━━━━━<br/>TensorFormat enum {<br/>  Flat (T,F)<br/>  DeepLOB (T,4,L)<br/>  HLOB (T,L,4)<br/>  Image (T,C,H,W)<br/>}<br/>TensorFormatter<br/>FeatureMapping<br/>TensorOutput enum"]
        dataset_config["dataset_config.rs<br/>━━━━━━━━━━━━<br/>DatasetConfig<br/>SymbolConfig<br/>DataPathConfig<br/>DateRangeConfig<br/>FeatureSetConfig<br/>ExportSamplingConfig<br/>SplitConfig<br/>load_toml()/save_toml()"]
        export_aligned["export_aligned.rs<br/>━━━━━━━━━━━━<br/>AlignedBatchExporter<br/>align_sequences_with_labels()<br/>normalize_sequences()<br/>NormalizationStrategy<br/>NormalizationParams<br/>market_structure_zscore"]
    end

    lib --> core
    lib --> features
    lib --> preprocessing
    lib --> sequence_builder
    lib --> labeling
    lib --> schema
    lib --> export

    style lib fill:#bbdefb
    style core fill:#c8e6c9
    style features fill:#fff9c4
    style preprocessing fill:#ffccbc
    style sequence_builder fill:#d1c4e9
    style labeling fill:#f8bbd9
    style schema fill:#b2dfdb
    style export fill:#cfd8dc
```

---

## 3. Feature Extraction Pipeline Detail

```mermaid
flowchart TB
    subgraph input["Input State"]
        LobState["LobState (from mbo-lob-reconstructor)<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>bid_prices: [i64; 20]  // fixed-point nanodollars<br/>bid_sizes: [u32; 20]   // shares<br/>ask_prices: [i64; 20]<br/>ask_sizes: [u32; 20]<br/>best_bid: Option i64<br/>best_ask: Option i64<br/>levels: usize (≤20)<br/>timestamp: Option i64  // nanoseconds<br/>sequence: u64"]
        MboMsg["MboMessage Stream<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>order_id: u64<br/>action: Add|Modify|Cancel|Trade|Fill|Clear<br/>side: Bid|Ask|None<br/>price: i64 (nanodollars)<br/>size: u32 (shares)<br/>timestamp: Option i64"]
    end

    subgraph extractor["FeatureExtractor"]
        direction TB
        config["FeatureConfig<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>lob_levels: usize = 10<br/>tick_size: f64 = 0.01<br/>include_derived: bool = false<br/>include_mbo: bool = false<br/>mbo_window_size: usize = 1000<br/>include_signals: bool = false<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>DERIVED_FEATURE_COUNT = 8<br/>MBO_FEATURE_COUNT = 36<br/>SIGNAL_FEATURE_COUNT = 14"]
        
        extract_into["extract_into(&amp;lob, &amp;mut Vec f64)<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>• Clears and reuses buffer<br/>• Zero heap allocations<br/>• Single-pass extraction"]
        extract_arc["extract_arc(&amp;lob) → Arc Vec f64<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>• Calls extract_into internally<br/>• Wraps result in Arc<br/>• 8-byte clone cost"]
    end

    subgraph raw["Raw LOB Features (40)"]
        direction TB
        ask_prices["ask_prices [0-9]<br/>━━━━━━━━━━━━━━━━━━━━<br/>lob.ask_prices[i] / 1e9<br/>Convert i64 → f64 dollars<br/>0.0 if level empty"]
        ask_sizes["ask_sizes [10-19]<br/>━━━━━━━━━━━━━━━━━━━━<br/>lob.ask_sizes[i] as f64<br/>Raw share count"]
        bid_prices["bid_prices [20-29]<br/>━━━━━━━━━━━━━━━━━━━━<br/>lob.bid_prices[i] / 1e9<br/>Convert i64 → f64 dollars"]
        bid_sizes["bid_sizes [30-39]<br/>━━━━━━━━━━━━━━━━━━━━<br/>lob.bid_sizes[i] as f64<br/>Raw share count"]
    end

    subgraph derived["Derived Features (8)"]
        direction TB
        d0["[40] mid_price<br/>(best_bid + best_ask) / 2 / 1e9"]
        d1["[41] spread<br/>best_ask - best_bid / 1e9"]
        d2["[42] spread_bps<br/>spread / mid × 10000"]
        d3["[43] total_bid_volume<br/>Σ bid_sizes[0..L]"]
        d4["[44] total_ask_volume<br/>Σ ask_sizes[0..L]"]
        d5["[45] volume_imbalance<br/>(bid_v - ask_v) / (bid_v + ask_v)<br/>Range: [-1, 1]"]
        d6["[46] weighted_mid_price<br/>(bid×ask_v + ask×bid_v) / (bid_v + ask_v)"]
        d7["[47] price_impact<br/>|mid - weighted_mid|"]
    end

    subgraph mbo["MBO Features (36) via MboAggregator"]
        direction TB
        subgraph mbo_windows["Multi-Timescale Windows"]
            fast_w["Fast Window: 100 events (~2s)"]
            med_w["Medium Window: 1000 events (~20s)<br/>PRIMARY extraction source"]
            slow_w["Slow Window: 5000 events (~100s)"]
        end
        subgraph mbo_flow["Order Flow [48-59] (12)"]
            of1["add_rate_bid/ask<br/>cancel_rate_bid/ask<br/>trade_rate_bid/ask"]
            of2["net_order_flow<br/>net_cancel_flow<br/>net_trade_flow<br/>aggressive_order_ratio<br/>order_flow_volatility<br/>flow_regime_indicator"]
        end
        subgraph mbo_size["Size Distribution [60-67] (8)"]
            sz["size_p25/p50/p75/p90<br/>size_zscore<br/>large_order_ratio<br/>size_skewness<br/>size_concentration"]
        end
        subgraph mbo_queue["Queue &amp; Depth [68-73] (6)"]
            q["avg_queue_position<br/>queue_size_ahead<br/>orders_per_level<br/>level_concentration<br/>depth_ticks_bid/ask"]
        end
        subgraph mbo_inst["Institutional [74-77] (4)"]
            inst["large_order_frequency<br/>large_order_imbalance<br/>modification_score<br/>iceberg_proxy"]
        end
        subgraph mbo_core["Core MBO [78-83] (6)"]
            core["avg_order_age<br/>median_order_lifetime<br/>avg_fill_ratio<br/>avg_time_to_first_fill<br/>cancel_to_add_ratio<br/>active_order_count"]
        end
    end

    subgraph signals["Trading Signals (14) via OfiComputer"]
        direction TB
        subgraph ofi_comp["OfiComputer State"]
            ofi_state["prev_best_bid/ask: Option i64<br/>prev_best_bid/ask_size: u32<br/>ofi_bid/ofi_ask: i64 accumulators<br/>depth_sum/count: u64<br/>state_changes_since_reset: u64<br/>MIN_WARMUP = 100"]
        end
        subgraph sig_dir["Direction [84-86]"]
            s84["[84] true_ofi: Σ(bid_Δ - ask_Δ)<br/>Cont et al. 2014"]
            s85["[85] depth_norm_ofi: ofi/avg_depth"]
            s86["[86] executed_pressure:<br/>trade_rate_ask - trade_rate_bid"]
        end
        subgraph sig_timing["Timing [87, 93]"]
            s87["[87] signed_mp_delta_bps:<br/>(microprice - mid) / mid × 10000<br/>Stoikov 2018"]
            s93["[93] time_regime: 0-4<br/>Open|Early|Midday|Close|Closed<br/>UTC→ET with DST handling"]
        end
        subgraph sig_confirm["Confirmation [88-89]"]
            s88["[88] trade_asymmetry:<br/>(buys - sells) / total"]
            s89["[89] cancel_asymmetry:<br/>(cancel_ask - cancel_bid) / total"]
        end
        subgraph sig_impact["Impact [90-91]"]
            s90["[90] fragility_score: [0,∞)<br/>level_conc / ln(avg_depth)"]
            s91["[91] depth_asymmetry:<br/>(depth_ticks_bid - depth_ticks_ask) / total"]
        end
        subgraph sig_safety["Safety Gates [92, 94]"]
            s92["[92] book_valid: {0,1}<br/>bid>0 &amp;&amp; ask>0 &amp;&amp; bid&lt;ask"]
            s94["[94] mbo_ready: {0,1}<br/>state_changes ≥ 100"]
        end
        subgraph sig_meta["Meta [95-97]"]
            s95["[95] dt_seconds: f64<br/>Time since last sample"]
            s96["[96] invalidity_delta: f64<br/>Quote anomaly count"]
            s97["[97] schema_version: 2.1"]
        end
    end

    subgraph output["Output Buffer"]
        buffer["Vec&lt;f64&gt; output<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>40 features: Raw LOB only<br/>48 features: + Derived<br/>76 features: + MBO<br/>84 features: + Derived + MBO<br/>98 features: + All (Signals)<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Layout: GROUPED<br/>[ask_p, ask_v, bid_p, bid_v, derived, mbo, signals]"]
        arc_vec["Arc&lt;Vec&lt;f64&gt;&gt; (FeatureVec)<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Clone cost: 8 bytes (vs 672B)<br/>Shared across:<br/>• Overlapping sequences<br/>• Multi-scale windows<br/>• Parallel threads"]
    end

    LobState --> extract_into
    MboMsg -->|"process_event()"| mbo_windows
    config --> extract_into

    extract_into --> raw
    extract_into --> derived
    extract_into --> mbo
    extract_into --> signals

    raw --> buffer
    derived --> buffer
    mbo --> buffer
    signals --> buffer

    buffer --> arc_vec
    extract_arc --> arc_vec

    style extractor fill:#e3f2fd
    style raw fill:#c8e6c9
    style derived fill:#fff9c4
    style mbo fill:#ffccbc
    style signals fill:#f8bbd9
    style output fill:#e1f5fe
```

---

## 4. Sequence Building & Windowing

```mermaid
flowchart TB
    subgraph input["Feature Input"]
        features["Arc&lt;Vec&lt;f64&gt;&gt; (FeatureVec)<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>40-98 f64 values<br/>Wrapped in Arc for zero-copy"]
        timestamp["Timestamp: u64<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Nanoseconds since epoch<br/>From LobState.timestamp"]
    end

    subgraph seq_config["SequenceConfig"]
        cfg_detail["window_size: usize = 100<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Snapshots per sequence<br/>TLOB paper default"]
        stride_detail["stride: usize = 10<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Skip between sequences<br/>90% overlap with stride=10"]
        max_buf_detail["max_buffer_size: usize = 1000<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Circular buffer capacity<br/>Prevents unbounded growth"]
        feat_count_detail["feature_count: usize<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>MUST match FeatureConfig<br/>Auto-synced by PipelineConfig"]
    end

    subgraph builder["SequenceBuilder Internal State"]
        direction TB
        circular["buffer: VecDeque&lt;Snapshot&gt;<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Snapshot {<br/>  features: FeatureVec (Arc)<br/>  timestamp: u64<br/>}<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>O(1) push_back/pop_front"]
        counters["Counters<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>total_pushed: u64<br/>total_sequences: u64<br/>last_sequence_pos: usize"]
        push_arc["push_arc(ts, features)<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>1. Create Snapshot<br/>2. Push to buffer<br/>3. Evict if > max_buffer<br/>4. Increment counters"]
        try_build["try_build_sequence()<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Called after each push<br/>Returns Option Sequence<br/>STREAMING mode"]
    end

    subgraph streaming["Streaming Sequence Generation Logic"]
        direction TB
        check["Check: buffer.len() >= window_size?<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Need 100 snapshots minimum"]
        stride_check["Check: (total_pushed - last_pos) >= stride?<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Enforce stride=10 spacing"]
        build["Build Sequence<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>1. Slice last window_size items<br/>2. Clone Arc refs (cheap)<br/>3. Calculate timestamps<br/>4. Update last_sequence_pos"]
        accumulate["Accumulated Sequences<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>PipelineOutput.sequences<br/>No data loss from eviction<br/>100% sequence efficiency"]
    end

    subgraph multiscale["MultiScaleWindow (Optional)"]
        direction TB
        ms_config["MultiScaleConfig<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>fast_window: 100<br/>medium_window: 1000<br/>slow_window: 5000"]
        fast["fast_builder: SequenceBuilder<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>~2 seconds of data<br/>High-frequency patterns"]
        medium["medium_builder: SequenceBuilder<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>~20 seconds of data<br/>Medium-term trends"]
        slow["slow_builder: SequenceBuilder<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>~100 seconds of data<br/>Long-term context"]
        ms_push["push_arc(ts, features)<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Same Arc pushed to ALL scales<br/>Zero additional allocation<br/>Memory: 16 bytes (2 Arc clones)"]
    end

    subgraph output["Sequence Output Structures"]
        sequence["Sequence {<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>  features: Vec&lt;FeatureVec&gt;<br/>    // [window_size] Arc refs<br/>  start_timestamp: u64<br/>  end_timestamp: u64<br/>  duration_ns: u64<br/>  length: usize  // == window_size<br/>}<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Memory: ~848 bytes (40 feat)<br/>100 × 8-byte Arc ptrs + metadata"]
        multi_seq["MultiScaleSequence {<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>  fast: Option&lt;Sequence&gt;<br/>  medium: Option&lt;Sequence&gt;<br/>  slow: Option&lt;Sequence&gt;<br/>}<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>3 temporal resolutions<br/>Concatenate for transformer"]
    end

    features --> push_arc
    timestamp --> push_arc
    seq_config --> builder

    push_arc --> circular
    circular --> counters
    counters --> try_build
    try_build --> check
    check -->|"Yes"| stride_check
    check -->|"No"| input
    stride_check -->|"Yes"| build
    stride_check -->|"No"| input
    build --> sequence
    build --> accumulate

    features --> ms_push
    ms_push --> fast
    ms_push --> medium
    ms_push --> slow
    fast --> multi_seq
    medium --> multi_seq
    slow --> multi_seq

    style builder fill:#e8f5e9
    style streaming fill:#fff3e0
    style multiscale fill:#f3e5f5
    style output fill:#e1f5fe
```

---

## 5. Sampling Strategies

```mermaid
flowchart TB
    subgraph trigger["Sampling Trigger Inputs"]
        msg["MboMessage<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Each valid MBO event<br/>Filtered: order_id != 0<br/>           size != 0<br/>           price > 0"]
        vol["Event Volume: msg.size<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>u32 shares traded<br/>Accumulated for volume sampling"]
        ts["Timestamp: msg.timestamp<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>u64 nanoseconds<br/>For minimum interval check"]
    end

    subgraph event["EventBasedSampler"]
        event_state["State<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>sample_interval: u64 = 1000<br/>event_count: u64 = 0<br/>sample_count: u64 = 0"]
        event_logic["should_sample() Logic<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>1. event_count += 1<br/>2. if count >= interval:<br/>     count = 0<br/>     sample_count += 1<br/>     return true<br/>3. return false"]
        event_check{"event_count >= sample_interval?"}
        event_sample["Sample &amp; Reset<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>event_count = 0<br/>return true"]
        event_char["Characteristics<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>✓ Simple, deterministic<br/>✓ Predictable sample count<br/>✗ No market adaptation<br/>✗ Over-samples in quiet periods"]
    end

    subgraph volume["VolumeBasedSampler"]
        vol_state["State<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>target_volume: u64 = 1000<br/>min_time_interval_ns: u64 = 1_000_000<br/>accumulated_volume: u64 = 0<br/>last_sample_time: u64 = 0"]
        vol_logic["should_sample(size, ts) Logic<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>1. accumulated += size<br/>2. elapsed = ts - last_sample_time<br/>3. if vol >= target AND<br/>      elapsed >= min_interval:<br/>     accumulated = 0<br/>     last_sample_time = ts<br/>     return true<br/>4. return false"]
        vol_check{"volume >= target<br/>AND<br/>time >= min_interval?"}
        vol_sample["Sample &amp; Reset<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>accumulated = 0<br/>last_sample_time = ts<br/>return true"]
        vol_char["Characteristics<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>✓ Adapts to market activity<br/>✓ TLOB paper recommended<br/>✓ Better signal quality<br/>✓ Captures institutional flow<br/>✗ Variable samples/day"]
    end

    subgraph adaptive["AdaptiveVolumeThreshold"]
        adapt_state["State<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>base_threshold: u64<br/>min_multiplier: f64 = 0.5<br/>max_multiplier: f64 = 2.0<br/>volatility_estimator: VolatilityEstimator"]
        volatility["VolatilityEstimator<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Welford online algorithm<br/>O(1) per update<br/>Numerically stable<br/>No overflow risk"]
        adjust["Threshold Adjustment<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Low volatility → 0.5× threshold<br/>  (more samples in quiet market)<br/>High volatility → 2.0× threshold<br/>  (fewer samples in chaos)<br/>Smooth transition via sigmoid"]
        adapt_output["current_threshold()<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>base × multiplier<br/>Dynamically adjusted"]
    end

    subgraph output["Sample Decision"]
        should_sample["should_sample() → bool<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Pipeline calls on each message<br/>Triggers feature extraction"]
        extract["Trigger Feature Extraction<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>extractor.extract_into()<br/>Capture current LobState<br/>Record mid_price for labels"]
        stats["Typical Statistics<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Event (1000): ~10K-18K samples/day<br/>Volume (1000): ~5K-25K samples/day<br/>Adaptive: varies with volatility"]
    end

    msg --> event_logic
    msg --> vol_logic
    vol --> vol_logic
    ts --> vol_logic

    event_logic --> event_check
    event_check -->|"Yes"| event_sample
    event_check -->|"No"| trigger
    event_sample --> should_sample

    vol_logic --> vol_check
    vol_check -->|"Yes"| vol_sample
    vol_check -->|"No"| trigger
    vol_sample --> should_sample

    volatility --> adjust
    adjust --> adapt_output
    adapt_output -.->|"modifies"| vol_state

    should_sample -->|"true"| extract

    style event fill:#c8e6c9
    style volume fill:#fff9c4
    style adaptive fill:#ffccbc
    style output fill:#e1f5fe
```

---

## 6. Label Generation Flow

```mermaid
flowchart TB
    subgraph input["Input Data"]
        mid_prices["Mid Prices: Vec&lt;f64&gt;<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>From sampled LobState<br/>mid = (best_bid + best_ask) / 2 / 1e9<br/>One per sampled snapshot"]
        label_config["LabelConfig<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>horizon: usize = 50<br/>smoothing_window: usize = 10<br/>threshold: f64 = 0.0008 (8 bps)<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Presets:<br/>  hft(): h=10, k=5, θ=0.0002<br/>  short_term(): h=50, k=10, θ=0.002<br/>  medium_term(): h=100, k=20, θ=0.005"]
    end

    subgraph tlob["TLOB Labeling Method (Recommended)"]
        tlob_params["Parameters (Decoupled)<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>h = horizon (prediction distance)<br/>k = smoothing_window (independent)<br/>θ = threshold"]
        past_smooth["Past Smoothed Average<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>w⁻(t,h,k) = (1/(k+1)) × Σᵢ₌₀ᵏ p(t-i)<br/>Average of k+1 past prices<br/>Centered at current time t"]
        future_smooth["Future Smoothed Average<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>w⁺(t,h,k) = (1/(k+1)) × Σᵢ₌₀ᵏ p(t+h-i)<br/>Average of k+1 future prices<br/>Centered at time t+h"]
        pct_change["Percentage Change<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>l(t,h,k) = (w⁺ - w⁻) / w⁻<br/>Smoothed price change ratio"]
        tlob_valid["Valid Label Range<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>min_prices = k + h + k + 1<br/>Example: 10 + 50 + 10 + 1 = 71<br/>First valid: index k = 10<br/>Last valid: N - h - 1"]
    end

    subgraph deeplob["DeepLOB Labeling Method"]
        deeplob_params["Parameters (Coupled)<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>k = horizon (k = h always)<br/>θ = threshold<br/>HORIZON BIAS: longer h = more smoothing"]
        method_enum["DeepLobMethod Enum<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>VsCurrentPrice (default)<br/>VsPastAverage (FI-2010 style)"]
        future_avg["Method 1: VsCurrentPrice<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>m⁺(t) = (1/k) × Σᵢ₌₁ᵏ p(t+i)<br/>l_t = (m⁺ - p_t) / p_t<br/>Compare future avg to current"]
        vs_past["Method 2: VsPastAverage<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>m⁻(t) = (1/k) × Σᵢ₌₀ᵏ⁻¹ p(t-i)<br/>m⁺(t) = (1/k) × Σᵢ₌₁ᵏ p(t+i)<br/>l_t = (m⁺ - m⁻) / m⁻"]
        deeplob_valid["Valid Label Range<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>VsCurrentPrice: min = k+1<br/>VsPastAverage: min = 2k"]
    end

    subgraph threshold["ThresholdStrategy Enum"]
        fixed["Fixed(f64)<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Constant threshold<br/>Example: 0.0008 = 8 bps<br/>Simple, reproducible"]
        rolling_spread["RollingSpread<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>window_size: usize = 100<br/>multiplier: f64 = 0.5<br/>fallback: f64 = 0.002<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>θ = avg_spread × multiplier<br/>Adapts to volatility"]
        quantile["Quantile<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>target_proportion: f64 = 0.33<br/>window_size: usize = 1000<br/>fallback: f64 = 0.002<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Auto-balances classes<br/>~33% Up + Down"]
    end

    subgraph classification["Classification Logic"]
        classify{"Classify: l vs θ"}
        up["Up = +1<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>l > θ<br/>Price increased by > threshold"]
        stable["Stable = 0<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>-θ ≤ l ≤ θ<br/>Price within threshold"]
        down["Down = -1<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>l < -θ<br/>Price decreased by > threshold"]
    end

    subgraph multi["MultiHorizonLabelGenerator"]
        multi_config["MultiHorizonConfig<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>horizons: Vec usize<br/>smoothing_window: usize<br/>threshold_strategy: ThresholdStrategy<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Presets:<br/>  fi2010(): [10,20,30,50,100]<br/>  deeplob(): [10,20,50,100]<br/>  tlob(): [1,3,5,10,30,50]"]
        gen_all["Generate for All Horizons<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Single price buffer shared<br/>Parallel label computation<br/>One pass per horizon"]
        multi_labels["MultiHorizonLabels<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>horizons: Vec usize<br/>labels: BTreeMap horizon → Vec<br/>  (sample_idx, TrendLabel, pct_change)<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>labels_for_horizon(h) → Option<br/>summary() → MultiHorizonSummary"]
    end

    subgraph output["Output Types"]
        trend_label["TrendLabel Enum<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Down = -1<br/>Stable = 0<br/>Up = 1<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>as_int() → i8<br/>as_class_index() → usize"]
        class_idx["Class Index Conversion<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Down (-1) → 0<br/>Stable (0) → 1<br/>Up (+1) → 2<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>For PyTorch CrossEntropyLoss<br/>labels.as_class_index()"]
        label_stats["LabelStats<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>up_count: usize<br/>stable_count: usize<br/>down_count: usize<br/>total: usize<br/>up_pct, stable_pct, down_pct"]
    end

    mid_prices --> tlob
    mid_prices --> deeplob
    label_config --> tlob
    label_config --> deeplob

    tlob_params --> past_smooth
    past_smooth --> future_smooth
    future_smooth --> pct_change

    deeplob_params --> method_enum
    method_enum --> future_avg
    method_enum --> vs_past

    pct_change --> classify
    future_avg --> classify
    vs_past --> classify

    threshold --> classify
    classify -->|"l > θ"| up
    classify -->|"-θ ≤ l ≤ θ"| stable
    classify -->|"l < -θ"| down

    up --> trend_label
    stable --> trend_label
    down --> trend_label

    tlob --> multi_config
    deeplob --> multi_config
    multi_config --> gen_all
    gen_all --> multi_labels

    trend_label --> class_idx
    trend_label --> label_stats

    style tlob fill:#e8f5e9
    style deeplob fill:#fff9c4
    style multi fill:#f3e5f5
    style threshold fill:#ffccbc
    style classification fill:#e1f5fe
```

---

## 7. Export Pipeline

```mermaid
flowchart TB
    subgraph input["Pipeline Output (PipelineOutput struct)"]
        sequences["sequences: Vec&lt;Sequence&gt;<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Each: [100 × F] feature matrix<br/>F = 40-98 depending on config"]
        mid_prices_out["mid_prices: Vec&lt;f64&gt;<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>One per sampled snapshot<br/>For label generation"]
        stats["Statistics<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>messages_processed: usize<br/>features_extracted: usize<br/>sequences_generated: usize<br/>stride: usize<br/>window_size: usize"]
    end

    subgraph alignment["Sequence-Label Alignment (export_aligned.rs)"]
        align_fn["align_sequences_with_labels()<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Input: sequences, labels<br/>Output: (aligned_seqs, aligned_labels)<br/>Ensures 1:1 mapping"]
        ending_idx["Ending Index Formula<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>ending_idx = seq_idx × stride + window_size - 1<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Seq 0: 0×10 + 100 - 1 = 99<br/>Seq 1: 1×10 + 100 - 1 = 109<br/>Seq 2: 2×10 + 100 - 1 = 119"]
        match_label["Label Matching<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Label at ending_idx predicts<br/>future from sequence end<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Drop if no valid label<br/>Typical: 5-15% dropped at day end"]
    end

    subgraph normalization["Per-Day Normalization (normalize_sequences)"]
        norm_strategy["NormalizationStrategy Enum<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>None<br/>PerFeatureZScore<br/>MarketStructureZScore ← DEFAULT<br/>GlobalZScore<br/>Bilinear"]
        price_norm["Price Normalization<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>SHARED mean/std per level<br/>ask_price_L and bid_price_L<br/>use SAME statistics<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Preserves: ask > bid invariant<br/>10 means + 10 stds"]
        size_norm["Size Normalization<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>INDEPENDENT per feature<br/>Each size column has own mean/std<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>20 means + 20 stds<br/>(10 ask + 10 bid sizes)"]
        norm_params["NormalizationParams Output<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>strategy: String<br/>price_means: Vec f64 [10]<br/>price_stds: Vec f64 [10]<br/>size_means: Vec f64 [20]<br/>size_stds: Vec f64 [20]<br/>sample_count: usize<br/>levels: usize = 10<br/>feature_layout: String"]
        epsilon["Epsilon Handling<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>const EPSILON = 1e-8<br/>if variance < ε: std = 1.0<br/>Prevents division by zero"]
    end

    subgraph tensor_fmt["TensorFormatter (tensor_format.rs)"]
        format_enum["TensorFormat Enum<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Flat            → (T, F)<br/>DeepLOB{levels} → (T, 4, L)<br/>HLOB{levels}    → (T, L, 4)<br/>Image{c,h,w}    → (T, C, H, W)"]
        mapping["FeatureMapping<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>ask_price_start: 0<br/>ask_volume_start: 10<br/>bid_price_start: 20<br/>bid_volume_start: 30<br/>levels: 10<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>standard_lob(10)<br/>with_derived(10)"]
        formatter["TensorFormatter Methods<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>deeplob(levels) constructor<br/>hlob(levels) constructor<br/>format_sequence(&amp;features)<br/>format_batch(&amp;sequences)"]
        tensor_output["TensorOutput Enum<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Flat2D(Array2 f64)<br/>Channel3D(Array3 f64)<br/>Image4D(Array4 f64)<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>shape() → Vec usize<br/>as_flat() → Option"]
    end

    subgraph exporters["Exporter Classes"]
        numpy_exp["NumpyExporter<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>output_dir: PathBuf<br/>export_day(name, output)<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Creates:<br/>  features.npy<br/>  mid_prices.npy<br/>  metadata.json"]
        batch_exp["BatchExporter<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>output_dir: PathBuf<br/>label_config: Option LabelConfig<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>export_day(day_name, output)<br/>  → DayExportResult"]
        aligned_exp["AlignedBatchExporter<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Full pipeline export<br/>Alignment + Normalization<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Creates all artifacts<br/>Handles multi-horizon labels"]
    end

    subgraph config["DatasetConfig (TOML/JSON)"]
        symbol["SymbolConfig<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>name: NVDA<br/>exchange: XNAS<br/>filename_pattern:<br/>  xnas-itch-{date}.mbo.dbn.zst<br/>tick_size: 0.01"]
        dates["DateRangeConfig<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>start_date: 2025-02-03<br/>end_date: 2025-02-28<br/>exclude_weekends: true<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>get_trading_days() → Vec"]
        features_cfg["FeatureSetConfig<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>lob_levels: 10<br/>include_derived: true<br/>include_mbo: true<br/>include_signals: true<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>feature_count() → 98"]
        split["SplitConfig<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>train_ratio: 0.7<br/>val_ratio: 0.15<br/>test_ratio: 0.15<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Must sum to 1.0"]
        processing["ProcessingConfig<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>num_threads: 8<br/>error_mode: CollectErrors<br/>hot_store_dir: Option"]
    end

    subgraph output["Output File Artifacts"]
        npy_seq["*_sequences.npy<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Shape: (N, 100, F)<br/>Dtype: float32<br/>Size: N × 100 × F × 4 bytes<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>F=40: ~16 KB/sequence<br/>F=98: ~39 KB/sequence"]
        npy_labels["*_labels.npy<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Shape: (N,) or (N, H)<br/>Dtype: int8<br/>Values: {-1, 0, +1}<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>H = num horizons<br/>Multi-horizon: [10,20,50,100]"]
        json_norm["*_normalization.json<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>strategy: market_structure_zscore<br/>price_means: [120.50, ...]<br/>price_stds: [0.05, ...]<br/>size_means: [150.0, ...]<br/>size_stds: [50.0, ...]<br/>sample_count: 180000"]
        json_meta["*_metadata.json<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>config snapshot<br/>validation results<br/>processing stats<br/>normalization reference"]
        manifest["dataset_manifest.json<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>version: 1.0<br/>symbol: NVDA<br/>split: train/val/test days<br/>sequence_length: 100<br/>n_features: 40-98<br/>horizons: [10,20,50,100]<br/>labeling config"]
    end

    sequences --> align_fn
    mid_prices_out --> align_fn
    align_fn --> ending_idx
    ending_idx --> match_label

    match_label --> norm_strategy
    norm_strategy --> price_norm
    norm_strategy --> size_norm
    price_norm --> norm_params
    size_norm --> norm_params
    epsilon --> norm_params

    sequences --> tensor_fmt
    format_enum --> formatter
    mapping --> formatter
    formatter --> tensor_output

    norm_params --> numpy_exp
    norm_params --> batch_exp
    norm_params --> aligned_exp

    symbol --> aligned_exp
    dates --> aligned_exp
    features_cfg --> aligned_exp
    split --> aligned_exp
    processing --> aligned_exp

    numpy_exp --> npy_seq
    batch_exp --> npy_seq
    batch_exp --> npy_labels
    aligned_exp --> npy_seq
    aligned_exp --> npy_labels
    aligned_exp --> json_norm
    aligned_exp --> json_meta
    aligned_exp --> manifest

    style alignment fill:#e8f5e9
    style normalization fill:#fff9c4
    style tensor_fmt fill:#e1f5fe
    style config fill:#f3e5f5
    style exporters fill:#ffccbc
```

---

## 8. Parallel Batch Processing

```mermaid
flowchart TB
    subgraph config["Configuration Layer"]
        pipeline_config["PipelineConfig (Arc-wrapped)<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>features: FeatureConfig<br/>sequence: SequenceConfig<br/>sampling: SamplingConfig<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Cloned to each thread<br/>Immutable shared state"]
        batch_config["BatchConfig<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>num_threads: Option usize<br/>  None = Rayon default (num_cpus)<br/>error_mode: ErrorMode<br/>report_progress: bool<br/>stack_size: Option usize<br/>hot_store_dir: Option PathBuf"]
        error_mode["ErrorMode Enum<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>FailFast<br/>  Stop on first error<br/>  Fast feedback<br/>CollectErrors<br/>  Continue processing<br/>  Aggregate all errors"]
        cancel_token["CancellationToken<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>cancelled: Arc AtomicBool<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>new() → Token<br/>cancel() → set true<br/>is_cancelled() → bool<br/>reset() → for reuse<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Thread-safe, cheap clone"]
    end

    subgraph processor["BatchProcessor"]
        create["BatchProcessor::new(pipeline_cfg, batch_cfg)<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Auto-creates HotStoreManager<br/>if hot_store_dir is set"]
        builders["Builder Methods<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>.with_cancellation_token(token)<br/>.with_progress_callback(cb)<br/>.with_hot_store(manager)"]
        process["process_files(&amp;[paths]) → Result BatchOutput<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Creates local Rayon thread pool<br/>Parallel iteration over files<br/>Each thread: own Pipeline instance"]
        cancel_methods["Cancellation Methods<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>cancel() → signal stop<br/>is_cancelled() → check<br/>cancellation_token() → clone"]
    end

    subgraph threadpool["Rayon Thread Pool (Local)"]
        pool_config["ThreadPoolBuilder<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>num_threads: batch_config.num_threads<br/>stack_size: batch_config.stack_size<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Local pool (not global)<br/>Correct parallel execution"]
        thread1["Thread 1<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Pipeline::from_config(cfg.clone())<br/>pipeline.process(file1)<br/>No shared mutable state"]
        thread2["Thread 2<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Pipeline::from_config(cfg.clone())<br/>pipeline.process(file2)<br/>Independent execution"]
        thread3["Thread 3<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Pipeline::from_config(cfg.clone())<br/>pipeline.process(file3)<br/>Bit-identical to sequential"]
        threadN["Thread N<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Pipeline::from_config(cfg.clone())<br/>pipeline.process(fileN)<br/>No mutex contention"]
    end

    subgraph files["Input Files (par_iter)"]
        file1["day1.dbn.zst<br/>~7M messages"]
        file2["day2.dbn.zst<br/>~7M messages"]
        file3["day3.dbn.zst<br/>~7M messages"]
        fileN["dayN.dbn.zst<br/>~7M messages"]
    end

    subgraph hot_store["HotStoreManager (Optional)"]
        hs_manager["HotStoreManager<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>hot_store_dir: PathBuf<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>get_or_decompress(path)<br/>  → decompressed path<br/>~30% faster reads"]
        hs_files["Decompressed Cache<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>.dbn files (no zstd)<br/>Reused across runs<br/>Managed lifecycle"]
    end

    subgraph results["Result Aggregation"]
        day_result["DayResult (per file)<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>day: String (filename stem)<br/>file_path: String<br/>output: PipelineOutput<br/>elapsed: Duration<br/>thread_id: usize"]
        batch_output["BatchOutput (aggregated)<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>results: Vec DayResult<br/>errors: Vec FileError<br/>elapsed: Duration (total)<br/>threads_used: usize<br/>was_cancelled: bool<br/>skipped_count: usize<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>successful_count() → usize<br/>failed_count() → usize<br/>throughput_msg_per_sec() → f64"]
    end

    subgraph cancellation["Cancellation Flow"]
        check_cancel{"Check: is_cancelled()?<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Before each file<br/>O(1) atomic read"}
        skip["Skip Remaining Files<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>skipped_count += remaining<br/>was_cancelled = true"]
        continue_proc["Continue Processing<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Process current file<br/>Collect DayResult"]
        external_cancel["External Cancellation<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>From another thread:<br/>  token.cancel()<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Use case: timeout<br/>Use case: user interrupt"]
    end

    subgraph performance["Performance Characteristics"]
        perf_notes["Throughput Notes<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Sequential: ~42K msg/sec<br/>2 threads: ~64K msg/sec<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━<br/>Bottleneck: zstd decompress<br/>  (single-threaded per stream)<br/>Sub-linear scaling >2-4 threads<br/>I/O saturation limit"]
    end

    pipeline_config --> create
    batch_config --> create
    error_mode --> batch_config
    create --> builders
    cancel_token --> builders
    builders --> process

    process --> pool_config
    pool_config --> thread1
    pool_config --> thread2
    pool_config --> thread3
    pool_config --> threadN

    file1 --> thread1
    file2 --> thread2
    file3 --> thread3
    fileN --> threadN

    hot_store -.-> thread1
    hot_store -.-> thread2
    hot_store -.-> thread3
    hot_store -.-> threadN

    thread1 --> day_result
    thread2 --> day_result
    thread3 --> day_result
    threadN --> day_result

    day_result --> batch_output

    cancel_token --> check_cancel
    external_cancel --> cancel_token
    check_cancel -->|"Yes"| skip
    check_cancel -->|"No"| continue_proc
    skip --> batch_output
    continue_proc --> day_result

    style processor fill:#e8f5e9
    style threadpool fill:#fff9c4
    style cancellation fill:#ffccbc
    style results fill:#e1f5fe
    style hot_store fill:#f3e5f5
```

---

## 9. Complete Data Flow Summary

```mermaid
flowchart LR
    subgraph stage1["Stage 1: Load<br/>━━━━━━━━━━<br/>DbnLoader"]
        A1["DBN File<br/>.dbn.zst<br/>~500MB/day"]
        A2["DbnLoader::new()<br/>iter_messages()"]
        A3["MboMessage<br/>order_id, action<br/>side, price, size"]
    end

    subgraph stage2["Stage 2: Reconstruct<br/>━━━━━━━━━━<br/>LobReconstructor"]
        B1["process_message()<br/>BTreeMap bids/asks<br/>AHashMap orders"]
        B2["LobState<br/>[i64;20] prices<br/>[u32;20] sizes<br/>10 levels"]
    end

    subgraph stage3["Stage 3: Sample<br/>━━━━━━━━━━<br/>Sampler"]
        C1["should_sample()<br/>Event: every N msgs<br/>Volume: every N shares"]
        C2["Sampled LobState<br/>+ mid_price: f64<br/>+ timestamp: u64"]
    end

    subgraph stage4["Stage 4: Extract<br/>━━━━━━━━━━<br/>FeatureExtractor"]
        D1["extract_into()<br/>Zero-allocation<br/>Single-pass"]
        D2["Vec f64<br/>40-98 features<br/>GROUPED layout"]
    end

    subgraph stage5["Stage 5: Sequence<br/>━━━━━━━━━━<br/>SequenceBuilder"]
        E1["push_arc()<br/>try_build_sequence()<br/>Streaming mode"]
        E2["Sequence<br/>[100 × F]<br/>Arc-based storage"]
    end

    subgraph stage6["Stage 6: Label<br/>━━━━━━━━━━<br/>LabelGenerator"]
        F1["TLOB/DeepLOB<br/>Multi-horizon<br/>ThresholdStrategy"]
        F2["TrendLabel<br/>-1, 0, +1<br/>per horizon"]
    end

    subgraph stage7["Stage 7: Align<br/>━━━━━━━━━━<br/>export_aligned"]
        G1["align_sequences<br/>_with_labels()<br/>ending_idx formula"]
        G2["Aligned Pairs<br/>1:1 mapping<br/>Drop invalid"]
    end

    subgraph stage8["Stage 8: Normalize<br/>━━━━━━━━━━<br/>normalize_sequences"]
        H1["market_structure<br/>_zscore<br/>price: shared/level<br/>size: independent"]
        H2["Normalized<br/>mean ≈ 0<br/>std ≈ 1"]
    end

    subgraph stage9["Stage 9: Export<br/>━━━━━━━━━━<br/>Exporter"]
        I1["NumpyExporter<br/>BatchExporter<br/>TensorFormatter"]
        I2["*_sequences.npy<br/>*_labels.npy<br/>*.json metadata"]
    end

    A1 --> A2 --> A3
    A3 --> B1 --> B2
    B2 --> C1 --> C2
    C2 --> D1 --> D2
    D2 --> E1 --> E2
    B2 -->|"mid_prices"| F1
    F1 --> F2
    E2 --> G1
    F2 --> G1
    G1 --> G2
    G2 --> H1 --> H2
    H2 --> I1 --> I2

    style stage1 fill:#ffebee
    style stage2 fill:#e3f2fd
    style stage3 fill:#e8f5e9
    style stage4 fill:#fff3e0
    style stage5 fill:#f3e5f5
    style stage6 fill:#fce4ec
    style stage7 fill:#e0f7fa
    style stage8 fill:#fff9c4
    style stage9 fill:#efebe9
```

---

## 10. Feature Index Reference

```mermaid
graph LR
    subgraph raw["Raw LOB (40 features)"]
        direction TB
        R1["[0-9] ask_prices<br/>━━━━━━━━━━━━━━━━<br/>i64/1e9 → f64 dollars<br/>Best ask at index 0"]
        R2["[10-19] ask_sizes<br/>━━━━━━━━━━━━━━━━<br/>u32 → f64 shares<br/>Volume at each level"]
        R3["[20-29] bid_prices<br/>━━━━━━━━━━━━━━━━<br/>i64/1e9 → f64 dollars<br/>Best bid at index 0"]
        R4["[30-39] bid_sizes<br/>━━━━━━━━━━━━━━━━<br/>u32 → f64 shares<br/>Volume at each level"]
    end

    subgraph derived["Derived (8 features)"]
        direction TB
        D1["[40] mid_price<br/>(bid+ask)/2"]
        D2["[41] spread<br/>ask-bid dollars"]
        D3["[42] spread_bps<br/>spread/mid×10000"]
        D4["[43] total_bid_vol<br/>Σ bid_sizes"]
        D5["[44] total_ask_vol<br/>Σ ask_sizes"]
        D6["[45] vol_imbalance<br/>[-1,1] normalized"]
        D7["[46] weighted_mid<br/>microprice"]
        D8["[47] price_impact<br/>|mid-microprice|"]
    end

    subgraph mbo["MBO (36 features)"]
        direction TB
        M1["[48-59] Order Flow<br/>━━━━━━━━━━━━━━━━<br/>add/cancel/trade rates<br/>net flows, volatility"]
        M2["[60-67] Size Dist<br/>━━━━━━━━━━━━━━━━<br/>p25/p50/p75/p90<br/>zscore, skewness"]
        M3["[68-73] Queue<br/>━━━━━━━━━━━━━━━━<br/>position, depth<br/>concentration"]
        M4["[74-77] Institutional<br/>━━━━━━━━━━━━━━━━<br/>large orders<br/>iceberg proxy"]
        M5["[78-83] Core MBO<br/>━━━━━━━━━━━━━━━━<br/>order age, fill ratio<br/>cancel/add ratio"]
    end

    subgraph signals["Signals (14 features)"]
        direction TB
        S1["[84-86] Direction<br/>━━━━━━━━━━━━━━━━<br/>true_ofi (Cont 2014)<br/>depth_norm_ofi<br/>executed_pressure"]
        S2["[87] Timing<br/>━━━━━━━━━━━━━━━━<br/>signed_mp_delta_bps<br/>(Stoikov 2018)"]
        S3["[88-89] Confirm<br/>━━━━━━━━━━━━━━━━<br/>trade_asymmetry<br/>cancel_asymmetry"]
        S4["[90-91] Impact<br/>━━━━━━━━━━━━━━━━<br/>fragility_score<br/>depth_asymmetry"]
        S5["[92-94] Safety<br/>━━━━━━━━━━━━━━━━<br/>book_valid {0,1}<br/>time_regime {0-4}<br/>mbo_ready {0,1}"]
        S6["[95-97] Meta<br/>━━━━━━━━━━━━━━━━<br/>dt_seconds<br/>invalidity_delta<br/>schema_version=2.1"]
    end

    raw --> derived --> mbo --> signals

    style raw fill:#c8e6c9
    style derived fill:#fff9c4
    style mbo fill:#ffccbc
    style signals fill:#f8bbd9
```

---

## 11. Memory & Performance Characteristics

```mermaid
flowchart TB
    subgraph memory["Memory Budget"]
        lob_mem["LobState<br/>━━━━━━━━━━━━━━━━<br/>~560 bytes (stack)<br/>[i64;20]×2 = 320B<br/>[u32;20]×2 = 160B<br/>+ metadata"]
        feat_mem["Feature Vector<br/>━━━━━━━━━━━━━━━━<br/>40 feat: 320 bytes<br/>84 feat: 672 bytes<br/>98 feat: 784 bytes"]
        arc_mem["Arc Overhead<br/>━━━━━━━━━━━━━━━━<br/>8 bytes per clone<br/>vs 672B Vec clone<br/>99.99% savings"]
        seq_mem["Sequence (40 feat)<br/>━━━━━━━━━━━━━━━━<br/>~848 bytes<br/>100 × 8B Arc ptrs<br/>+ timestamps"]
        mbo_mem["MboAggregator<br/>━━━━━━━━━━━━━━━━<br/>~8 MB per symbol<br/>3 windows<br/>Order tracker"]
        buffer_mem["Sequence Buffer<br/>━━━━━━━━━━━━━━━━<br/>1000 × 848B<br/>~848 KB max"]
    end

    subgraph performance["Performance Targets"]
        hot_paths["Hot Path Operations<br/>━━━━━━━━━━━━━━━━<br/>should_sample(): ~2-3 cycles<br/>extract_raw_features(): ~50 ns<br/>push_arc(): O(1) amortized<br/>try_build_sequence(): O(window)"]
        throughput["Throughput<br/>━━━━━━━━━━━━━━━━<br/>Sequential: ~42K msg/sec<br/>Parallel 2T: ~64K msg/sec<br/>Bottleneck: zstd decompress"]
        zero_alloc["Zero-Allocation APIs<br/>━━━━━━━━━━━━━━━━<br/>extract_into(): buffer reuse<br/>push_arc(): Arc clone only<br/>No allocations in hot path"]
    end

    subgraph optimizations["Key Optimizations"]
        welford["Welford's Algorithm<br/>━━━━━━━━━━━━━━━━<br/>Online mean/variance<br/>O(1) per update<br/>Numerically stable"]
        cache["PriceLevel O(1) Cache<br/>━━━━━━━━━━━━━━━━<br/>In mbo-lob-reconstructor<br/>Eliminates O(n) sum<br/>Per-level volume cached"]
        streaming["Streaming Sequences<br/>━━━━━━━━━━━━━━━━<br/>Build during processing<br/>No buffer eviction loss<br/>100% data utilization"]
    end

    style memory fill:#e8f5e9
    style performance fill:#fff9c4
    style optimizations fill:#e1f5fe
```

---

## Legend

| Symbol | Meaning |
|--------|---------|
| 📁 | Input/Output files |
| 🔗 | External dependency |
| ⚙️ | Core processing |
| 📊 | Data sampling |
| 🧮 | Feature computation |
| 📦 | Data packaging |
| 🏷️ | Label generation |
| 💾 | Data export |
| 📤 | Final output |

---

## Quick Reference Tables

### Feature Count Configurations

| Configuration | Count | Indices |
|--------------|-------|---------|
| Raw LOB only | 40 | 0-39 |
| + Derived | 48 | 0-47 |
| + MBO | 76 | 0-39, 48-83 |
| + Derived + MBO | 84 | 0-83 |
| + All (Signals) | 98 | 0-97 |

### Key Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| LOB levels | 10 | Standard for research |
| Window size | 100 | TLOB paper |
| Stride | 10 | 90% overlap |
| Volume threshold | 1000 | Shares per sample |
| Min time interval | 1 ms | Prevents over-sampling |
| Label horizon | 50 | Steps ahead |
| Label threshold | 0.0008 | 8 basis points |

### Data Types Summary

| Stage | Type | Size |
|-------|------|------|
| DBN message | `dbn::MboMsg` | ~120 bytes |
| Internal message | `MboMessage` | 32 bytes |
| LOB state | `LobState` | ~560 bytes |
| Feature vector (98) | `Arc<Vec<f64>>` | 8 + 784 bytes |
| Sequence (100×98) | `Sequence` | ~848 bytes |

---

*Generated from codebase analysis on 2025-12-21 | Version 2.0 (Enhanced Technical Details)*
