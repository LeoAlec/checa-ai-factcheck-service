# Implementation plan

1. Build a deterministic, hash-traceable dataset from the local FakeWhatsApp.BR CSV.
2. Normalize URLs/whitespace, deduplicate messages, and exclude cross-period groups.
3. Train word and character TF-IDF logistic-regression baselines.
4. Evaluate temporal validation/test splits with discrimination, calibration, latency,
   and confusion-matrix metrics.
5. Fine-tune BERTimbau in Colab with dynamic padding, three epochs maximum, early
   stopping, and a pinned upstream revision.
6. Compare candidates against the gates and retain models as shadow-only until an
   external contemporary dataset is available.
