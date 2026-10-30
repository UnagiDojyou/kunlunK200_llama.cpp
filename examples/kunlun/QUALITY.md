# Small fixed-token quality evaluation

The inference library remains 0.7.12. SwiGLU and Large PP remain opt-in; the four bitwise Q4 optimizations keep their 0.7.12 defaults. See docs/backend/kunlun-quality-20261030.md for PPL, task results, limitations, and the independent decisions.

The existing kunlun-quality-eval tool scores the same teacher tokens at every mode. The CLI's ctx argument defines an independent scoring block, not a mandatory allocation size: 513 input IDs yield 512 scored transitions. BOS is conditioning input and is not itself scored. Clear KV separates blocks; no transition across blocks is scored. Every input row requests logits, including rows not selected for detailed distribution metrics.

Optional diagnostic-only settings:

| Setting | Unset | Meaning |
| --- | --- | --- |
| K200_QUALITY_BATCH | 8 | Logical process batch, 1..document block size |
| K200_QUALITY_UBATCH | batch | Physical ubatch, 1..batch |
| K200_QUALITY_CONTEXT | CLI ctx | Allocated runtime context, at least the scoring block |
| K200_QUALITY_DOCS | none | Whitespace-separated document ID/category manifest, one entry per block |
| K200_QUALITY_SAMPLE_STRIDE | 1 | Only sample logits/reference files; all tokens still contribute to NLL |

With stride>1, selected document positions are multiples of stride plus 127,128,255,256,257,511 when in bounds. Stride32/block513 gives20 rows. A versioned sampled-reference header records magic 0x4b515331, version1, vocabulary size, total scored transitions, block size and stride. References must use identical IDs, labels, boundaries, context, batch/ubatch and sample schedule. Stride1 preserves the old two-integer header and full-reference behavior. The corpus context field retains its legacy meaning of scoring block length; actual allocation and ubatch are in run metadata/settings.

The tool reports per-document and global mean NLL/PPL, sampled NMSE/KL, Top1/Top5 overlap, reference Top2 gap, correct-label log-probability changes and a worst-KL example. A gap<=0.1 and KL>0.1 are descriptive diagnostic thresholds, not quality acceptance standards. Do not average NMSE from different reference energies or treat boundary-enriched samples as all-token estimates.

kunlun-llm-smoke also accepts K200_TASK_MANIFEST: one ID and prompt-file path per whitespace-separated line. It loads the model/context once, clears KV before each trial, and emits a JSON K200TASK answer with an alphanumeric/hyphen/underscore ID. It uses the original greedy argmax behavior, generation cap and EOS policy; no hidden chat template is added. It rejects mixing the manifest with input-ID/PP-length/teacher inputs. K200_NO_DUMP=1 avoids raw logits. Ordinary single-prompt invocation is unchanged.

Use the existing guarded runner, fixed models and fixed executables. The tokenizer-only path uses vocab_only and records a vocabulary fingerprint/BOS/EOS for fixture compatibility. Sampled references are temporary comparison data and can be deleted after aggregates have been checked. Keep quality diagnostics separate from ordinary performance measurements.
