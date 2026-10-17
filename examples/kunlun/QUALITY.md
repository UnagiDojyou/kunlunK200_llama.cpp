# Fixed-input K200 quality evaluation

`kunlun-quality-eval` is a standalone example; it adds no instrumentation to
ordinary backend builds. Tokenize once, then feed the identical ID file to
the CPU Q4_0 reference, legacy rowwise INT8 mode and candidate mode.

```bash
bin=build/perf-llama/bin
model=models/llama2-7b-q4_0.gguf
"$bin/kunlun-quality-eval" "$model" tokenize wiki.test.raw 8192 input.ids
"$bin/kunlun-quality-eval" "$model" cpu input.ids 256 cpu.bin
GGML_KUNLUN_REQUANT_I8=1 GGML_KUNLUN_CPU_KV=1 KUNLUN_GEMV=legacy KUNLUN_RMS_NORM=legacy \
  "$bin/kunlun-quality-eval" "$model" none input.ids 256 legacy.bin cpu.bin
GGML_KUNLUN_REQUANT_I8=1 GGML_KUNLUN_CPU_KV=1 \
  "$bin/kunlun-quality-eval" "$model" none input.ids 256 - cpu.bin legacy.bin
```

Repeat with context 1024. Threads are 6, batch/ubatch are 8, Flash Attention
is disabled. Each independent block contains `context` IDs; its first token
has no prediction and the remaining tokens are scored. State is cleared
between blocks. No additional BOS is inserted at block boundaries. With
8192 input IDs this scores 8160 targets at context 256 or 8184 at context
1024. These are fixed comparison scores, not a claim to reproduce a
different benchmark's canonical WikiText perplexity protocol.

The source is [Salesforce WikiText-2 raw test](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/test-00000-of-00001.parquet).
Pin revision `b08601e04326c79dfdd32d625aee71d232d685c3` and Parquet SHA256
`5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`.
Join every text row with one newline, including empty rows. The resulting
UTF-8 text SHA256 is
`aca2f46735043bcfd0a44eca981d04627b9cdf74c4c9a04bf0856d04066f58fc`.
Tokenization adds the model's BOS once. Record the ID file hash and model
hash with results.

Optional dump/reference files contain two native little-endian int32 values
(`vocabulary size`, `scored positions`) followed by FP32 vocabulary rows.
Reference comparison streams these rows and reports global logits NMSE,
mean/max KL(reference || candidate), maximum absolute difference and greedy
argmax agreement. The CPU Q4_0 baseline excludes the earlier FP16-to-Q4_0
weight approximation; the legacy reference separates existing rowwise INT8
error from a new implementation's added error.

Use `kunlun-llm-smoke` separately for PP/TG timing and free generation. Its
`K200_INPUT_IDS`, `K200_TEACHER_FILE`, `K200_CTX`, `K200_BATCH`,
`K200_UBATCH`, `K200_REPEATS`, `K200_WARMUP` and `K200_IGNORE_EOS` settings
permit fixed teacher forcing, full-context boundary tests and independent
free generation. Teacher-forced agreement does not establish free-running
long-form quality. Preserve generated token IDs and text as well as metrics.
