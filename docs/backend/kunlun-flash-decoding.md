# Kunlun K200 Flash-Decoding

KunlunK200 0.7.3 adds TG chunk execution inside the existing FLASH_ATTN_EXT backend operator. Graph construction and the public library API are unchanged. Enable the existing FlashAttention flags and `-fa on` (test tools: K200_FLASH=1). KUNLUN_FLASH_DECODING=auto/off/on controls the path. Auto uses only measured TinyLlama-like long-context shapes; nongrouped7B keeps0.7.2. Legacy KQ/Online and narrow KQV choices remain honored by falling back.

See vendor/KunlunK200/docs/flash-decoding.md and its Japanese evaluation report for tuning, accuracy, performance limits and reproduction. The new kunlun-flash-decoding-bench target is a controlled teacher-forced A/B tool: state clear/restore and logits checksums are outside timed llama_process calls. The original smoke tool remains unchanged.
