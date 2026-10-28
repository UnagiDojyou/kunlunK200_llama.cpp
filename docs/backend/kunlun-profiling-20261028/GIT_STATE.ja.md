# Git状態

|用途|commit / branch|
|---|---|
|固定library 0.7.6|`cbe2be1ae5324edb714472538472f4089e6d9c51`|
|通常llama.cpp|`1e6191a47b24813396238caad9294d570f6f0446`|
|実際にcompileした計測用GGML source|`a4f96e61b6906f71416bec1d8daebd46447f9fe3`|
|standalone observerと解析tools|`3601155397edfcfb7b6a68830271bdc05c4c8456`|
|開発branch|`codex/k200-profile-20261028`|
|全てのparent submodule pin|`vendor/KunlunK200` → `cbe2be1ae5324edb714472538472f4089e6d9c51`|

この文書を含むレポートcommitのhashは、commit後に生成する成果物の
`delivery-summary.json`へ記録する。Git内の文書に自己参照hashは埋め込まない。
source observerのcompileと実機試験はa4f96e61の時点で行い、後続はstandalone
計測器・オフライン解析・文書・集計図表。library/shader/defaultは無変更。

publication用2 worktreeは開始時HEADのままcleanで、既存のdirty実験worktreeも
変更していない。今回のbranchはローカルcommitのみ。push/PR/main/master更新なし。
