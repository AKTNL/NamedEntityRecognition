#!/usr/bin/env bash
# =============================================================================
# GPT-NER 示例检索策略消融实验
# =============================================================================
# 复现论文消融：random < sentence-level < entity-level
#
# 用法：
#   bash generative/ablation.sh            # 默认前 200 句
#   N=500 bash generative/ablation.sh      # 前 500 句
#
# 说明：
#   - N 为**原句数**，每句仍展开全部 10 个类型，因此实际查询数 = N × 10
#   - 结果写入 results/generative/，可用 summarize.py 汇总
# =============================================================================
set -u

N="${N:-200}"
PY="${PY:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"

for STRATEGY in random sentence entity; do
  echo "=================================================================="
  echo "  消融：strategy=${STRATEGY}  N=${N}  k=8  neg_ratio=0.5"
  echo "=================================================================="
  "$PY" -u generative/run_gptner.py \
      --strategy "$STRATEGY" \
      --k 8 \
      --neg-ratio 0.5 \
      --limit "$N" \
      --batch-size 32 \
      2>&1 | tee "logs/ablation_${STRATEGY}_n${N}.log"
done

echo ""
echo "全部消融完成，汇总如下："
"$PY" generative/summarize.py
