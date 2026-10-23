# -*- coding: utf-8 -*-
"""
GPT-NER 链路 · 步骤 3：少样本示例检索（Few-shot Demonstration Retrieval）
=========================================================================
论文指出示例检索策略是 GPT-NER 三大机制之一，且消融实验显示差异极大
（随机 72.62 → 句子级 84.36 → 实体级 89.97）。本模块完整复现三档策略：

    random          随机抽取同类型训练样本
    sentence-level  句子级 kNN：用整句向量检索语义最接近的训练句
    entity-level    实体级 kNN：用**实体首字**的上下文表示检索实体最相似的训练样本
                    （对应论文 §3.3 与官方 `offline_comput_feature.py`）

★ 正负示例混合（本项目最重要的一个修正）
----------------------------------------
官方 `get_results_mrc_knn.py` 里有这样两行被注释掉的代码：

    # prompt += get_knn(..., nums=example_false, label_name=..., positive_idx=0)
    # prompt += get_knn(..., nums=example_true,  label_name=..., positive_idx=1)

即作者预留了**同时检索正例（含目标类型）与负例（不含目标类型）**的接口。
这一点对 CLUENER 至关重要：dev 侧每句展开全部 10 个类型后，
**83.5% 的查询是负例**（该句不含此类型）。若示例清一色是"含实体的正例"，
模型会误以为"必须标出一个"，于是在负例上疯狂过预测，精确率崩塌。

因此本模块默认 `neg_ratio=0.5`：一半示例演示"没有此类实体就原样输出"。

英文原实现的两处中文改造
------------------------
1. **句向量**：官方用英文 SimCSE `sup-simcse-roberta-large`。本项目改用中文
   句向量模型 GTE（`iic/nlp_gte_sentence-embedding_chinese-base`），
   若加载失败则回退到 `bert-base-chinese` 的 mean-pooling。
2. **实体表示**：官方按**空格分词**取"实体首词"的隐状态。中文无空格，改为
   **取实体首字的上下文隐状态**（bert-base-chinese 逐字输入）。
   测试侧对应地取**每一个字**的隐状态作为候选实体表示 —— 这与论文
   "对测试句所有词位都做候选检索" 的设定一致，只是中文下词位 = 字位。

输出
----
`retrieve()` 返回与测试查询列表**等长**的示例列表，元素为
`{"input": 原句, "output": 标注句}`，可直接喂给 `build_extraction_prompt()`。

作者：《自然语言处理》课程综合实践 · 生成式 NER 链路
"""

import os
import sys
import json
import random
from typing import List, Dict, Any, Optional, Tuple, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gptner_format import CATEGORIES, mark_entities  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# 路径与默认配置
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, ".data", "gptner")
CACHE_DIR = os.path.join(OUT_DIR, "retrieval_cache")

DEFAULT_SENTENCE_MODEL = os.path.join(
    PROJECT_ROOT, "models", "nlp_gte_sentence-embedding_chinese-base"
)
FALLBACK_SENTENCE_MODEL = "bert-base-chinese"
DEFAULT_CONTEXT_MODEL = "bert-base-chinese"   # 用于实体级：逐字上下文隐状态


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def resolve_local_model(name_or_path: str) -> str:
    """把 HF 模型名解析为**本地目录**，全程不联网。

    国内网络常无法直连 huggingface.co，而权重其实已在本地缓存中。本函数直接到
    `~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<rev>` 取目录，
    使 `from_pretrained` 走本地路径，从而绕开 hub 的联网校验。
    """
    if os.path.isdir(name_or_path):
        return name_or_path

    cache_home = os.environ.get(
        "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    )
    snap_root = os.path.join(
        cache_home, "hub", "models--" + name_or_path.replace("/", "--"), "snapshots"
    )
    if os.path.isdir(snap_root):
        revs = sorted(os.listdir(snap_root))
        if revs:
            return os.path.join(snap_root, revs[-1])
    return name_or_path


def _clean_context(text: str) -> bool:
    """示例池准入检查：原文不得含 @@ / ##，否则会破坏标记协议。"""
    return "@@" not in text and "##" not in text


def _l2norm(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


# ---------------------------------------------------------------------------
# 编码器
# ---------------------------------------------------------------------------

class SentenceEncoder:
    """中文句向量编码器：优先 GTE，失败则回退 BERT mean-pooling。"""

    def __init__(self, model_path: str = DEFAULT_SENTENCE_MODEL, device: Optional[str] = None):
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            local = resolve_local_model(model_path)
            self.tok = AutoTokenizer.from_pretrained(local)
            self.model = AutoModel.from_pretrained(local)
            self.mode = "gte"
            self.name = os.path.basename(local)
        except Exception as e:  # noqa: BLE001
            print(f"  [回退] GTE 加载失败（{e}），改用 {FALLBACK_SENTENCE_MODEL} mean-pooling")
            local = resolve_local_model(FALLBACK_SENTENCE_MODEL)
            self.tok = AutoTokenizer.from_pretrained(local)
            self.model = AutoModel.from_pretrained(local)
            self.mode = "mean"
            self.name = FALLBACK_SENTENCE_MODEL

        self.model.to(self.device).eval()
        self.hidden = getattr(self.model.config, "hidden_size", 768)

    @staticmethod
    def _mean_pooling(last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def encode(self, texts: List[str], batch_size: int = 64, max_len: int = 128) -> np.ndarray:
        """编码为句向量，返回 (n, hidden) 的 float32 数组（未归一化）。"""
        torch = self.torch
        out: List[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                enc = self.tok(chunk, padding=True, truncation=True,
                               max_length=max_len, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                hs = self.model(**enc).last_hidden_state
                # GTE 的 config 中 pooler_type = first_token_transform，即取 [CLS]
                vec = hs[:, 0] if self.mode == "gte" else self._mean_pooling(hs, enc["attention_mask"])
                out.append(vec.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.hidden), dtype=np.float32)


class ContextCharEncoder:
    """逐字上下文编码器：返回每个句子每个**字**的隐状态（用于实体级检索）。"""

    def __init__(self, model_name: str = DEFAULT_CONTEXT_MODEL, device: Optional[str] = None):
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        local = resolve_local_model(model_name)
        self.tok = AutoTokenizer.from_pretrained(local)
        self.model = AutoModel.from_pretrained(local).to(self.device).eval()

    def encode_chars(self, texts: List[str], batch_size: int = 32,
                     max_len: int = 128) -> List[np.ndarray]:
        """返回 List[np.ndarray]，第 j 个数组 shape=(len(text_j), hidden)。"""
        torch = self.torch
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))

        with torch.no_grad():
            for s in range(0, len(order), batch_size):
                ids = order[s:s + batch_size]
                chunk = [texts[i] for i in ids]
                enc = self.tok(chunk, padding=True, truncation=True,
                               max_length=max_len, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                hs = self.model(**enc).last_hidden_state.float().cpu().numpy()
                for local, gi in enumerate(ids):
                    n = len(chunk[local])
                    # token 1..n 对应原句 0..n-1 个字
                    results[gi] = hs[local, 1:1 + n]
        return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 检索器
# ---------------------------------------------------------------------------

class ExampleRetriever:
    """三档示例检索策略 × 正负示例混合的统一封装。

    参数:
        strategy:  "random" | "sentence" | "entity"
        k:         示例总条数
        neg_ratio: 示例中"不含目标类型"的负例占比（默认 0.5）
        seed:      随机策略种子
    """

    STRATEGIES = ("random", "sentence", "entity")

    def __init__(
        self,
        strategy: str = "sentence",
        k: int = 8,
        neg_ratio: float = 0.5,
        seed: int = 42,
        entity_topm: int = 8,
        device: Optional[str] = None,
        cache_dir: str = CACHE_DIR,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy 必须是 {self.STRATEGIES} 之一，收到 {strategy!r}")
        self.strategy = strategy
        self.k = k
        self.neg_ratio = neg_ratio
        self.seed = seed
        # 实体级检索中，每个候选字位保留的近邻数（用于保证正例多样性）
        self.entity_topm = entity_topm
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self._sent_encoder: Optional[SentenceEncoder] = None
        self._char_encoder: Optional[ContextCharEncoder] = None

        # 由 fit() 填充
        self.sent_index: List[Dict[str, Any]] = []
        self.pos_by_type: Dict[str, List[int]] = {c: [] for c in CATEGORIES}
        self.neg_by_type: Dict[str, List[int]] = {c: [] for c in CATEGORIES}

    # ---------------- 惰性加载 ----------------

    @property
    def sent_encoder(self) -> SentenceEncoder:
        if self._sent_encoder is None:
            self._sent_encoder = SentenceEncoder(device=self.device)
        return self._sent_encoder

    @property
    def char_encoder(self) -> ContextCharEncoder:
        if self._char_encoder is None:
            self._char_encoder = ContextCharEncoder(device=self.device)
        return self._char_encoder

    # ---------------- 语料准备 ----------------

    def fit(self, sentence_index: List[Dict[str, Any]]) -> "ExampleRetriever":
        """按类型把训练语料切成正例池 / 负例池。

        参数:
            sentence_index: `sentence-index.train.json`，每项 {sample_id, text, spans}
        """
        self.sent_index = [s for s in sentence_index if _clean_context(s["text"])]
        for i, s in enumerate(self.sent_index):
            present: Set[str] = {sp["type"] for sp in s["spans"]}
            for c in CATEGORIES:
                (self.pos_by_type[c] if c in present else self.neg_by_type[c]).append(i)
        n_pos = sum(len(v) for v in self.pos_by_type.values())
        n_neg = sum(len(v) for v in self.neg_by_type.values())
        print(f"  示例池：{len(self.sent_index)} 句 | 正例槽位 {n_pos} | 负例槽位 {n_neg}")
        return self

    def _demo(self, global_i: int, entity_type: str) -> Dict[str, str]:
        """把一句训练语料渲染成 (input, output) 示例。"""
        s = self.sent_index[global_i]
        spans = [(sp["start"], sp["end"]) for sp in s["spans"] if sp["type"] == entity_type]
        return {"input": s["text"], "output": mark_entities(s["text"], spans)}

    # ---------------- 索引 ----------------

    def _sentence_vectors(self, tag: str) -> np.ndarray:
        """全部训练句的句向量（按 self.sent_index 顺序），带磁盘缓存。"""
        path = os.path.join(self.cache_dir, f"sentvec_{tag}.npy")
        if os.path.exists(path):
            v = np.load(path)
            if v.shape[0] == len(self.sent_index):
                return v
        texts = [s["text"] for s in self.sent_index]
        print(f"  [句向量] 编码 {len(texts)} 条训练句 ...")
        v = self.sent_encoder.encode(texts)
        np.save(path, v)
        return v

    def _entity_index(self, tag: str) -> Dict[str, Dict[str, Any]]:
        """实体级索引：每类存"实体首字"表示 + 归属句子下标。"""
        path = os.path.join(self.cache_dir, f"entity_{tag}.npz")
        if os.path.exists(path):
            z = np.load(path, allow_pickle=True)
            return {c: {"vec": z[f"{c}_vec"], "global_idx": z[f"{c}_gidx"]}
                    for c in CATEGORIES}

        print(f"  [实体级索引] 编码 {len(self.sent_index)} 条训练句（实体首字表示）...")
        char_vecs = self.char_encoder.encode_chars([s["text"] for s in self.sent_index])

        index: Dict[str, Dict[str, Any]] = {}
        for c in CATEGORIES:
            vecs, gidx = [], []
            for gi in self.pos_by_type[c]:
                cv = char_vecs[gi]
                for sp in self.sent_index[gi]["spans"]:
                    if sp["type"] != c or sp["start"] >= len(cv):
                        continue
                    vecs.append(cv[sp["start"]])
                    gidx.append(gi)
            mat = np.stack(vecs) if vecs else np.zeros((0, char_vecs[0].shape[-1]), dtype=np.float32)
            index[c] = {"vec": _l2norm(mat.astype(np.float32)),
                        "global_idx": np.array(gidx, dtype=np.int64)}
        np.savez_compressed(
            path,
            **{f"{c}_vec": index[c]["vec"] for c in CATEGORIES},
            **{f"{c}_gidx": index[c]["global_idx"] for c in CATEGORIES},
        )
        return index

    # ---------------- 检索原语 ----------------

    @staticmethod
    def _knn(query: np.ndarray, base: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """余弦相似度 top-k。query/base 均为 (n, hidden)，内部自行归一化。"""
        if base.shape[0] == 0 or k <= 0:
            return (np.zeros((query.shape[0], 0), dtype=np.int64),
                    np.zeros((query.shape[0], 0), dtype=np.float32))
        q = _l2norm(query.astype(np.float32))
        sims = q @ _l2norm(base.astype(np.float32)).T
        kk = min(k, base.shape[0])
        part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        rows = np.arange(sims.shape[0])[:, None]
        ps = sims[rows, part]
        order = np.argsort(-ps, axis=1)
        return part[rows, order], ps[rows, order]

    def _split_k(self, k: int) -> Tuple[int, int]:
        """把 k 拆成正例数 / 负例数。"""
        k_neg = int(round(k * self.neg_ratio))
        k_neg = max(0, min(k_neg, k - 1))   # 至少留 1 条正例，否则模型学不到标记行为
        return k - k_neg, k_neg

    @staticmethod
    def _interleave(pos: List[Any], neg: List[Any]) -> List[Any]:
        """交替排列正反示例，避免同类示例扎堆（模型会学到位置偏置）。"""
        out: List[Any] = []
        i = j = 0
        while i < len(pos) or j < len(neg):
            if i < len(pos):
                out.append(pos[i]); i += 1
            if j < len(neg):
                out.append(neg[j]); j += 1
        return out

    # ---------------- 主入口 ----------------

    def retrieve(
        self,
        test_samples: List[Dict[str, Any]],
        tag: str = "default",
        k: Optional[int] = None,
    ) -> List[List[Dict[str, str]]]:
        """为每条 per-type 查询返回 k 条示例（已渲染为 input/output 对）。"""
        if not self.sent_index:
            raise RuntimeError("请先调用 fit(sentence_index) 装载训练语料")
        k = k or self.k
        k_pos, k_neg = self._split_k(k)
        rng = random.Random(self.seed)

        # ---------- random ----------
        if self.strategy == "random":
            out: List[List[Dict[str, str]]] = []
            for s in test_samples:
                c = s["entity_label"]
                p = self.pos_by_type[c][:]; n = self.neg_by_type[c][:]
                rng.shuffle(p); rng.shuffle(n)
                out.append(self._interleave([self._demo(i, c) for i in p[:k_pos]],
                                            [self._demo(i, c) for i in n[:k_neg]]))
            return out

        # ---------- sentence ----------
        if self.strategy == "sentence":
            base = self._sentence_vectors(tag)
            unique = sorted({s["context"] for s in test_samples})
            print(f"  [句子级检索] 编码 {len(unique)} 条测试句 ...")
            qv = self.sent_encoder.encode(unique)
            qmap = dict(zip(unique, qv))

            res: List[List[Dict[str, str]]] = []
            for s in test_samples:
                c = s["entity_label"]
                q = qmap[s["context"]][None, :]
                pi, _ = self._knn(q, base[self.pos_by_type[c]], k_pos)
                ni, _ = self._knn(q, base[self.neg_by_type[c]], k_neg)
                pos = [self._demo(self.pos_by_type[c][int(i)], c) for i in pi[0]]
                neg = [self._demo(self.neg_by_type[c][int(i)], c) for i in ni[0]]
                res.append(self._interleave(pos, neg))
            return res

        # ---------- entity ----------
        eidx = self._entity_index(tag)
        base = self._sentence_vectors(tag)
        unique = sorted({s["context"] for s in test_samples})
        print(f"  [实体级检索] 编码 {len(unique)} 条测试句（全字位候选）...")
        cvs = self.char_encoder.encode_chars(unique)
        cmap = dict(zip(unique, cvs))

        res_e: List[List[Dict[str, str]]] = []
        for s in test_samples:
            c = s["entity_label"]
            cv = cmap[s["context"]]
            gidx = eidx[c]["global_idx"]

            if cv.shape[0] == 0 or gidx.shape[0] == 0:
                pos = [self._demo(i, c) for i in self.pos_by_type[c][:k_pos]]
            else:
                # 每个字位都是一个候选实体起点；对每个字位取 TOPM 个近邻，
                # 再按相似度全局排序去重。
                # ⚠ 若每个字位只取 top-1，相邻字位几乎必然命中同一个训练实体，
                #   去重后只剩 1 条示例，提示会退化成"几乎没有正例"，F1 直接归零。
                #   这是初版实现的真实 bug，勿改回。
                top_idx, top_sim = self._knn(cv, eidx[c]["vec"], self.entity_topm)
                best_score: Dict[int, float] = {}
                for i in range(top_idx.shape[0]):
                    for j in range(top_idx.shape[1]):
                        gi = int(gidx[int(top_idx[i, j])])
                        sc = float(top_sim[i, j])
                        if gi not in best_score or sc > best_score[gi]:
                            best_score[gi] = sc
                ranked = sorted(best_score.items(), key=lambda kv: -kv[1])[:k_pos]
                pos = [self._demo(gi, c) for gi, _ in ranked]

            # 兜底：正例不足时用随机正例补齐，保证提示结构恒定。
            # 否则正例数会随样本波动，导致部分提示几乎无正例可参考。
            if len(pos) < k_pos:
                seen_f = {id(d) for d in pos}
                for gi in self.pos_by_type[c]:
                    if gi in seen_f:
                        continue
                    pos.append(self._demo(gi, c)); seen_f.add(gi)
                    if len(pos) >= k_pos:
                        break

            # 负例仍走句子级（负句没有"实体"可供实体级检索）
            q = self.sent_encoder.encode([s["context"]])
            ni, _ = self._knn(q, base[self.neg_by_type[c]], k_neg)
            neg = [self._demo(self.neg_by_type[c][int(i)], c) for i in ni[0]]
            res_e.append(self._interleave(pos, neg))
        return res_e

    def release(self) -> None:
        """释放编码器显存，给后续 LLM 推理腾地方。"""
        self._sent_encoder = None
        self._char_encoder = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# 自检 / 演示
# ---------------------------------------------------------------------------

def selftest() -> bool:
    """用极小假数据跑通三档策略（random 不需要 GPU；另两档缺模型时跳过）。"""
    idx = [
        {"sample_id": 0, "text": "他前往上海市浦东新区出差。",
         "spans": [{"start": 3, "end": 9, "type": "address"}]},
        {"sample_id": 1, "text": "我校位于北京市海淀区中关村大街。",
         "spans": [{"start": 4, "end": 14, "type": "address"}]},
        {"sample_id": 2, "text": "华为公司发布了新手机。",
         "spans": [{"start": 0, "end": 1, "type": "company"}]},
    ]
    tests = [
        {"context": "北京时间3月15日，华为在东莞松山湖基地发布了新款折叠屏手机。",
         "entity_label": "company"},
    ]
    ok = True

    r = ExampleRetriever(strategy="random", k=4, neg_ratio=0.5).fit(idx)
    got = r.retrieve(tests, tag="selftest")
    print(f"  random   -> {[(d['input'][:12], d['output'][:16]) for d in got[0]]}")
    if not any("@@" in d["output"] for d in got[0]):
        print("  [FAIL] random 未产出任何正例"); ok = False

    for st in ("sentence", "entity"):
        try:
            r = ExampleRetriever(strategy=st, k=4, neg_ratio=0.5).fit(idx)
            got = r.retrieve(tests, tag="selftest")
            print(f"  {st:<9}-> {[(d['input'][:12], d['output'][:16]) for d in got[0]]}")
            r.release()
        except Exception as e:  # noqa: BLE001
            print(f"  {st:<9}-> 跳过（{type(e).__name__}: {e}）")
    return ok


if __name__ == "__main__":
    print("=" * 78)
    print("GPT-NER 示例检索模块 · 自检")
    print("=" * 78)
    selftest()
