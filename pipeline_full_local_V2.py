#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

"""
EmpathyEval full pipeline: 训练 + 推理 + JSONL 提交。

训练数据:
  1. empatheticDialogue_n_multi-emotion_flat.jsonl
  2. empatheticDialogue_t_multi-context_flat.jsonl

测试数据:
  phase1-test_multi-context_gigaspeech/phase1-test_gigaspeech_release.json

单条音频特征:
  WavLM 177 维 + emotion2vec 178 维 = 355 维

Pairwise 特征:
  concat([A, B, A-B, |A-B|, A/(B+1e-8)]) = 1775 维
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DATA_DIR = "/root/autodl-tmp/EmpathyEval"
TEST_JSON = (
    "/root/autodl-tmp/EmpathyEval/"
    "phase1-test_multi-context_gigaspeech/phase1-test_gigaspeech_release.json"
)
TEST_AUDIO_BASE = "/root/autodl-tmp/EmpathyEval/phase1-test_multi-context_gigaspeech"

N_MULTI_JSONL = "/root/autodl-tmp/EmpathyEval/empatheticDialogue_n_multi-emotion_flat.jsonl"
N_MULTI_AUDIO_DIR = (
    "/root/autodl-tmp/EmpathyEval/"
    "n_multi-emotion/empatheticDialogue_n_multi-emotion/response_audio"
)

T_MULTI_JSONL = "/root/autodl-tmp/EmpathyEval/empatheticDialogue_t_multi-context_flat.jsonl"
T_MULTI_AUDIO_DIR = (
    "/root/autodl-tmp/EmpathyEval/"
    "t_multi-context/empatheticDialogue_t_multi-context/response_audio"
)
DEFAULT_WAVLM_PATH = "/root/autodl-tmp/models/wavlm"
DEFAULT_EMOTION2VEC_MODEL = "iic/emotion2vec_plus_large"

WAVLM_DIM = 177
EMOTION_DIM = 178
AUDIO_FEAT_DIM = WAVLM_DIM + EMOTION_DIM
PAIR_FEAT_DIM = AUDIO_FEAT_DIM * 5
EMOTIONS = ("happy", "sad", "fearful", "angry", "surprised")


def set_seed(seed: int = 42) -> None:
    """固定随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_random_vector(key: str, dim: int) -> np.ndarray:
    """用 key 生成确定性随机向量，便于失败时稳定占位。"""
    seed = abs(hash(key)) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=dim).astype(np.float32)


def fit_dim(vec: Any, dim: int) -> np.ndarray:
    """将任意向量截断或补零到固定维度。"""
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == dim:
        return arr
    if arr.size > dim:
        return arr[:dim].astype(np.float32)
    out = np.zeros(dim, dtype=np.float32)
    out[: arr.size] = arr
    return out


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """读取 jsonl 文件。"""
    if not os.path.exists(path):
        print(f"[WARN] 训练标注文件不存在: {path}")
        return []

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] JSONL 解析失败: {path}:{line_no}, 原因: {exc}")
    return rows


def resolve_audio_path(rel_path: Optional[str]) -> Optional[str]:
    """将测试集相对音频路径转换为绝对路径。"""
    if not rel_path:
        return None

    path = str(rel_path)
    if os.path.isabs(path) and os.path.exists(path):
        return path

    path = path.replace("\\", "/")
    for marker in ("phase1-test_gigaspeech_options/", "phase1-test_gigaspeech/"):
        if marker in path:
            rel_part = path[path.index(marker) :]
            return os.path.join(TEST_AUDIO_BASE, rel_part)

    if path.startswith("./"):
        path = path[2:]
    return os.path.join(TEST_AUDIO_BASE, path)


def load_audio_torchaudio(audio_path: str, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """用 torchaudio 读取单声道 16k 音频。"""
    import torchaudio

    wav, sr = torchaudio.load(audio_path)
    if wav.ndim == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav.squeeze(0), sr


class AudioFeatureExtractor:
    """WavLM + emotion2vec 音频特征提取器。"""

    def __init__(
        self,
        device: torch.device,
        use_wavlm: bool = True,
        use_emotion2vec: bool = True,
        wavlm_path: str = DEFAULT_WAVLM_PATH,
        emotion2vec_model: str = DEFAULT_EMOTION2VEC_MODEL,
    ) -> None:
        self.device = device
        self.cache: Dict[str, np.ndarray] = {}
        self.wavlm_path = wavlm_path
        self.emotion2vec_model = emotion2vec_model

        self.wavlm_processor = None
        self.wavlm_model = None
        self.use_wavlm = False

        self.emotion_model = None
        self.use_emotion2vec = False

        if use_wavlm:
            self._load_wavlm()
        if use_emotion2vec:
            self._load_emotion2vec()

    def _load_wavlm(self) -> None:
        """加载 WavLM。"""
        try:
            from transformers import AutoFeatureExtractor, AutoModel

            model_name = self.wavlm_path
            self.wavlm_processor = AutoFeatureExtractor.from_pretrained(
                model_name,
                local_files_only=True,
            )
            self.wavlm_model = AutoModel.from_pretrained(
                model_name,
                local_files_only=True,
            ).to(self.device)
            self.wavlm_model.eval()
            self.use_wavlm = True
            print(f"[INFO] 已加载 WavLM: {model_name}")
        except Exception as exc:
            print(f"[WARN] WavLM 加载失败，将使用占位特征。原因: {exc}")

    def _load_emotion2vec(self) -> None:
        """加载 emotion2vec。"""
        try:
            from funasr import AutoModel as FunASRAutoModel

            errors = []
            for hub in ("ms", "hf"):
                try:
                    self.emotion_model = FunASRAutoModel(
                        model=self.emotion2vec_model,
                        hub=hub,
                        device=str(self.device),
                        trust_remote_code=True,
                    )
                    self.use_emotion2vec = True
                    print(f"[INFO] 已加载 emotion2vec: {self.emotion2vec_model} (hub={hub})")
                    return
                except Exception as exc:
                    errors.append(f"hub={hub}: {exc}")
            raise RuntimeError("; ".join(errors))
        except Exception as exc:
            print(f"[WARN] emotion2vec 加载失败，将使用占位特征。原因: {exc}")

    @torch.no_grad()
    def extract_wavlm(self, audio_path: str) -> np.ndarray:
        """提取 WavLM 特征并压缩到 177 维。"""
        if not self.use_wavlm:
            return stable_random_vector(audio_path + "::wavlm", WAVLM_DIM)

        try:
            wav, sr = load_audio_torchaudio(audio_path, target_sr=16000)
            inputs = self.wavlm_processor(
                wav.cpu().numpy(),
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.wavlm_model(**inputs)
            feat = outputs.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy()
            return fit_dim(feat, WAVLM_DIM)
        except Exception as exc:
            print(f"[WARN] WavLM 特征提取失败，使用占位特征: {audio_path}, 原因: {exc}")
            return stable_random_vector(audio_path + "::wavlm_fail", WAVLM_DIM)

    def extract_emotion2vec(self, audio_path: str) -> np.ndarray:
        """提取 emotion2vec 特征并压缩到 178 维。"""
        if not self.use_emotion2vec:
            return stable_random_vector(audio_path + "::emotion2vec", EMOTION_DIM)

        try:
            result = self.emotion_model.generate(
                audio_path,
                granularity="utterance",
                extract_embedding=True,
            )
            feat = self._parse_emotion2vec_result(result)
            return fit_dim(feat, EMOTION_DIM)
        except Exception as exc:
            print(f"[WARN] emotion2vec 特征提取失败，使用占位特征: {audio_path}, 原因: {exc}")
            return stable_random_vector(audio_path + "::emotion2vec_fail", EMOTION_DIM)

    @staticmethod
    def _parse_emotion2vec_result(result: Any) -> np.ndarray:
        """兼容 FunASR 不同版本的返回字段。"""
        candidates = ("feats", "embedding", "xvector", "hidden_states", "scores")

        if isinstance(result, list) and result:
            item = result[0]
            if isinstance(item, dict):
                for key in candidates:
                    if key in item:
                        return np.asarray(item[key], dtype=np.float32)

        if isinstance(result, dict):
            for key in candidates:
                if key in result:
                    return np.asarray(result[key], dtype=np.float32)

        raise ValueError("emotion2vec 返回结果中没有可用 embedding 字段")

    def extract(
        self,
        audio_path: Optional[str],
        use_cache: bool = True,
        log_timing: bool = False,
    ) -> np.ndarray:
        """提取 355 维音频特征，可选择打印耗时诊断。"""
        key = str(audio_path) if audio_path else "__missing_audio__"
        if use_cache and key in self.cache:
            if log_timing:
                print(f"[FEAT] cache hit: {audio_path}")
            return self.cache[key]

        start = time.time()
        if not audio_path or not os.path.exists(audio_path):
            print(f"[WARN] 音频不存在，使用占位特征: {audio_path}")
            feat = stable_random_vector(key + "::missing", AUDIO_FEAT_DIM)
            if use_cache:
                self.cache[key] = feat
            elapsed = time.time() - start
            if log_timing:
                print(f"[FEAT] {audio_path} | elapsed={elapsed:.3f}s | missing -> random")
            if elapsed < 0.01:
                print(f"[WARN] 特征提取耗时过短 ({elapsed:.3f}s)，可能使用了占位特征: {audio_path}")
            return feat

        wavlm_feat = self.extract_wavlm(audio_path)
        emotion_feat = self.extract_emotion2vec(audio_path)
        feat = np.concatenate([wavlm_feat, emotion_feat], axis=0).astype(np.float32)
        feat = fit_dim(feat, AUDIO_FEAT_DIM)
        if use_cache:
            self.cache[key] = feat
        elapsed = time.time() - start
        if log_timing:
            print(
                f"[FEAT] {audio_path} | elapsed={elapsed:.3f}s | "
                f"wavlm={self.use_wavlm} emotion2vec={self.use_emotion2vec}"
            )
        if elapsed < 0.01:
            print(f"[WARN] 特征提取耗时过短 ({elapsed:.3f}s)，可能使用了占位特征: {audio_path}")
        return feat

    def warmup(self, audio_paths: Iterable[Optional[str]], desc: str = "Extract features") -> None:
        """提前缓存音频特征。"""
        unique_paths = list(dict.fromkeys([p for p in audio_paths if p]))
        found = sum(1 for p in unique_paths if os.path.exists(p))
        print(f"[INFO] {desc}: 音频存在数量 {found}/{len(unique_paths)}")
        for path in tqdm(unique_paths, desc=desc):
            self.extract(path)


def construct_pairwise_features(feat_a: np.ndarray, feat_b: np.ndarray) -> np.ndarray:
    """构造 1775 维 pairwise 特征。"""
    feat_a = fit_dim(feat_a, AUDIO_FEAT_DIM)
    feat_b = fit_dim(feat_b, AUDIO_FEAT_DIM)
    pair_feat = np.concatenate(
        [
            feat_a,
            feat_b,
            feat_a - feat_b,
            np.abs(feat_a - feat_b),
            feat_a / (feat_b + 1e-8),
        ],
        axis=0,
    )
    return pair_feat.astype(np.float32)


class PairwiseMLP(nn.Module):
    """Pairwise MLP: 1775 -> 256 -> 128 -> 1。"""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PAIR_FEAT_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PairwiseDataset(Dataset):
    """MLP 训练数据集。"""

    def __init__(self, xs: Sequence[np.ndarray], ys: Sequence[float]) -> None:
        self.xs = [np.asarray(x, dtype=np.float32) for x in xs]
        self.ys = [float(y) for y in ys]

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.xs[idx])
        y = torch.tensor(self.ys[idx], dtype=torch.float32)
        return x, y


def load_n_multi_emotion_pairs(jsonl_path: str, audio_dir: str) -> List[Tuple[str, str]]:
    """
    加载 n_multi-emotion 训练音频对。

    音频命名:
      {id}_{emotion}_goodPara.wav
      {id}_{emotion}_badPara.wav
    """
    rows = read_jsonl(jsonl_path)
    pairs: List[Tuple[str, str]] = []

    for item in rows:
        sample_id = item.get("id")
        contexts = item.get("contexts", {})
        if not sample_id or not isinstance(contexts, dict):
            continue

        for emotion in EMOTIONS:
            response_key = f"{emotion}_response"
            if response_key not in contexts:
                continue
            good = os.path.join(audio_dir, f"{sample_id}_{emotion}_goodPara.wav")
            bad = os.path.join(audio_dir, f"{sample_id}_{emotion}_badPara.wav")
            pairs.append((good, bad))

    print(f"[INFO] n_multi-emotion 音频对数量: {len(pairs)}")
    return pairs


def load_t_multi_context_pairs(jsonl_path: str, audio_dir: str) -> List[Tuple[str, str]]:
    """
    加载 t_multi-context 训练音频对。

    音频命名:
      {id}_{number}_goodPara.wav
      {id}_{number}_badPara.wav
    """
    rows = read_jsonl(jsonl_path)
    pairs: List[Tuple[str, str]] = []

    for item in rows:
        sample_id = item.get("id")
        contexts = item.get("contexts", [])
        if not sample_id or not isinstance(contexts, list):
            continue

        for idx, ctx in enumerate(contexts, start=1):
            if idx > 2:
                break
            if not isinstance(ctx, dict) or "response" not in ctx:
                continue
            good = os.path.join(audio_dir, f"{sample_id}_{idx}_goodPara.wav")
            bad = os.path.join(audio_dir, f"{sample_id}_{idx}_badPara.wav")
            pairs.append((good, bad))

    print(f"[INFO] t_multi-context 音频对数量: {len(pairs)}")
    return pairs


def load_training_pairs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    """从两个训练集加载 good/bad 音频路径对。"""
    pairs: List[Tuple[str, str]] = []
    pairs.extend(load_n_multi_emotion_pairs(args.n_multi_jsonl, args.n_multi_audio_dir))
    pairs.extend(load_t_multi_context_pairs(args.t_multi_jsonl, args.t_multi_audio_dir))

    found_good_bad = sum(1 for g, b in pairs if os.path.exists(g) and os.path.exists(b))
    print(f"[INFO] 训练音频对存在数量: {found_good_bad}/{len(pairs)}")

    if args.sample_ratio < 1.0:
        if args.sample_ratio <= 0:
            raise ValueError("--sample_ratio 必须大于 0")
        keep = max(1, int(len(pairs) * args.sample_ratio))
        pairs = pairs[:keep]
        print(f"[INFO] sample_ratio 生效后训练音频对数量: {len(pairs)}")

    return pairs


def build_pairwise_dataset(
    train_pairs: Sequence[Tuple[str, str]],
    extractor: AudioFeatureExtractor,
) -> Tuple[List[np.ndarray], List[float]]:
    """将 good/bad 音频对转换为 pairwise MLP 训练样本。"""
    xs: List[np.ndarray] = []
    ys: List[float] = []

    for good_path, bad_path in tqdm(train_pairs, desc="Build pairwise train data"):
        good_feat = extractor.extract(good_path)
        bad_feat = extractor.extract(bad_path)

        xs.append(construct_pairwise_features(good_feat, bad_feat))
        ys.append(1.0)

        xs.append(construct_pairwise_features(bad_feat, good_feat))
        ys.append(0.0)

    print(f"[INFO] Pairwise 训练样本数量: {len(xs)}")
    return xs, ys


def train_model(
    xs: Sequence[np.ndarray],
    ys: Sequence[float],
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    checkpoint_path: str = "pairwise_mlp_best.pt",
) -> PairwiseMLP:
    """训练 MLP，并保存训练 loss 最低的权重。"""
    if len(xs) == 0:
        raise ValueError("没有可训练的 pairwise 样本")

    dataset = PairwiseDataset(xs, ys)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = PairwiseMLP().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_y.size(0)
            total_count += batch_y.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(total_count, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            if checkpoint_path:
                save_checkpoint(model, checkpoint_path)
                print(f"[TRAIN] 保存当前最佳模型: loss={best_loss:.4f}")

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"[TRAIN] Epoch {epoch:02d}/{epochs} | loss={avg_loss:.4f}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        model = load_checkpoint(PairwiseMLP(), checkpoint_path, device)
        print(f"[TRAIN] 训练结束，已重新加载最佳模型: {checkpoint_path}")

    return model


def load_test_data(test_json_path: str, sample_ratio: float = 1.0) -> List[Dict[str, Any]]:
    """加载测试集 JSON。"""
    if not os.path.exists(test_json_path):
        raise FileNotFoundError(f"测试集 JSON 不存在: {test_json_path}")

    with open(test_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("测试集 JSON 顶层必须为 list")

    for idx, item in enumerate(data):
        item.setdefault("question_id", str(idx))
        if "options" not in item or "opt-A" not in item["options"] or "opt-B" not in item["options"]:
            raise ValueError(f"测试样本缺少 opt-A/opt-B: {item.get('question_id')}")

    if sample_ratio < 1.0:
        if sample_ratio <= 0:
            raise ValueError("--sample_ratio 必须大于 0")
        keep = max(1, int(len(data) * sample_ratio))
        data = data[:keep]

    print(f"[INFO] 测试样本数量: {len(data)}")
    return data


def collect_train_audio_paths(train_pairs: Sequence[Tuple[str, str]]) -> List[str]:
    """收集训练音频路径。"""
    paths: List[str] = []
    for good, bad in train_pairs:
        paths.append(good)
        paths.append(bad)
    return paths


def collect_test_audio_paths(test_data: Sequence[Dict[str, Any]]) -> List[str]:
    """收集测试集候选音频路径。"""
    paths: List[str] = []
    for item in test_data:
        paths.append(resolve_audio_path(item["options"]["opt-A"]))
        paths.append(resolve_audio_path(item["options"]["opt-B"]))
    return [p for p in paths if p]


@torch.no_grad()
def predict_one(
    model: PairwiseMLP,
    extractor: AudioFeatureExtractor,
    opt_a_path: str,
    opt_b_path: str,
    device: torch.device,
) -> Tuple[str, float]:
    """预测单条样本答案。"""
    feat_a = extractor.extract(opt_a_path)
    feat_b = extractor.extract(opt_b_path)
    pair_feat = construct_pairwise_features(feat_a, feat_b)

    model.eval()
    x = torch.from_numpy(pair_feat).unsqueeze(0).to(device)
    prob_a_better = float(model(x).item())
    answer = "A" if prob_a_better >= 0.5 else "B"
    return answer, prob_a_better


def run_inference(
    model: PairwiseMLP,
    extractor: AudioFeatureExtractor,
    test_data: Sequence[Dict[str, Any]],
    device: torch.device,
    force_reextract_debug: bool = False,
) -> List[Dict[str, str]]:
    """测试集推理。"""
    predictions: List[Dict[str, str]] = []
    found_audio = 0
    total_audio = 0
    answer_counts = {"A": 0, "B": 0}

    model.eval()
    first_param = next(model.parameters()).detach().float().cpu()
    print(f"[INFO] 推理模型参数诊断: first_param_mean={first_param.mean().item():.6f}")
    print(
        f"[INFO] 特征模型状态: wavlm_loaded={extractor.use_wavlm}, "
        f"emotion2vec_loaded={extractor.use_emotion2vec}, cache_size={len(extractor.cache)}"
    )

    for idx, item in enumerate(tqdm(test_data, desc="Inference")):
        opt_a = resolve_audio_path(item["options"]["opt-A"])
        opt_b = resolve_audio_path(item["options"]["opt-B"])

        for path in (opt_a, opt_b):
            total_audio += 1
            if path and os.path.exists(path):
                found_audio += 1
            else:
                print(f"[WARN] 测试音频不存在，将使用占位特征: {path}")

        if force_reextract_debug and idx < 2:
            extractor.extract(opt_a, use_cache=False, log_timing=True)
            extractor.extract(opt_b, use_cache=False, log_timing=True)

        answer, _ = predict_one(model, extractor, opt_a, opt_b, device)
        answer_counts[answer] += 1
        predictions.append({"question_id": str(item["question_id"]), "answer": answer})

    print(f"[INFO] 测试音频存在数量: {found_audio}/{total_audio}")
    print(f"[INFO] 推理答案分布: A={answer_counts['A']}, B={answer_counts['B']}")
    return predictions


def save_submission(predictions: Sequence[Dict[str, str]], output_path: str) -> None:
    """保存 JSONL 提交文件。"""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[INFO] 提交文件已保存: {output_path}")


def save_checkpoint(model: PairwiseMLP, output_path: str) -> None:
    """保存模型权重。"""
    if not output_path:
        return
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"[INFO] 模型权重已保存: {output_path}")


def load_checkpoint(model: PairwiseMLP, checkpoint_path: str, device: torch.device) -> PairwiseMLP:
    """加载模型权重。"""
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[INFO] 已加载模型权重: {checkpoint_path}")
    return model


def parse_args() -> argparse.Namespace:
    """命令行参数。"""
    parser = argparse.ArgumentParser(description="EmpathyEval full training and inference pipeline")
    parser.add_argument("--test_json", type=str, default=TEST_JSON, help="测试集 JSON 路径")
    parser.add_argument("--output", type=str, default="TeamID.json", help="提交文件输出路径")

    parser.add_argument("--n_multi_jsonl", type=str, default=N_MULTI_JSONL, help="n_multi-emotion 标注 JSONL")
    parser.add_argument("--n_multi_audio_dir", type=str, default=N_MULTI_AUDIO_DIR, help="n_multi-emotion 音频目录")
    parser.add_argument("--t_multi_jsonl", type=str, default=T_MULTI_JSONL, help="t_multi-context 标注 JSONL")
    parser.add_argument("--t_multi_audio_dir", type=str, default=T_MULTI_AUDIO_DIR, help="t_multi-context 音频目录")

    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam 学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="调试用采样比例")

    parser.add_argument("--wavlm_path", type=str, default=DEFAULT_WAVLM_PATH, help="本地 WavLM 模型目录")
    parser.add_argument(
        "--emotion2vec_model",
        type=str,
        default=DEFAULT_EMOTION2VEC_MODEL,
        help="emotion2vec 模型名或本地缓存模型名",
    )
    parser.add_argument("--no_wavlm", action="store_true", help="禁用 WavLM，使用占位特征")
    parser.add_argument("--no_emotion2vec", action="store_true", help="禁用 emotion2vec，使用占位特征")
    parser.add_argument("--skip_train", action="store_true", help="跳过训练，必须配合 --checkpoint 使用")
    parser.add_argument("--checkpoint", type=str, default="pairwise_mlp_best.pt", help="最佳模型权重保存/加载路径")
    parser.add_argument("--debug_reextract_test", action="store_true", help="推理时强制重提取前 2 条测试音频并打印耗时")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    extractor = AudioFeatureExtractor(
        device=device,
        use_wavlm=not args.no_wavlm,
        use_emotion2vec=not args.no_emotion2vec,
        wavlm_path=args.wavlm_path,
        emotion2vec_model=args.emotion2vec_model,
    )

    test_data = load_test_data(args.test_json, sample_ratio=args.sample_ratio)

    if args.skip_train:
        if not args.checkpoint or not os.path.exists(args.checkpoint):
            raise FileNotFoundError("--skip_train 需要提供存在的 --checkpoint")
        model = load_checkpoint(PairwiseMLP(), args.checkpoint, device)
    else:
        train_pairs = load_training_pairs(args)

        # 先缓存训练和测试音频特征，后续构造 pair 和推理会直接命中缓存。
        train_audio_paths = collect_train_audio_paths(train_pairs)
        test_audio_paths = collect_test_audio_paths(test_data)
        extractor.warmup(train_audio_paths + test_audio_paths, desc="Extract all audio features")

        xs, ys = build_pairwise_dataset(train_pairs, extractor)
        model = train_model(
            xs,
            ys,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            checkpoint_path=args.checkpoint,
        )
        if args.checkpoint and os.path.exists(args.checkpoint):
            model = load_checkpoint(PairwiseMLP(), args.checkpoint, device)

    # 如果是加载已有权重，测试特征可能还没有缓存，这里单独预热一次。
    extractor.warmup(collect_test_audio_paths(test_data), desc="Extract test audio features")
    predictions = run_inference(
        model,
        extractor,
        test_data,
        device,
        force_reextract_debug=args.debug_reextract_test,
    )
    save_submission(predictions, args.output)


if __name__ == "__main__":
    main()
