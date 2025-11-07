#!/usr/bin/env python3
"""
Attach text-only emotion labels (7 classes) to your ASR segments JSON.

Model (default): cardiffnlp/twitter-roberta-base-emotion
Labels: anger, joy, sadness, fear, disgust, surprise, neutral

Input  : transcript.json  (list of dicts: {type, start, end, speaker, text, ...})
Output : transcript_with_text_emotion.json (same + text emotion fields)

Install:
    pip install torch transformers tqdm

Usage:
    python text_emotion_extractor.py \
        --asr_json transcript.json \
        --out_json transcript_with_text_emotion.json
    # (optional)
    #   --min_chars 3 --max_seq_len 128 --batch_size 32 --write_probs --topk 5
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

MODEL_ID = "cardiffnlp/twitter-roberta-base-emotion"  # 7 emotions
BATCH_SIZE = 32
MIN_CHARS = 2                  # skip very short/empty turns
MAX_SEQ_LEN = 256             # truncate long utterances
WRITE_PROBS = False           # store full (or top-k) probability dict
TOPK = 0                      # if >0, keep only top-k probs per segment

def load_segments(path: str) -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text())

def save_segments(path: str, segs: List[Dict[str, Any]]):
    Path(path).write_text(json.dumps(segs, indent=2, ensure_ascii=False))

def chunked(iterable, n):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

def prepare_text(s: Dict[str, Any]) -> str:
    return (s.get("text") or "").strip()

def run_text_emotion(
    texts: List[str],
    tok: AutoTokenizer,
    mdl: AutoModelForSequenceClassification,
    device: torch.device,
    max_len: int = 256,
    batch_size: int = 32,
    write_probs: bool = False,
    topk: int = 0,
) -> Tuple[List[str], List[float], List[Dict[str, float]]]:

    mdl.eval()
    labels_out, confs_out, probs_out = [], [], []
    id2label = {int(k): v for k, v in mdl.config.id2label.items()}

    with torch.no_grad():
        for batch in chunked(texts, batch_size):
            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = mdl(**enc).logits          # [B, C]
            prob = F.softmax(logits, dim=-1)    # [B, C]
            top_idx = prob.argmax(dim=-1)       # [B]

            for i in range(prob.size(0)):
                idx = int(top_idx[i].item())
                label = id2label[idx]
                conf = float(prob[i, idx].item())

                if write_probs or topk > 0:
                    dist = {id2label[j]: float(prob[i, j].item()) for j in range(prob.size(1))}
                    if topk > 0:
                        dist = dict(sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:topk])
                    probs_out.append(dist)
                else:
                    probs_out.append({})

                labels_out.append(label)
                confs_out.append(conf)

    return labels_out, confs_out, probs_out


def main():
    ap = argparse.ArgumentParser(description="Attach 7-class text emotion to ASR JSON.")
    ap.add_argument("--asr_json", required=True, help="Path to transcript.json (ASR segments)")
    ap.add_argument("--out_json", required=True, help="Output path for enriched JSON")
    ap.add_argument("--model", default=MODEL_ID, help="HF model id (default: cardiffnlp/twitter-roberta-base-emotion)")
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    ap.add_argument("--min_chars", type=int, default=MIN_CHARS, help="Skip text shorter than this")
    ap.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN, help="Truncate long inputs")
    ap.add_argument("--write_probs", action="store_true", help="Store probability distribution per segment")
    ap.add_argument("--topk", type=int, default=TOPK, help="If >0, keep only top-k probs per segment")
    args = ap.parse_args()

    model_id = args.model
    batch_size = int(args.batch_size)
    min_chars = int(args.min_chars)
    max_len = int(args.max_seq_len)
    write_probs = bool(args.write_probs)
    topk = int(args.topk)

    # Load data
    segs = load_segments(args.asr_json)

    # Collect texts to process (speech segments with enough text)
    idxs, texts = [], []
    for i, s in enumerate(segs):
        if s.get("type") != "speech":
            continue
        t = prepare_text(s)
        if len(t) >= min_chars:
            idxs.append(i)
            texts.append(t)

    if not texts:
        save_segments(args.out_json, segs)
        print("No eligible speech segments found. Wrote unchanged output.")
        return

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model & tokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)

    print(f"\n[TER] Running text emotion on {len(texts)} segments using {model_id} "
          f"(batch={batch_size}, max_len={max_len}) ...\n")

    labels, confs, probs = run_text_emotion(
        texts=texts,
        tok=tok,
        mdl=mdl,
        device=device,
        max_len=max_len,
        batch_size=batch_size,
        write_probs=write_probs,
        topk=topk,
    )

    for j, i_seg in enumerate(tqdm(idxs, desc="Annotating", unit="seg")):
        s = segs[i_seg]
        em = s.get("emotion") or {}
        em["category_text"] = labels[j]
        em["category_text_conf"] = confs[j]
        if write_probs or topk > 0:
            em["category_text_probs"] = probs[j]
        s["emotion"] = em

    save_segments(args.out_json, segs)
    print(f"\nWrote file to {args.out_json}")

if __name__ == "__main__":
    main()
