#!/usr/bin/env python
"""Generate the two notebooks of this project.

    notebooks/01_train_bart_cnn_dailymail.ipynb   Kaggle training + evaluation pipeline
    notebooks/02_test_summarizer.ipynb            load the trained model and test it on custom texts

Both notebooks are self-contained for Kaggle: every module in src/ is embedded with %%writefile and
the texts in test_cases/ are embedded as a dict. Edit src/ or test_cases/, then regenerate with

    python scripts/build_notebooks.py
"""
from __future__ import annotations

import glob
import os

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_FILES = ["__init__.py", "preprocessing.py", "extractive.py", "metrics.py", "summarizer.py"]
REPO = "https://github.com/ksidharth8/text-summarization-transformers"


def read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def test_cases_code() -> str:
    lines = ["TEST_CASES = {"]
    for p in sorted(glob.glob(os.path.join(ROOT, "test_cases", "*.txt"))):
        text = open(p, encoding="utf-8").read().strip()
        assert '"""' not in text and "\\" not in text, p
        lines.append(f'    {os.path.splitext(os.path.basename(p))[0]!r}: """{text}""",')
    lines.append("}")
    return "\n".join(lines)


def notebook(cells: list) -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = [new_markdown_cell(src) if kind == "md" else new_code_cell(src) for kind, src in cells]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True,
                   "language": "python", "sourceType": "notebook"},
    }
    return nb


# =============================================================================================
# 01 - training notebook
# =============================================================================================
TRAIN = []
md = lambda s: TRAIN.append(("md", s.strip()))      # noqa: E731
code = lambda s: TRAIN.append(("code", s.strip()))  # noqa: E731

md(r'''
# Text Summarization with Transformers — fine-tuning BART on CNN/DailyMail

**Deep Learning course project** · Abhinav Anand · B.Tech AI & ML, BIT Mesra · [github.com/ksidharth8/text-summarization-transformers](https://github.com/ksidharth8/text-summarization-transformers)

Self-contained training + evaluation pipeline. It compares **extractive** summarizers (LEAD-3, TextRank, transformer-embedding MMR) with an **abstractive** transformer (`facebook/bart-base`, fine-tuned here).

| § | Step | Main output |
|---|---|---|
| 1 | Setup, configuration, project modules (`src/`) | |
| 2 | Load + clean the dataset, statistics | `results/figures/fig_lengths.png` |
| 3 | Extractive baselines on the test set | ROUGE table |
| 4 | Fine-tune BART-base (fp16, time-budgeted LR schedule) | `results/figures/fig_training_curves.png` |
| 5 | Save the model (Hugging Face format + zip) | `bart-base-cnn-summarizer/`, `bart-base-cnn-summarizer.zip` |
| 6 | Evaluation: ROUGE-1/2/L/Lsum + 95% CIs, zero-shot vs fine-tuned, significance, abstractiveness | `results/metrics.json`, figures |
| 7 | Qualitative examples + custom test cases (incl. long-document mode) | `results/custom_cases.md` |

### How to run on Kaggle
1. **Add Input** → search `newspaper-text-summarization-cnn-dailymail` (by *gowrishankarp*) → **Add**.
2. **Settings** → *Accelerator*: **GPU T4 x2** · *Internet*: **On** (a phone-verified account is required; the internet is used to download `facebook/bart-base`).
3. Leave `RUN_MODE = "smoke"` and click **Run All** (≈10–15 min). This tests data, GPU, internet and every step on small subsets.
4. Change to `RUN_MODE = "full"`, then **Save Version → Save & Run All (Commit)**. Runs in the background (≈5.5–6.5 h; training itself is capped by `train_hours`). You can close the browser.
5. When the version is complete: **Output** tab → download `bart-base-cnn-summarizer.zip` and `results/`.
''')

code(r'''
# 1. Setup --------------------------------------------------------------------------------------
# Kaggle images already contain torch / transformers / datasets / accelerate; only small extras may be missing.
import importlib.util, subprocess, sys
for pip_name, module in [("rouge-score", "rouge_score"), ("nltk", "nltk")]:
    if importlib.util.find_spec(module) is None:
        print(f"installing {pip_name} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=False)
''')

code(r'''
import os, json

# "smoke": ~10-15 min end-to-end check on small subsets (the scores are NOT meaningful)
# "full" : real training on all ~287k training articles + evaluation on the full 11,490-article test set
RUN_MODE = os.environ.get("SUMM_RUN_MODE", "smoke")

CFG = dict(
    model_name="facebook/bart-base",       # alternatives: "facebook/bart-large" (per_device_batch=4), "t5-small", "google/flan-t5-base"
    embed_model="sentence-transformers/all-MiniLM-L6-v2",   # sentence encoder for the embedding-MMR baseline
    seed=42,
    # data
    train_samples=None, val_samples=2000, test_samples=None,  # None = the full split
    max_source_len=512, max_target_len=128,
    # optimisation: training stops after `epochs` OR `train_hours`, whichever comes first (the LR schedule adapts)
    epochs=3, train_hours=4.5,
    lr=5e-5, weight_decay=0.01, warmup_steps=500, label_smoothing=0.1, max_grad_norm=1.0,
    per_device_batch=16, effective_batch=32, fp16=True,       # CUDA out of memory? -> per_device_batch=8
    eval_every=1000, log_every=100,
    # decoding (CNN/DailyMail settings of the BART paper)
    gen_batch=32, num_beams=4, max_new_tokens=142, min_new_tokens=56, length_penalty=2.0, no_repeat_ngram_size=3,
    # evaluation extras
    zero_shot_samples=500, novelty_samples=2000, run_embed_extractive=True, n_examples=5,
    push_to_hub=False, hub_repo="redcode333/bart-base-cnn-dailymail",   # needs a Kaggle secret named HF_TOKEN
)
SMOKE = dict(train_samples=2000, val_samples=200, test_samples=200, epochs=1, train_hours=0.12, warmup_steps=20,
             eval_every=40, log_every=10, zero_shot_samples=40, novelty_samples=200)
if RUN_MODE == "smoke":
    CFG.update(SMOKE)
# environment overrides (used to test this notebook offline with tiny models)
if os.environ.get("SUMM_MODEL_NAME"):
    CFG["model_name"] = os.environ["SUMM_MODEL_NAME"]
if os.environ.get("SUMM_EMBED_MODEL"):
    CFG["embed_model"] = os.environ["SUMM_EMBED_MODEL"]
CFG.update(json.loads(os.environ.get("SUMM_CFG_OVERRIDES", "{}")))
print("RUN_MODE =", RUN_MODE)
print(json.dumps(CFG, indent=1))
''')

code(r'''
import os, sys, re, gc, glob, json, math, time, shutil, inspect, zipfile, platform, dataclasses, warnings, datetime
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import torch
import transformers
import datasets
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

transformers.logging.set_verbosity_error()
try:
    datasets.utils.logging.set_verbosity_error()
    datasets.disable_progress_bars()
except Exception:
    pass

IN_KAGGLE = os.path.isdir("/kaggle/working")
WORK_DIR = os.path.abspath(os.environ.get("SUMM_WORK_DIR", "/kaggle/working" if IN_KAGGLE else "outputs"))
DATA_DIR = os.environ.get("SUMM_DATA_DIR", "/kaggle/input" if IN_KAGGLE else "data")
CKPT_DIR = os.environ.get("SUMM_CKPT_DIR", "/tmp/checkpoints")
RESULTS_DIR = os.path.join(WORK_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
for d in (WORK_DIR, os.path.join(WORK_DIR, "src"), os.path.join(WORK_DIR, "test_cases"), RESULTS_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)
MODEL_TAG = os.path.basename(CFG["model_name"].rstrip("/"))
MODEL_DIR = os.path.join(WORK_DIR, f"{MODEL_TAG}-cnn-summarizer")
if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)

N_GPU = torch.cuda.device_count()
GPU_NAME = torch.cuda.get_device_name(0) if N_GPU else "none"
if RUN_MODE == "full" and N_GPU == 0:
    raise RuntimeError("No GPU visible. Kaggle: Settings -> Accelerator -> GPU T4 x2 (then run again).")
transformers.set_seed(CFG["seed"])
NUM_PROC = max(1, min(4, os.cpu_count() or 1))
NUM_WORKERS = min(2, max(0, (os.cpu_count() or 1) - 1))
ENV = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
           datasets=datasets.__version__, numpy=np.__version__, pandas=pd.__version__, cuda=torch.version.cuda,
           gpu=f"{GPU_NAME} x{N_GPU}" if N_GPU else "CPU only", cpus=os.cpu_count())
FIGURES = {}
T_NOTEBOOK = time.time()
print(json.dumps(ENV, indent=1))
print("WORK_DIR:", WORK_DIR, "| DATA_DIR:", DATA_DIR)
''')

md(r'''
### Project modules
The cells below write the project's Python package to `src/` (identical to the GitHub repository), so the notebook output is a complete, importable package next to the trained model.
''')
for name in SRC_FILES:
    code("%%writefile {WORK_DIR}/src/" + name + "\n" + read(f"src/{name}").rstrip())

code(r'''
import importlib
import src, src.preprocessing, src.extractive, src.metrics, src.summarizer
for _m in (src, src.preprocessing, src.extractive, src.metrics, src.summarizer):
    importlib.reload(_m)
from src.preprocessing import clean_article, clean_summary, split_sentences, count_words
from src.extractive import lead_k, textrank, EmbeddingExtractor
from src.metrics import evaluate_system, paired_bootstrap, novel_ngram_ratio, ROUGE_TYPES
from src.summarizer import Summarizer, is_t5_like
print("src package ready, version", src.__version__)
''')

md(r'''
## 2. Data
The Kaggle dataset is the non-anonymised CNN/DailyMail corpus (version 3.0.0): columns `id`, `article`, `highlights` (the bullet-point highlights are the reference summaries). The CSV files are discovered automatically under `/kaggle/input`. Rows with a missing article/summary or an article shorter than 20 words are dropped.

Cleaning removes scraping boilerplate from articles (e.g. `By . Daily Mail Reporter . PUBLISHED: . 14:11 EST, 25 October 2013 .` or `LONDON, England (CNN) --`). Reference summaries are only whitespace-normalised (one highlight per line).
''')

code(r'''
ARTICLE_COLS = ("article", "text", "document", "story", "content")
SUMMARY_COLS = ("highlights", "summary", "abstract", "headline", "summaries")


def discover_csvs(root):
    csvs = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if not csvs:
        raise FileNotFoundError(f"No CSV files under {root!r}. On Kaggle: 'Add Input' -> "
                                "'newspaper-text-summarization-cnn-dailymail' (gowrishankarp).")
    pool = [p for p in csvs if re.search(r"cnn|dailymail|daily_mail", p, re.I)] or csvs

    def pick(pattern):
        hits = [p for p in pool if re.search(pattern, os.path.basename(p), re.I)]
        return max(hits, key=os.path.getsize) if hits else None
    return {"train": pick(r"train"), "validation": pick(r"val|dev"), "test": pick(r"test")}


def load_split(path):
    df = pd.read_csv(path)
    cols = {str(c).lower().strip(): c for c in df.columns}
    a_col = next((cols[c] for c in ARTICLE_COLS if c in cols), None)
    s_col = next((cols[c] for c in SUMMARY_COLS if c in cols), None)
    if a_col is None or s_col is None:
        raise ValueError(f"{path}: cannot find article/summary columns in {list(df.columns)}")
    ids = df[cols["id"]].astype(str).to_numpy() if "id" in cols else np.arange(len(df)).astype(str)
    out = pd.DataFrame({"id": ids, "article": df[a_col].to_numpy(), "highlights": df[s_col].to_numpy()})
    n_raw = len(out)
    out = out.dropna(subset=["article", "highlights"])
    out = out[out["article"].astype(str).str.count(r"\S+") >= 20]
    out = out[out["highlights"].astype(str).str.strip().str.len() > 0].reset_index(drop=True)
    out["article"], out["highlights"] = out["article"].astype(str), out["highlights"].astype(str)
    return out, n_raw - len(out)


DATA_FILES = discover_csvs(DATA_DIR)
print(json.dumps(DATA_FILES, indent=1))
frames, DROPPED = {}, {}
for split, path in DATA_FILES.items():
    if path:
        frames[split], DROPPED[split] = load_split(path)
if "train" not in frames:
    raise FileNotFoundError("No training CSV found (file name must contain 'train').")
if "test" not in frames or "validation" not in frames:            # single-file datasets -> 90/5/5
    full = frames["train"].sample(frac=1.0, random_state=CFG["seed"]).reset_index(drop=True)
    n_hold = max(1, int(0.05 * len(full)))
    frames.setdefault("test", full.iloc[:n_hold].reset_index(drop=True))
    frames.setdefault("validation", full.iloc[n_hold:2 * n_hold].reset_index(drop=True))
    frames["train"] = full.iloc[2 * n_hold:].reset_index(drop=True)
SPLIT_SIZES = {k: len(v) for k, v in frames.items()}


def subsample(df, n):
    return df if (n is None or n >= len(df)) else df.sample(n=n, random_state=CFG["seed"]).reset_index(drop=True)


train_df = subsample(frames["train"], CFG["train_samples"])
val_df = subsample(frames["validation"], CFG["val_samples"])
test_df = subsample(frames["test"], CFG["test_samples"])
del frames
gc.collect()
test_df["article_clean"] = test_df["article"].map(clean_article)
test_df["reference"] = test_df["highlights"].map(clean_summary)
print("rows per split:", SPLIT_SIZES, "| dropped:", DROPPED)
print(f"used -> train {len(train_df):,} | validation (eval loss) {len(val_df):,} | test {len(test_df):,}")
''')

code(r'''
from transformers import AutoConfig, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])
MODEL_TYPE = AutoConfig.from_pretrained(CFG["model_name"]).model_type
T5_LIKE = is_t5_like(MODEL_TYPE)
PREFIX = "summarize: " if T5_LIKE else ""

eda = train_df.sample(n=min(20000, len(train_df)), random_state=CFG["seed"])
eda_art = eda["article"].map(clean_article)
eda_sum = eda["highlights"].map(clean_summary)
art_words = eda_art.str.split().str.len().to_numpy()
sum_words = eda_sum.str.split().str.len().to_numpy()
sum_sents = eda_sum.map(lambda s: len(split_sentences(s))).to_numpy()
tok_lens = np.array([len(x) for x in tokenizer(eda_art.iloc[:2000].tolist())["input_ids"]])
_nov = min(CFG["novelty_samples"], len(test_df))
DATA_STATS = dict(
    eda_sample=len(eda),
    article_words_mean=round(float(art_words.mean()), 1), article_words_median=float(np.median(art_words)),
    article_words_p95=float(np.percentile(art_words, 95)),
    summary_words_mean=round(float(sum_words.mean()), 1), summary_sentences_mean=round(float(sum_sents.mean()), 2),
    compression_ratio=round(float(np.mean(art_words / np.maximum(sum_words, 1))), 1),
    article_tokens_mean=round(float(tok_lens.mean()), 1),
    pct_articles_over_max_source_len=round(100 * float((tok_lens > CFG["max_source_len"]).mean()), 1),
    reference_novel_ngrams=novel_ngram_ratio(test_df["article_clean"].iloc[:_nov].tolist(),
                                             test_df["reference"].iloc[:_nov].tolist()),
)
CLEAN_EXAMPLE = None
for raw in test_df["article"].head(500):
    flat = " ".join(str(raw).split())
    cleaned = clean_article(raw)
    if cleaned[:60] != flat[:60]:
        CLEAN_EXAMPLE = {"before": flat[:250], "after": cleaned[:180]}
        break
print(json.dumps(DATA_STATS, indent=1))
print("cleaning example:", json.dumps(CLEAN_EXAMPLE, indent=1, ensure_ascii=False))

fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
ax[0].hist(np.clip(art_words, 0, 2000), bins=60, color="#4C72B0")
ax[0].set_title("Article length (words)")
ax[1].hist(np.clip(tok_lens, 0, 2500), bins=60, color="#55A868")
ax[1].axvline(CFG["max_source_len"], color="red", ls="--", label=f"max_source_len = {CFG['max_source_len']}")
ax[1].set_title(f"Article length ({MODEL_TAG} tokens)")
ax[1].legend()
ax[2].hist(np.clip(sum_words, 0, 150), bins=50, color="#C44E52")
ax[2].set_title("Reference summary length (words)")
for a in ax:
    a.set_ylabel("articles")
fig.suptitle(f"CNN/DailyMail training set (random sample of {len(eda):,} articles)")
fig.tight_layout()
FIGURES["lengths"] = "results/figures/fig_lengths.png"
fig.savefig(os.path.join(WORK_DIR, FIGURES["lengths"]), dpi=150)
plt.show()
''')

md(r'''
## 3. Extractive baselines
* **LEAD-3** — the first three sentences. News is written as an *inverted pyramid*, so this is a very strong baseline on CNN/DailyMail.
* **TextRank** (Mihalcea & Tarau, 2004) — PageRank over a sentence graph, edge weight `|Si ∩ Sj| / (log|Si| + log|Sj|)`; top-3 sentences in document order.
* **MiniLM embedding + MMR** — sentences embedded with a transformer encoder (`all-MiniLM-L6-v2`, mean pooling); relevance = cosine similarity to the document centroid, diversity via Maximal Marginal Relevance (λ = 0.7).

ROUGE-1/2/L/Lsum F1 (× 100, Porter stemming) with 95% bootstrap confidence intervals. **ROUGE-Lsum is the summary-level ROUGE-L that papers report as "ROUGE-L" on CNN/DailyMail.**
''')

code(r'''
rng = np.random.default_rng(CFG["seed"])
N_TEST = len(test_df)
NOVELTY_IDX = np.sort(rng.choice(N_TEST, size=min(CFG["novelty_samples"], N_TEST), replace=False))
TEST_ARTICLES = test_df["article_clean"].tolist()
REFS = test_df["reference"].tolist()
PREDS, PER_SAMPLE, RESULTS, TIMING = {}, {}, {}, {}
DISPLAY_NAMES = {"lead3": "LEAD-3", "textrank": "TextRank", "embed_mmr": "MiniLM embedding + MMR",
                 "zero_shot": f"{MODEL_TAG} zero-shot (no fine-tuning)", "finetuned": f"{MODEL_TAG} fine-tuned (ours)"}


def run_system(name, fn):
    t0 = time.time()
    PREDS[name] = fn()
    TIMING[name] = round(time.time() - t0, 1)
    RESULTS[name], PER_SAMPLE[name] = evaluate_system(PREDS[name], REFS, TEST_ARTICLES, NOVELTY_IDX)
    r = RESULTS[name]
    print(f"{DISPLAY_NAMES.get(name, name):<34} R1 {r['rouge1']:6.2f} | R2 {r['rouge2']:6.2f} | RL {r['rougeL']:6.2f} "
          f"| RLsum {r['rougeLsum']:6.2f} | {TIMING[name]:.0f}s", flush=True)


def results_frame(results):
    rows = []
    for k, r in results.items():
        rows.append({"system": DISPLAY_NAMES.get(k, k), "n": r["n"], "ROUGE-1": r["rouge1"], "ROUGE-2": r["rouge2"],
                     "ROUGE-L": r["rougeL"], "ROUGE-Lsum": r["rougeLsum"], "avg words": r["avg_words"],
                     "novel 2-grams %": (r.get("novel_ngrams") or {}).get("2")})
    return pd.DataFrame(rows).set_index("system")


run_system("lead3", lambda: [lead_k(a, 3) for a in TEST_ARTICLES])
run_system("textrank", lambda: [textrank(a, 3) for a in TEST_ARTICLES])
if CFG["run_embed_extractive"]:
    try:
        _ext = EmbeddingExtractor(CFG["embed_model"])
        run_system("embed_mmr", lambda: _ext.summarize_many(TEST_ARTICLES, k=3, verbose=True))
        del _ext
    except Exception as e:                                   # e.g. internet off -> keep going
        print("embedding baseline skipped:", repr(e)[:300])
    gc.collect()
    torch.cuda.empty_cache()
display(results_frame(RESULTS))
''')

md(r'''
## 4. Fine-tuning BART-base
* **Model**: `facebook/bart-base` — 6+6-layer encoder–decoder transformer (≈140 M parameters) pre-trained as a denoising autoencoder.
* **Input / target**: cleaned article truncated to 512 tokens → highlights joined into one paragraph (≤ 128 tokens).
* **Loss**: token-level cross-entropy with label smoothing ε = 0.1 (teacher forcing).
* **Optimiser**: AdamW (lr 5e-5, weight decay 0.01, none on biases/LayerNorm), gradient clipping 1.0, fp16 mixed precision, effective batch 32 (2 × T4 via DataParallel).
* **Time-budgeted learning-rate schedule** (so the run always finishes inside Kaggle's session limit, with a properly annealed LR):

$$\text{lr}(s) = \text{lr}_{\max}\cdot\min\!\Big(1, \frac{s+1}{W}\Big)\cdot\Big(1-\max\big(\tfrac{s}{S},\ \tfrac{t}{T}\big)\Big)$$

with step *s*, warm-up *W*, planned steps *S* (= `epochs` × steps/epoch), elapsed training time *t* and time budget *T* (`train_hours`). Training stops when either fraction reaches 1.
''')

code(r'''
from datasets import Dataset


def to_target(summary):
    """Highlights -> one paragraph of sentences (each ends with punctuation)."""
    lines = [l.strip() for l in clean_summary(summary).split("\n") if l.strip()]
    return " ".join(l if l[-1] in ".!?'\"" else l + "." for l in lines)


def preprocess(batch):
    sources = [PREFIX + clean_article(a) for a in batch["article"]]
    targets = [to_target(s) for s in batch["highlights"]]
    enc = tokenizer(sources, max_length=CFG["max_source_len"], truncation=True)
    enc["labels"] = tokenizer(text_target=targets, max_length=CFG["max_target_len"], truncation=True)["input_ids"]
    return enc


def tokenize(df, name):
    ds = Dataset.from_pandas(df[["article", "highlights"]].reset_index(drop=True), preserve_index=False)
    nproc = NUM_PROC if len(df) > 5000 else None
    print(f"tokenizing {name}: {len(df):,} examples with {nproc or 1} process(es) ...", flush=True)
    try:
        return ds.map(preprocess, batched=True, batch_size=1000, num_proc=nproc, remove_columns=ds.column_names,
                      desc=f"tokenize {name}")
    except Exception as e:                      # safety net: parallel map failed -> single process (~4x slower)
        if not nproc:
            raise
        print(f"parallel tokenization failed ({repr(e)[:200]}) -> retrying in one process", flush=True)
        return ds.map(preprocess, batched=True, batch_size=1000, remove_columns=ds.column_names,
                      desc=f"tokenize {name}")


t0 = time.time()
train_ds = tokenize(train_df, "train")
val_ds = tokenize(val_df, "validation")
_lab = np.array([len(x) for x in train_ds.select(range(min(5000, len(train_ds))))["labels"]])
print(f"tokenized train {len(train_ds):,} / validation {len(val_ds):,} in {time.time() - t0:.0f}s")
print(f"target length: mean {_lab.mean():.1f} tokens, {100 * (_lab >= CFG['max_target_len']).mean():.2f}% truncated")
print("example target:", tokenizer.decode(train_ds[0]["labels"], skip_special_tokens=True)[:300])
''')

code(r'''
from transformers import (AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          TrainerCallback)

model = AutoModelForSeq2SeqLM.from_pretrained(CFG["model_name"])
USE_FP16 = bool(CFG["fp16"]) and N_GPU > 0 and not T5_LIKE          # T5 overflows in fp16
GEN_KW = dict(num_beams=CFG["num_beams"], max_new_tokens=CFG["max_new_tokens"], min_new_tokens=CFG["min_new_tokens"],
              length_penalty=CFG["length_penalty"], no_repeat_ngram_size=CFG["no_repeat_ngram_size"], early_stopping=True)
model.generation_config.update(**GEN_KW)
N_PARAMS = sum(p.numel() for p in model.parameters())

n_dev = max(1, N_GPU)
GRAD_ACCUM = max(1, CFG["effective_batch"] // (CFG["per_device_batch"] * n_dev))
EFFECTIVE_BATCH = CFG["per_device_batch"] * n_dev * GRAD_ACCUM
STEPS_PER_EPOCH = max(1, math.ceil(len(train_ds) / (CFG["per_device_batch"] * n_dev) / GRAD_ACCUM))
TOTAL_STEPS = max(1, int(round(STEPS_PER_EPOCH * CFG["epochs"])))
WARMUP = max(1, min(CFG["warmup_steps"], TOTAL_STEPS // 10))
EVAL_EVERY = max(1, min(CFG["eval_every"], TOTAL_STEPS))


def param_groups(model, weight_decay):
    """AdamW parameter groups: no weight decay for biases and normalisation weights."""
    no_decay = set()
    for mod_name, mod in model.named_modules():
        if isinstance(mod, torch.nn.LayerNorm) or "norm" in type(mod).__name__.lower():
            no_decay.update(f"{mod_name}.{p}" if mod_name else p for p, _ in mod.named_parameters(recurse=False))
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (nodecay if (n in no_decay or n.endswith("bias")) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": nodecay, "weight_decay": 0.0}]


if N_GPU:
    model.to("cuda")
optimizer = torch.optim.AdamW(param_groups(model, CFG["weight_decay"]), lr=CFG["lr"], betas=(0.9, 0.999), eps=1e-8)


class TimeBudget:
    def __init__(self, hours):
        self.seconds, self.t0 = max(1.0, float(hours) * 3600.0), None

    def start(self):
        self.t0 = time.time()

    def elapsed(self):
        return 0.0 if self.t0 is None else time.time() - self.t0

    def fraction(self):
        return self.elapsed() / self.seconds


BUDGET = TimeBudget(CFG["train_hours"])


def lr_multiplier(step):
    """Linear warm-up, then linear decay driven by max(step progress, wall-clock progress)."""
    warm = min(1.0, (step + 1) / WARMUP)
    progress = max(step / TOTAL_STEPS, BUDGET.fraction())
    return warm * max(0.0, 1.0 - progress)


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)


class TimeBudgetCallback(TrainerCallback):
    def __init__(self, budget):
        self.budget, self.stopped_early = budget, False

    def on_train_begin(self, args, state, control, **kwargs):
        self.budget.start()

    def on_step_end(self, args, state, control, **kwargs):
        if self.budget.fraction() >= 1.0 and state.global_step < state.max_steps:
            control.should_training_stop = True
            self.stopped_early = True
            print(f"time budget ({CFG['train_hours']} h) reached at step {state.global_step}/{state.max_steps} -> stop")
        return control


class LogPrinter(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        logs, el = logs or {}, BUDGET.elapsed()
        if "loss" in logs:
            frac = max(state.global_step / max(1, state.max_steps), BUDGET.fraction())
            eta = el / frac - el if frac > 0 else float("nan")
            print(f"step {state.global_step:>6}/{state.max_steps} | epoch {state.epoch or 0:5.2f} | loss {logs['loss']:.4f} "
                  f"| lr {logs.get('learning_rate', float('nan')):.2e} | {el / 60:6.1f} min | ETA {eta / 60:6.1f} min",
                  flush=True)
        if "eval_loss" in logs:
            print(f"   >> step {state.global_step}: eval_loss {logs['eval_loss']:.4f}", flush=True)


def make_training_args(**kw):
    """Seq2SeqTrainingArguments that work across transformers 4.4x-5.x (renamed / removed fields)."""
    fields = {f.name for f in dataclasses.fields(Seq2SeqTrainingArguments)}
    if "eval_strategy" in kw and "eval_strategy" not in fields:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    ignored = sorted(k for k in kw if k not in fields)
    if ignored:
        print("TrainingArguments: ignoring unsupported", ignored)
    return Seq2SeqTrainingArguments(**{k: v for k, v in kw.items() if k in fields})


training_args = make_training_args(
    output_dir=CKPT_DIR, per_device_train_batch_size=CFG["per_device_batch"],
    per_device_eval_batch_size=CFG["per_device_batch"] * 2, gradient_accumulation_steps=GRAD_ACCUM,
    max_steps=TOTAL_STEPS, learning_rate=CFG["lr"], weight_decay=CFG["weight_decay"], max_grad_norm=CFG["max_grad_norm"],
    label_smoothing_factor=CFG["label_smoothing"], fp16=USE_FP16, eval_strategy="steps", eval_steps=EVAL_EVERY,
    eval_on_start=True, logging_strategy="steps", logging_steps=CFG["log_every"], logging_first_step=True,
    save_strategy="no", report_to="none", disable_tqdm=True, predict_with_generate=False,
    dataloader_num_workers=NUM_WORKERS, seed=CFG["seed"], remove_unused_columns=True)

collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100,
                                  pad_to_multiple_of=8 if USE_FP16 else None)
budget_cb = TimeBudgetCallback(BUDGET)
trainer_kw = dict(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator,
                  optimizers=(optimizer, scheduler), callbacks=[budget_cb, LogPrinter()])
_sig = inspect.signature(Seq2SeqTrainer.__init__).parameters
trainer_kw["processing_class" if "processing_class" in _sig else "tokenizer"] = tokenizer
trainer = Seq2SeqTrainer(**trainer_kw)
try:
    from transformers.trainer_callback import PrinterCallback
    trainer.remove_callback(PrinterCallback)                 # LogPrinter replaces the raw dict printer
except Exception:
    pass

print(f"model {CFG['model_name']}: {N_PARAMS / 1e6:.1f} M parameters | fp16 {USE_FP16} | GPUs {N_GPU}")
print(f"batch {CFG['per_device_batch']} x {n_dev} device(s) x {GRAD_ACCUM} accumulation = {EFFECTIVE_BATCH}")
print(f"{STEPS_PER_EPOCH:,} steps/epoch | planned {TOTAL_STEPS:,} steps ({CFG['epochs']} epochs) | warm-up {WARMUP} "
      f"| time budget {CFG['train_hours']} h | eval every {EVAL_EVERY} steps")
''')

code(r'''
t_train = time.time()
train_output = trainer.train()
TRAIN_SECONDS = time.time() - t_train
final_eval = trainer.evaluate()
state = trainer.state
TRAIN_INFO = dict(
    params_million=round(N_PARAMS / 1e6, 1), n_gpu=N_GPU, gpu=ENV["gpu"], fp16=USE_FP16,
    per_device_batch=CFG["per_device_batch"], grad_accum=GRAD_ACCUM, effective_batch=EFFECTIVE_BATCH,
    epochs_planned=CFG["epochs"], steps_per_epoch=STEPS_PER_EPOCH, total_steps_planned=TOTAL_STEPS,
    steps_completed=state.global_step, epochs_completed=round(state.global_step / STEPS_PER_EPOCH, 3),
    samples_seen=state.global_step * EFFECTIVE_BATCH, train_examples=len(train_ds), val_examples=len(val_ds),
    warmup_steps=WARMUP, lr=CFG["lr"], weight_decay=CFG["weight_decay"], label_smoothing=CFG["label_smoothing"],
    max_grad_norm=CFG["max_grad_norm"], max_source_len=CFG["max_source_len"], max_target_len=CFG["max_target_len"],
    time_budget_hours=CFG["train_hours"], stopped_by_time_budget=budget_cb.stopped_early,
    train_runtime_hours=round(TRAIN_SECONDS / 3600, 3), final_train_loss=round(float(train_output.training_loss), 4),
    final_eval_loss=round(float(final_eval["eval_loss"]), 4),
    throughput_samples_per_s=round(state.global_step * EFFECTIVE_BATCH / max(TRAIN_SECONDS, 1e-9), 2),
    log_history=state.log_history)
print(json.dumps({k: v for k, v in TRAIN_INFO.items() if k != "log_history"}, indent=1))
''')

md(r'''
## 5. Save the model
Saved in the standard **Hugging Face format** (`model.safetensors` + tokenizer + `generation_config.json`) plus `summarizer_config.json` (input prefix, max input length, decoding settings) used by `src/summarizer.py`. The folder is also zipped for a one-click download. (Why not a `.pkl`? See the FAQ in the README: pickles execute code when loaded and break across library versions; safetensors is the portable, safe format.)
''')

code(r'''
def zip_dir(folder):
    path = folder + ".zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:     # safetensors do not compress; STORED is fast
        for root, _, files in os.walk(folder):
            for f in sorted(files):
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, os.path.dirname(folder)))
    return path


os.makedirs(MODEL_DIR, exist_ok=True)
trainer.model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
SUMMARIZER_META = dict(
    base_model=CFG["model_name"], prefix=PREFIX, clean_input=True, max_source_len=CFG["max_source_len"],
    max_target_len=CFG["max_target_len"], generation=GEN_KW, run_mode=RUN_MODE,
    dataset="CNN/DailyMail 3.0.0 (Kaggle: gowrishankarp/newspaper-text-summarization-cnn-dailymail)",
    steps=TRAIN_INFO["steps_completed"], samples_seen=TRAIN_INFO["samples_seen"],
    train_hours=TRAIN_INFO["train_runtime_hours"], created=datetime.datetime.now().isoformat(timespec="seconds"),
    project="https://github.com/ksidharth8/text-summarization-transformers")
with open(os.path.join(MODEL_DIR, "summarizer_config.json"), "w") as f:
    json.dump(SUMMARIZER_META, f, indent=2)
MODEL_ZIP = zip_dir(MODEL_DIR)
for f in sorted(os.listdir(MODEL_DIR)):
    print(f"  {f:<28} {os.path.getsize(os.path.join(MODEL_DIR, f)) / 1e6:9.2f} MB")
print(f"saved -> {MODEL_DIR}\nzip   -> {MODEL_ZIP} ({os.path.getsize(MODEL_ZIP) / 1e6:.0f} MB)")
''')

code(r'''
hist = pd.DataFrame(TRAIN_INFO["log_history"])
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
if "loss" in hist:
    tr = hist.dropna(subset=["loss"])
    ax[0].plot(tr["step"], tr["loss"], label="train loss", color="#4C72B0", alpha=0.8)
if "eval_loss" in hist:
    ev = hist.dropna(subset=["eval_loss"])
    ax[0].plot(ev["step"], ev["eval_loss"], "o-", label="validation loss", color="#C44E52")
ax[0].set_xlabel("optimizer step")
ax[0].set_ylabel("label-smoothed cross-entropy")
ax[0].set_title("Loss")
ax[0].legend()
if "learning_rate" in hist:
    lr_ = hist.dropna(subset=["learning_rate"])
    ax[1].plot(lr_["step"], lr_["learning_rate"], color="#55A868")
ax[1].set_xlabel("optimizer step")
ax[1].set_title("Learning rate (time-budgeted schedule)")
fig.tight_layout()
FIGURES["training_curves"] = "results/figures/fig_training_curves.png"
fig.savefig(os.path.join(WORK_DIR, FIGURES["training_curves"]), dpi=150)
plt.show()

del trainer, model, optimizer, scheduler, train_ds, val_ds, train_df
gc.collect()
torch.cuda.empty_cache()
''')

md(r'''
## 6. Evaluation on the test set
The **saved** model is reloaded (this tests exactly the artifact you download) and summarizes every test article with beam search (4 beams, 56–142 new tokens, length penalty 2.0, no repeated trigrams). For comparison, the *untouched* pre-trained model is run zero-shot on a random subset.
''')

code(r'''
summarizer = Summarizer(MODEL_DIR)
print(f"loaded {MODEL_DIR} on {summarizer.device} | max_source_len {summarizer.max_source_len} | {summarizer.generation}")
_n_batches = math.ceil(N_TEST / CFG["gen_batch"])
run_system("finetuned", lambda: summarizer.summarize(TEST_ARTICLES, batch_size=CFG["gen_batch"], adaptive_length=False,
                                                     progress_every=max(1, _n_batches // 20)))
GEN_INFO = dict(GEN_KW, batch_size=CFG["gen_batch"], seconds=TIMING["finetuned"],
                docs_per_second=round(N_TEST / max(TIMING["finetuned"], 1e-9), 2))
''')

code(r'''
ZS_IDX = np.sort(rng.choice(N_TEST, size=min(CFG["zero_shot_samples"], N_TEST), replace=False))
ZS_PREDS = []
try:
    _zs = Summarizer(CFG["model_name"], max_source_len=CFG["max_source_len"], generation=GEN_KW)
    ZS_PREDS = _zs.summarize([TEST_ARTICLES[i] for i in ZS_IDX], batch_size=CFG["gen_batch"], adaptive_length=False)
    RESULTS["zero_shot"], PER_SAMPLE["zero_shot"] = evaluate_system(
        ZS_PREDS, [REFS[i] for i in ZS_IDX], [TEST_ARTICLES[i] for i in ZS_IDX])
    del _zs
except Exception as e:
    print("zero-shot run skipped:", repr(e)[:300])
gc.collect()
torch.cuda.empty_cache()
ZS_SUBSET = {k: {t: round(float(PER_SAMPLE[k][t][ZS_IDX].mean()), 2) for t in ROUGE_TYPES}
             for k in ("lead3", "finetuned") if k in PER_SAMPLE}
if "zero_shot" in ZS_SUBSET:
    ZS_SUBSET["zero_shot"] = {t: RESULTS["zero_shot"][t] for t in ROUGE_TYPES}
print(f"same {len(ZS_IDX)}-article subset:")
display(pd.DataFrame(ZS_SUBSET).T.rename(index=DISPLAY_NAMES))
''')

code(r'''
SIGNIFICANCE = {}
for base in ("lead3", "textrank", "embed_mmr"):
    if base in PER_SAMPLE:
        SIGNIFICANCE[f"finetuned_vs_{base}"] = {t: paired_bootstrap(PER_SAMPLE["finetuned"][t], PER_SAMPLE[base][t])
                                                for t in ROUGE_TYPES}
display(Markdown(f"### Main results — CNN/DailyMail test set (n = {N_TEST:,})"))
display(results_frame(RESULTS))
for name, sig in SIGNIFICANCE.items():
    print(f"{name}: " + " | ".join(f"{t} {v['diff']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] p={v['p_value']}"
                                 for t, v in sig.items()))

# ROUGE-1 by article length (quartiles): shows the effect of truncating long inputs to max_source_len
wc = np.array([len(a.split()) for a in TEST_ARTICLES])
edges = np.unique(np.quantile(wc, [0, 0.25, 0.5, 0.75, 1.0]))
buckets = np.clip(np.searchsorted(edges, wc, side="right") - 1, 0, max(0, len(edges) - 2))
BY_LENGTH = []
for b in range(max(0, len(edges) - 1)):
    m = buckets == b
    if m.sum() == 0:
        continue
    row = {"bucket": f"{int(edges[b])}-{int(edges[b + 1])} words", "n": int(m.sum())}
    for k in ("lead3", "textrank", "finetuned"):
        if k in PER_SAMPLE:
            row[k] = round(float(PER_SAMPLE[k]["rouge1"][m].mean()), 2)
    BY_LENGTH.append(row)
display(pd.DataFrame(BY_LENGTH))
''')

code(r'''
SYSTEMS = [k for k in ("lead3", "textrank", "embed_mmr", "zero_shot", "finetuned") if k in RESULTS]
COLORS = {"lead3": "#8C8C8C", "textrank": "#4C72B0", "embed_mmr": "#55A868", "zero_shot": "#DD8452", "finetuned": "#C44E52"}
fig, ax = plt.subplots(figsize=(12, 4.5))
mets = [("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"), ("rougeLsum", "ROUGE-Lsum")]
w, x = 0.8 / len(SYSTEMS), np.arange(len(mets))
for i, k in enumerate(SYSTEMS):
    vals = [RESULTS[k][m] for m, _ in mets]
    err = np.array([[RESULTS[k][m] - RESULTS[k][m + "_ci"][0], RESULTS[k][m + "_ci"][1] - RESULTS[k][m]] for m, _ in mets]).T
    bars = ax.bar(x + (i - (len(SYSTEMS) - 1) / 2) * w, vals, w, yerr=np.abs(err), capsize=3, color=COLORS[k],
                  label=f"{DISPLAY_NAMES[k]} (n={RESULTS[k]['n']:,})")
    ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
ax.set_xticks(x, [n for _, n in mets])
ax.set_ylabel("F1 × 100")
ax.set_title("ROUGE on the CNN/DailyMail test set (error bars: 95% bootstrap CI)")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
FIGURES["rouge"] = "results/figures/fig_rouge_comparison.png"
fig.savefig(os.path.join(WORK_DIR, FIGURES["rouge"]), dpi=150)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ns = ["1", "2", "3", "4"]
ax[0].plot(ns, [DATA_STATS["reference_novel_ngrams"][n] for n in ns], "k--o", label="reference summaries")
for k in SYSTEMS:
    if RESULTS[k].get("novel_ngrams"):
        ax[0].plot(ns, [RESULTS[k]["novel_ngrams"][n] for n in ns], "o-", color=COLORS[k], label=DISPLAY_NAMES[k])
ax[0].set_xlabel("n-gram order")
ax[0].set_ylabel("% of summary n-grams not in the article")
ax[0].set_title("Abstractiveness (novel n-grams)")
ax[0].legend(fontsize=8)
if BY_LENGTH:
    bl = pd.DataFrame(BY_LENGTH)
    xs = np.arange(len(bl))
    cols = [k for k in ("lead3", "textrank", "finetuned") if k in bl]
    for i, k in enumerate(cols):
        ax[1].bar(xs + (i - (len(cols) - 1) / 2) * 0.27, bl[k], 0.27, color=COLORS[k], label=DISPLAY_NAMES[k])
    ax[1].set_xticks(xs, bl["bucket"], fontsize=8)
    ax[1].set_ylabel("ROUGE-1")
    ax[1].set_title("ROUGE-1 by article length (test quartiles)")
    ax[1].legend(fontsize=8)
fig.tight_layout()
FIGURES["novelty_length"] = "results/figures/fig_novelty_and_length.png"
fig.savefig(os.path.join(WORK_DIR, FIGURES["novelty_length"]), dpi=150)
plt.show()
''')

md(r'''
## 7. Qualitative examples and custom test cases
Random test articles with the reference and system summaries (per-example ROUGE in brackets), then texts that are **not** from CNN/DailyMail (`test_cases/`), including a research-paper-style document longer than the 512-token input window, which is summarized hierarchically (chunk → partial summaries → final summary).
''')

code(r'''
EX_IDX = rng.choice(N_TEST, size=min(CFG["n_examples"], N_TEST), replace=False)
SAMPLES, md_lines = [], []
for n, i in enumerate(map(int, EX_IDX), 1):
    item = {"id": str(test_df["id"].iloc[i]), "article_excerpt": " ".join(TEST_ARTICLES[i].split()[:120]) + " ...",
            "reference": REFS[i]}
    md_lines.append(f"#### Example {n} (id `{item['id']}`)\n**Article (first 120 words):** {item['article_excerpt']}\n\n"
                    f"**Reference:** {REFS[i].replace(chr(10), ' ')}\n")
    for k in ("lead3", "textrank", "finetuned"):
        if k in PREDS:
            sc = {t: round(float(PER_SAMPLE[k][t][i]), 1) for t in ("rouge1", "rouge2", "rougeLsum")}
            item[k] = dict(summary=PREDS[k][i], **sc)
            md_lines.append(f"**{DISPLAY_NAMES[k]}** [R1 {sc['rouge1']} / R2 {sc['rouge2']} / RLsum {sc['rougeLsum']}]: "
                            f"{PREDS[k][i].replace(chr(10), ' ')}\n")
    SAMPLES.append(item)
display(Markdown("\n".join(md_lines)))

pred_df = pd.DataFrame({"id": test_df["id"].astype(str), "reference": REFS, **{k: PREDS[k] for k in PREDS}})
pred_df.to_csv(os.path.join(RESULTS_DIR, "test_predictions.csv"), index=False)
print("saved results/test_predictions.csv", pred_df.shape)
''')

code(test_cases_code() + r'''

for _name, _text in TEST_CASES.items():
    with open(os.path.join(WORK_DIR, "test_cases", f"{_name}.txt"), "w", encoding="utf-8") as f:
        f.write(_text + "\n")

CUSTOM, md_lines = [], []
_budget = summarizer.max_source_len - summarizer.count_tokens(summarizer.prefix) - 8
for name, text in TEST_CASES.items():
    n_tok = summarizer.count_tokens(summarizer.prepare(text))
    t0 = time.time()
    if n_tok > _budget:
        out = summarizer.summarize_long(text)
        abstractive, mode = out["summary"], f"long-document map-reduce: {out['chunks']} chunks, {out['rounds']} round(s)"
    else:
        out, abstractive, mode = None, summarizer.summarize(text), "single pass, adaptive length"
    secs = time.time() - t0
    nov = novel_ngram_ratio([text], [abstractive], ns=(1, 2))
    item = dict(name=name, words=count_words(text), tokens=n_tok, mode=mode, seconds=round(secs, 2),
                abstractive=abstractive, textrank=textrank(text, 3), lead3=lead_k(text, 3),
                novel_unigrams_pct=nov["1"], novel_bigrams_pct=nov["2"])
    if out:
        item["partial_summaries"] = out["partial_summaries"]
    CUSTOM.append(item)
    md_lines.append(f"### {name}\n*{item['words']} words / {n_tok} tokens — {mode} — {secs:.1f}s — "
                    f"novel bigrams {nov['2']}%*\n\n**Abstractive ({MODEL_TAG} fine-tuned):** {abstractive}\n\n"
                    f"**TextRank:** {item['textrank'].replace(chr(10), ' ')}\n\n**LEAD-3:** {item['lead3'].replace(chr(10), ' ')}\n")
    if out:
        md_lines.append("<details><summary>partial (chunk) summaries</summary>\n\n" +
                        "\n".join(f"{j}. {p}" for j, p in enumerate(out["partial_summaries"][0], 1)) + "\n</details>\n")
CUSTOM_MD = "\n".join(md_lines)
with open(os.path.join(RESULTS_DIR, "custom_cases.md"), "w", encoding="utf-8") as f:
    f.write(f"# Custom test cases — {MODEL_TAG} fine-tuned on CNN/DailyMail\n\n" + CUSTOM_MD)
display(Markdown(CUSTOM_MD))
''')

md(r'''
## 8. Save metrics, model card and outputs
`results/metrics.json` contains everything (configuration, environment, data statistics, training log, all scores, significance tests, examples, custom cases). It is the input for the project report.
''')

code(r'''
def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    return o


METRICS = dict(
    project=dict(title="Text Summarization with Transformers (CNN/DailyMail)", author="Abhinav Anand",
                 repo="https://github.com/ksidharth8/text-summarization-transformers"),
    run_mode=RUN_MODE, timestamp=datetime.datetime.now().isoformat(timespec="seconds"), model_name=CFG["model_name"],
    config=CFG, environment=ENV,
    data=dict(files=DATA_FILES, split_sizes=SPLIT_SIZES, dropped=DROPPED,
              used=dict(train=TRAIN_INFO["train_examples"], validation=TRAIN_INFO["val_examples"], test=N_TEST),
              stats=DATA_STATS, cleaning_example=CLEAN_EXAMPLE),
    training=TRAIN_INFO, generation=GEN_INFO, results=RESULTS, significance=SIGNIFICANCE,
    zero_shot_subset=dict(n=len(ZS_IDX), scores=ZS_SUBSET), rouge1_by_article_length=BY_LENGTH,
    timing_seconds=TIMING, samples=SAMPLES, custom_cases=CUSTOM, figures=FIGURES,
    total_runtime_hours=round((time.time() - T_NOTEBOOK) / 3600, 3))
with open(os.path.join(RESULTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(to_jsonable(METRICS), f, indent=2, ensure_ascii=False)

rows = ["| System | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | avg. words |", "|---|---|---|---|---|---|"]
for k in SYSTEMS:
    r = RESULTS[k]
    name = DISPLAY_NAMES[k] + (f" (n={r['n']})" if r["n"] != N_TEST else "")
    rows.append(f"| {name} | {r['rouge1']:.2f} | {r['rouge2']:.2f} | {r['rougeL']:.2f} | {r['rougeLsum']:.2f} | {r['avg_words']} |")
RESULTS_TABLE = "\n".join(rows)
with open(os.path.join(RESULTS_DIR, "results_table.md"), "w") as f:
    f.write(RESULTS_TABLE + "\n")

ft = RESULTS["finetuned"]
card = f"""---
language: en
license: apache-2.0
base_model: {CFG['model_name']}
tags: [summarization, bart, cnn_dailymail]
datasets: [cnn_dailymail]
---
# {MODEL_TAG} fine-tuned on CNN/DailyMail

Abstractive news summarizer from the project
[text-summarization-transformers](https://github.com/ksidharth8/text-summarization-transformers) (Abhinav Anand, BIT Mesra).

**Test set ({N_TEST:,} articles):** ROUGE-1 {ft['rouge1']:.2f} · ROUGE-2 {ft['rouge2']:.2f} · ROUGE-L {ft['rougeL']:.2f} · ROUGE-Lsum {ft['rougeLsum']:.2f}

Training: {TRAIN_INFO['steps_completed']:,} steps × batch {EFFECTIVE_BATCH} ({TRAIN_INFO['epochs_completed']} epochs, {TRAIN_INFO['train_runtime_hours']} h on {ENV['gpu']}),
input ≤ {CFG['max_source_len']} tokens, AdamW lr {CFG['lr']}, label smoothing {CFG['label_smoothing']}, fp16. Run mode: `{RUN_MODE}`.

```python
from transformers import pipeline
summarize = pipeline("summarization", model="PATH_OR_HUB_ID")
print(summarize(article, num_beams=4, max_new_tokens=142, min_new_tokens=56, length_penalty=2.0,
                no_repeat_ngram_size=3, truncation=True)[0]["summary_text"])
```
or, with long-document support, `src.summarizer.Summarizer("PATH")` from the repository.
"""
with open(os.path.join(MODEL_DIR, "README.md"), "w", encoding="utf-8") as f:
    f.write(card)
with zipfile.ZipFile(MODEL_ZIP, "a") as zf:
    zf.write(os.path.join(MODEL_DIR, "README.md"), os.path.join(os.path.basename(MODEL_DIR), "README.md"))
display(Markdown(RESULTS_TABLE))
print("saved results/metrics.json, results/results_table.md and the model card")
''')

code(r'''
# Optional: publish the model on the Hugging Face Hub (Kaggle: Add-ons -> Secrets -> HF_TOKEN, then push_to_hub=True)
if CFG["push_to_hub"] and RUN_MODE == "full":
    try:
        from kaggle_secrets import UserSecretsClient
        from huggingface_hub import HfApi
        token = UserSecretsClient().get_secret("HF_TOKEN")
        api = HfApi(token=token)
        api.create_repo(CFG["hub_repo"], exist_ok=True)
        api.upload_folder(folder_path=MODEL_DIR, repo_id=CFG["hub_repo"], commit_message="fine-tuned on CNN/DailyMail")
        print("pushed to https://huggingface.co/" + CFG["hub_repo"])
    except Exception as e:
        print("push to hub failed:", repr(e)[:300])
else:
    print("push_to_hub disabled")
''')

code(r'''
shutil.rmtree(CKPT_DIR, ignore_errors=True)
print(f"total runtime: {(time.time() - T_NOTEBOOK) / 3600:.2f} h\n\nOutputs in {WORK_DIR}:")
for root, dirs, files in os.walk(WORK_DIR):
    dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
    depth = root[len(WORK_DIR):].count(os.sep)
    if depth > 2:
        continue
    for f in sorted(files):
        p = os.path.join(root, f)
        print(f"  {os.path.relpath(p, WORK_DIR):<60} {os.path.getsize(p) / 1e6:9.2f} MB")
if RUN_MODE == "smoke":
    print("\n" + "!" * 90 + "\nSMOKE RUN FINISHED - pipeline works, but the scores are meaningless (tiny training run).\n"
          "Now set RUN_MODE = \"full\" and use 'Save Version -> Save & Run All (Commit)'.\n" + "!" * 90)
''')

# =============================================================================================
# 02 - test notebook
# =============================================================================================
TEST = []
md2 = lambda s: TEST.append(("md", s.strip()))      # noqa: E731
code2 = lambda s: TEST.append(("code", s.strip()))  # noqa: E731

md2(r'''
# Test the trained summarizer

Loads the model produced by `01_train_bart_cnn_dailymail.ipynb` and summarizes custom texts (news, a research abstract and a long research-paper-style document) with the abstractive model, TextRank and LEAD-3 side by side.

**Where the model is found (automatically):**
* **Kaggle:** create a new notebook → *Add Input* → *Notebook Output Files* → select your training notebook. The model folder, `src/` and `test_cases/` are all in its output.
* **Local / Colab:** clone the repo, unzip `bart-base-cnn-summarizer.zip` into `models/`, open this notebook from `notebooks/`.
* Or set the environment variable `SUMM_MODEL_DIR` / the variable `MODEL_DIR` below.

CPU is fine for a few texts (≈5–20 s per summary); a GPU is faster.
''')

code2(r'''
import os, re, sys, glob, json, time, zipfile, subprocess

MODEL_DIR = os.environ.get("SUMM_MODEL_DIR")   # None -> auto-detect
FALLBACK_MODEL = None                          # e.g. "facebook/bart-large-cnn" to try the notebook before your model exists
REPO_URL = "https://github.com/ksidharth8/text-summarization-transformers"
SEARCH_ROOTS = ["/kaggle/input", "/kaggle/working", "models", "../models", "outputs", "../outputs"]


def find_model_dir():
    for r in SEARCH_ROOTS:
        hits = sorted(glob.glob(os.path.join(r, "**", "summarizer_config.json"), recursive=True))
        if hits:
            return os.path.dirname(os.path.abspath(hits[0]))
    for r in SEARCH_ROOTS:                                    # only the zip is there -> extract it
        for z in sorted(glob.glob(os.path.join(r, "**", "*-cnn-summarizer.zip"), recursive=True)):
            dest = os.path.abspath("models")
            os.makedirs(dest, exist_ok=True)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest)
            hits = glob.glob(os.path.join(dest, "**", "summarizer_config.json"), recursive=True)
            if hits:
                return os.path.dirname(hits[0])
    return None


def find_src_root(model_dir):
    cands = [os.environ.get("SUMM_SRC_ROOT"), "..", ".", os.path.dirname(model_dir) if model_dir else None, "/kaggle/working"]
    cands += [os.path.dirname(os.path.dirname(p)) for p in glob.glob("/kaggle/input/**/src/summarizer.py", recursive=True)]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "src", "summarizer.py")):
            return os.path.abspath(c)
    if not os.path.isdir("text-summarization-transformers"):
        subprocess.run(["git", "clone", "-q", REPO_URL], check=False)
    if os.path.isfile("text-summarization-transformers/src/summarizer.py"):
        return os.path.abspath("text-summarization-transformers")
    raise FileNotFoundError("src/ package not found: run this notebook from the repo's notebooks/ folder, "
                            "or attach the training notebook's output on Kaggle.")


MODEL_DIR = MODEL_DIR or find_model_dir() or FALLBACK_MODEL
if MODEL_DIR is None:
    raise FileNotFoundError("No trained model found. Attach the training notebook output (Kaggle) or unzip "
                            "bart-base-cnn-summarizer.zip into models/ - or set FALLBACK_MODEL.")
SRC_ROOT = find_src_root(MODEL_DIR if os.path.isdir(str(MODEL_DIR)) else None)
sys.path.insert(0, SRC_ROOT)
print("model:", MODEL_DIR, "\nsrc  :", SRC_ROOT)
''')

code2(r'''
import torch
import transformers
from IPython.display import Markdown, display
transformers.logging.set_verbosity_error()

from src.summarizer import Summarizer
from src.extractive import textrank, lead_k
from src.metrics import novel_ngram_ratio

summarizer = Summarizer(MODEL_DIR)
print(f"{summarizer.num_parameters / 1e6:.1f} M parameters on {summarizer.device} | max input {summarizer.max_source_len} tokens")
print("decoding:", summarizer.generation)
if os.path.isfile(os.path.join(str(MODEL_DIR), "summarizer_config.json")):
    meta = json.load(open(os.path.join(MODEL_DIR, "summarizer_config.json")))
    print("trained:", {k: meta.get(k) for k in ("base_model", "run_mode", "steps", "samples_seen", "train_hours", "created")})
''')

code2(test_cases_code() + r'''

# prefer the files in the repository / training output if present (so you can add your own .txt files)
_files = sorted(glob.glob(os.path.join(SRC_ROOT, "test_cases", "*.txt")))
if _files:
    TEST_CASES = {os.path.splitext(os.path.basename(p))[0]: open(p, encoding="utf-8").read().strip() for p in _files}
print(f"{len(TEST_CASES)} test cases:", list(TEST_CASES))
''')

code2(r'''
def compare(text, name="input"):
    n_tok = summarizer.count_tokens(summarizer.prepare(text))
    budget = summarizer.max_source_len - summarizer.count_tokens(summarizer.prefix) - 8
    t0 = time.time()
    if n_tok > budget:
        out = summarizer.summarize_long(text)
        abstractive, mode = out["summary"], f"long-document map-reduce ({out['chunks']} chunks, {out['rounds']} round(s))"
    else:
        out, abstractive, mode = None, summarizer.summarize(text), "single pass"
    secs = time.time() - t0
    nov = novel_ngram_ratio([text], [abstractive], ns=(1, 2))
    display(Markdown(
        f"### {name}\n*{len(text.split())} words / {n_tok} tokens — {mode} — {secs:.1f}s — novel bigrams {nov['2']}%*\n\n"
        f"**Abstractive:** {abstractive}\n\n**TextRank:** {textrank(text, 3).replace(chr(10), ' ')}\n\n"
        f"**LEAD-3:** {lead_k(text, 3).replace(chr(10), ' ')}\n"))
    return out


LONG_RESULTS = {}
for name, text in TEST_CASES.items():
    out = compare(text, name)
    if out:
        LONG_RESULTS[name] = out
''')

code2(r'''
# Long-document mode in detail: chunk summaries (map) -> final summary (reduce)
for name, out in LONG_RESULTS.items():
    print(f"{name}: {out['input_tokens']} tokens -> {out['chunks']} chunks, {out['rounds']} round(s)")
    for r, partials in enumerate(out["partial_summaries"], 1):
        print(f"\n  round {r}:")
        for j, p in enumerate(partials, 1):
            print(f"   [{j}] {p}")
    print("\n  FINAL:", out["summary"])
''')

code2(r'''
# Try your own text -------------------------------------------------------------------------------
MY_TEXT = """Paste any news article, report or paper section here. Longer texts are summarized in several
chunks automatically. The summary length adapts to the input length for short texts."""
_ = compare(MY_TEXT, "my text")

# Decoding can be changed per call, e.g. shorter and more focused:
# summarizer.summarize(MY_TEXT, num_beams=6, max_new_tokens=60, min_new_tokens=20, length_penalty=1.0)
''')

code2(r'''
# Optional: quick ROUGE check on a few CNN/DailyMail test articles (if the dataset is attached)
EVAL_SAMPLES = 100
_root = os.environ.get("SUMM_DATA_DIR", "/kaggle/input")
_csvs = [p for p in glob.glob(os.path.join(_root, "**", "*.csv"), recursive=True) if "test" in os.path.basename(p).lower()]
_csvs = [p for p in _csvs if re.search(r"cnn|dailymail", p, re.I)] or _csvs
try:
    import re
    import pandas as pd
    from src.metrics import evaluate_system
    from src.preprocessing import clean_summary
    df = pd.read_csv(_csvs[0]).dropna(subset=["article", "highlights"])
    df = df.sample(n=min(EVAL_SAMPLES, len(df)), random_state=0)
    refs = [clean_summary(h) for h in df["highlights"]]
    preds = summarizer.summarize(df["article"].tolist(), batch_size=16, adaptive_length=False)
    res, _ = evaluate_system(preds, refs)
    print(f"{len(df)} random test articles: " + " | ".join(f"{k} {res[k]:.2f}" for k in ("rouge1", "rouge2", "rougeL", "rougeLsum")))
except (IndexError, KeyError, FileNotFoundError) as e:
    print("CNN/DailyMail test CSV not attached - skipping the ROUGE check", repr(e)[:120])
''')


def main():
    os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
    for fname, cells in (("01_train_bart_cnn_dailymail.ipynb", TRAIN), ("02_test_summarizer.ipynb", TEST)):
        path = os.path.join(ROOT, "notebooks", fname)
        nb = notebook(cells)
        nbformat.validate(nb)
        nbformat.write(nb, path)
        print(f"wrote {os.path.relpath(path, ROOT)} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
