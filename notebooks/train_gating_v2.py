#!/usr/bin/env python3
"""
AdaptRoute Gating Network Training — H200 optimised, 5-class version.

Trains a DistilBERT + LoRA classifier with 5 classes:
  0: code     →  route to lora-code
  1: math     →  route to lora-math
  2: qa       →  route to lora-qa
  3: medical  →  route to lora-medical
  4: general  →  use base model (no adapter)

The "general" class is synthetic — built from diverse sources so the model
learns to say "this doesn't belong to any specialist domain."

Usage (on H200 SSH server):
    export HF_TOKEN="hf_..."
    python train_gating_v2.py

Compatible with Python 3.11.
"""

import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, login
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

# ==============================================================================
# 1. Configuration
# ==============================================================================
HF_USERNAME   = "kunjcr2"
HF_REPO_NAME  = "gating-bert-adaptroute"


MODEL_CHECKPOINT = "distilbert-base-uncased"
MAX_LENGTH       = 128

# LoRA config — same as original
LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.1
LORA_TARGET_MODULES = ["q_lin", "k_lin", "v_lin"]

# H200 optimised — larger batch, bf16
BATCH_SIZE    = 128       # H200 has plenty of VRAM, 4x the original
NUM_EPOCHS    = 5
LEARNING_RATE = 2e-4
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.06
VAL_SPLIT     = 0.1
TEST_SPLIT    = 0.1
SEED          = 42
OUTPUT_DIR    = "./gate-checkpoint-v2"

# 5 classes now (added "general")
LABEL2ID = {"code": 0, "math": 1, "qa": 2, "medical": 3, "general": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
FULL_REPO_ID = f"{HF_USERNAME}/{HF_REPO_NAME}"

# Paths — relative to this script
SCRIPT_DIR   = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR.parent / "datasets"

# Data files for the 4 existing domains
DATA_FILES = {
    "code":    DATASETS_DIR / "code.json",
    "math":    DATASETS_DIR / "math.json",
    "qa":      DATASETS_DIR / "qa.json",
    "medical": DATASETS_DIR / "medical.json",
}

# ==============================================================================
# 2. Authentication
# ==============================================================================
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("✓ HuggingFace login OK")
else:
    print("⚠ HF_TOKEN not set — will not push to hub")



# ==============================================================================
# 3. Load & Build Dataset (with synthetic "general" class)
# ==============================================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_json_file(path: Path) -> tuple:
    """Load a JSON data file → (texts, labels)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts, labels = [], []
    for record in data:
        text  = str(record["user"]).strip()
        label = int(record["label"])
        if text:
            texts.append(text)
            labels.append(label)
    return texts, labels


def build_general_class(target_n: int) -> List[str]:
    """
    Build a synthetic 'general' class from diverse HuggingFace datasets.
    These are everyday questions that DON'T belong to code/math/qa/medical.
    """
    print(f"\n  Building 'general' class ({target_n:,} samples) ...")
    general_texts = []

    # Source 1: Alpaca instructions — very diverse everyday tasks
    print("    ↓ Loading tatsu-lab/alpaca ...")
    try:
        alpaca = load_dataset("tatsu-lab/alpaca", split="train", trust_remote_code=True)
        for rec in alpaca:
            instruction = (rec.get("instruction") or "").strip()
            _input = (rec.get("input") or "").strip()
            text = f"{instruction} {_input}".strip() if _input else instruction
            if text and len(text) > 20:
                general_texts.append(text)
            if len(general_texts) >= target_n * 2:
                break
        print(f"    ✓ Got {len(general_texts):,} from alpaca")
    except Exception as e:
        print(f"    ⚠ alpaca failed: {e}")

    # Source 2: SQuAD (Stanford QA) — general knowledge questions
    print("    ↓ Loading SQuAD ...")
    try:
        squad = load_dataset("squad", split="train", trust_remote_code=True)
        for rec in squad:
            question = (rec.get("question") or "").strip()
            if question and len(question) > 10 and len(general_texts) < target_n * 3:
                general_texts.append(question)
        print(f"    ✓ Got {len(general_texts):,} total after SQuAD")
    except Exception as e:
        print(f"    ⚠ SQuAD failed: {e}")

    # Source 3: WikiQA — Wikipedia Q&A pairs
    print("    ↓ Loading wikiqa ...")
    try:
        wikiqa = load_dataset("wikiqa", split="train", trust_remote_code=True)
        for rec in wikiqa:
            question = (rec.get("question") or "").strip()
            if question and len(question) > 10 and len(general_texts) < target_n * 4:
                general_texts.append(question)
        print(f"    ✓ Got {len(general_texts):,} total after wikiqa")
    except Exception as e:
        print(f"    ⚠ wikiqa failed: {e}")

    # Source 4: Natural Questions — Google's natural Q&A dataset
    print("    ↓ Loading natural_questions ...")
    try:
        nq = load_dataset("natural_questions", split="train", trust_remote_code=True)
        for i, rec in enumerate(nq):
            if i >= 5000:  # Limit to avoid huge download
                break
            question = (rec.get("question") or "").strip()
            if question and len(question) > 10 and len(general_texts) < target_n * 5:
                general_texts.append(question)
        print(f"    ✓ Got {len(general_texts):,} total after natural_questions")
    except Exception as e:
        print(f"    ⚠ natural_questions failed: {e}")

    # Source 5: CoQA (Conversational QA) — conversational questions
    print("    ↓ Loading coqa ...")
    try:
        coqa = load_dataset("coqa", split="train", trust_remote_code=True)
        for rec in coqa:
            questions = rec.get("questions", [])
            for q_obj in questions:
                question = (q_obj.get("input_text") or "").strip()
                if question and len(question) > 10 and len(general_texts) < target_n * 6:
                    general_texts.append(question)
        print(f"    ✓ Got {len(general_texts):,} total after coqa")
    except Exception as e:
        print(f"    ⚠ coqa failed: {e}")

    # Source 6: Hand-crafted general queries that historically get misrouted
    handcrafted = [
        # Everyday / how-to
        "How do I open a tight jar?",
        "What's the best way to remove a stain from a shirt?",
        "How do I change a car tire?",
        "What's the proper way to fold a fitted sheet?",
        "How do I unclog a drain?",
        "What's the best way to organize a closet?",
        "How do I get rid of ants in my kitchen?",
        "What's the easiest way to clean a microwave?",
        "How long should I cook pasta for?",
        "What temperature should I set my oven to for baking cookies?",
        "How do I tie a necktie?",
        "What's the best way to sharpen a knife?",
        "How do I parallel park?",
        "How do I iron a dress shirt properly?",
        "What's a good recipe for chocolate chip cookies?",
        "How do I boil an egg perfectly?",
        "How do I make pancakes from scratch?",
        "What's the best way to store vegetables?",
        "How do I fix a leaky faucet?",
        "What's the proper technique for doing push-ups?",
        # Tech / history (non-code, non-medical)
        "What is the history of the internet?",
        "Who invented the telephone?",
        "Explain how a refrigerator works.",
        "What is blockchain technology?",
        "How does GPS work?",
        "What is the history of video games?",
        "How does Wi-Fi work?",
        "What was the first computer?",
        "Explain the history of social media.",
        "How do electric cars work?",
        "What is artificial intelligence?",
        "How does a microwave oven work?",
        "What is the history of space exploration?",
        "How does 3D printing work?",
        "What is cryptocurrency?",
        "How do rockets work?",
        "What is the cloud and how does it work?",
        "How does a helicopter stay in the air?",
        "What is nuclear energy?",
        "How do solar cells convert sunlight to electricity?",
        # General knowledge (not QA-style factual)
        "Tell me something interesting about dolphins.",
        "What makes a good leader?",
        "How can I improve my public speaking skills?",
        "What are some tips for better sleep?",
        "How do I start learning to play guitar?",
        "What's the difference between a crocodile and an alligator?",
        "How does photography work?",
        "What are some fun activities to do on a rainy day?",
        "How do I write a good resume?",
        "What are some strategies for saving money?",
        "How do I train a puppy?",
        "What's the best way to learn a new language?",
        "How do I start a garden?",
        "What are some good books for beginners in philosophy?",
        "How do solar panels work?",
        "What's the difference between weather and climate?",
        "How can I improve my memory?",
        "What are some tips for reducing stress?",
        "How do I plan a budget for a vacation?",
        "What makes a good presentation?",
        "What's the best way to make coffee?",
        "How do I choose a good wine?",
        "What are the benefits of meditation?",
        "How do I organize my time better?",
        "What's a good exercise routine for beginners?",
        # Ambiguous queries that the old model misclassified
        "Can you explain how transformers work in NLP?",
        "Explain the historical significance of the Magna Carta.",
        "What is the tallest mountain in the solar system?",
        "What is the chemical symbol for gold?",
        "Which planet has the most moons?",
        "What are the most popular programming paradigms?",
        "Tell me about the Renaissance period.",
        "What are the different types of clouds?",
        "How do airplanes fly?",
        "What causes earthquakes?",
        "How is paper made?",
        "What is the theory of evolution?",
        "How do magnets work?",
        "What causes the Northern Lights?",
        "How do submarines work?",
        # Geography / nature
        "Which is the largest ocean on Earth?",
        "What is the deepest ocean trench?",
        "How many continents are there?",
        "What causes a tornado?",
        "How do coral reefs form?",
        "What is photosynthesis?",
        "How do birds migrate?",
        "What is biodiversity?",
        "How do volcanoes form?",
        "What causes tsunamis?",
        # History
        "When was the Roman Empire at its peak?",
        "Who was Julius Caesar?",
        "What started World War I?",
        "Who invented the printing press?",
        "What is the Silk Road?",
        "When did the Industrial Revolution begin?",
        "Who was Cleopatra?",
        "When did the Berlin Wall fall?",
        "What was the Black Death?",
        "Who were the Vikings?",
        # Arts & culture
        "Who painted the Mona Lisa?",
        "What is Shakespeare famous for?",
        "Who composed Beethoven's symphonies?",
        "What is the Louvre?",
        "Who was Michelangelo?",
        "What is opera?",
        "Who wrote Don Quixote?",
        "What is the purpose of art?",
        "Who invented ballet?",
        "What are the major art movements?",
        # Social / lifestyle
        "What are the benefits of reading?",
        "How do I build healthy habits?",
        "What makes a good friendship?",
        "How can I be more confident?",
        "What is work-life balance?",
        "How do I deal with anxiety?",
        "What is networking?",
        "How do I find my passion?",
        "What are soft skills?",
        "How do I become a better listener?",
        # Environment & sustainability
        "What is climate change?",
        "How can I reduce my carbon footprint?",
        "What is renewable energy?",
        "How does recycling help the environment?",
        "What is deforestation?",
        "How do I live sustainably?",
        "What is plastic pollution?",
        "How does pollution affect animals?",
        "What is the ozone layer?",
        "Why are bees important?",
        # Sports & fitness
        "What are the benefits of yoga?",
        "How do I train for a marathon?",
        "What is CrossFit?",
        "How do I improve my flexibility?",
        "What is the best time to exercise?",
        "How do I prevent sports injuries?",
        "What is a healthy diet?",
        "How do I build muscle mass?",
        "What is the difference between aerobic and anaerobic exercise?",
        "How do I stay motivated for fitness?",
    ]
    general_texts.extend(handcrafted)
    print(f"    ✓ Added {len(handcrafted)} handcrafted samples")

    # Deduplicate and shuffle
    general_texts = list(set(general_texts))
    random.shuffle(general_texts)

    # Trim to target
    if len(general_texts) > target_n:
        general_texts = general_texts[:target_n]
    print(f"    ✓ Final: {len(general_texts):,} general samples")

    return general_texts


# Load existing 4-class data
print("\nLoading domain datasets:")
all_texts, all_labels = [], []
for domain, path in DATA_FILES.items():
    texts, labels = load_json_file(path)
    all_texts += texts
    all_labels += labels
    print(f"  {path.name:15s} → {len(texts):,} samples  (label {LABEL2ID[domain]} / '{domain}')")

raw_counts = Counter(all_labels)
print(f"\nRaw class dist (4-class): { {ID2LABEL.get(k, str(k)): v for k, v in sorted(raw_counts.items())} }")

# ==============================================================================
# 4. Balance classes + add "general"
# ==============================================================================
by_class: Dict[int, List[str]] = {}
for text, label in zip(all_texts, all_labels):
    by_class.setdefault(label, []).append(text)

# Remap labels: the JSON files use their own label numbers, remap to LABEL2ID
# code.json has label 0, math.json has label 1, etc — matches our LABEL2ID
class_sizes = [len(by_class.get(i, [])) for i in range(4)]
target_n = int(np.median(class_sizes))
print(f"\nTarget per class: {target_n:,}")

# Build general class
general_texts = build_general_class(target_n)
by_class[LABEL2ID["general"]] = general_texts

# Balance all 5 classes to target_n
balanced_texts, balanced_labels = [], []
for label_id in sorted(LABEL2ID.values()):
    texts = by_class.get(label_id, [])
    if not texts:
        print(f"  ⚠ No data for label {label_id} / {ID2LABEL[label_id]}")
        continue
    if len(texts) >= target_n:
        sampled = random.sample(texts, target_n)
    else:
        sampled = texts + random.choices(texts, k=target_n - len(texts))
    balanced_texts += sampled
    balanced_labels += [label_id] * len(sampled)

balanced_counts = Counter(balanced_labels)
print(f"\nBalanced dist (5-class): { {ID2LABEL[k]: v for k, v in sorted(balanced_counts.items())} }")
print(f"Total samples: {len(balanced_texts):,}")

# ==============================================================================
# 5. Train/Val/Test splits (stratified)
# ==============================================================================
train_texts, train_labels = [], []
val_texts,   val_labels   = [], []
test_texts,  test_labels  = [], []

for label_id in sorted(set(balanced_labels)):
    class_texts = [t for t, l in zip(balanced_texts, balanced_labels) if l == label_id]
    tr_val, te  = train_test_split(class_texts, test_size=TEST_SPLIT, random_state=SEED)
    val_frac    = VAL_SPLIT / (1 - TEST_SPLIT)
    tr, va      = train_test_split(tr_val, test_size=val_frac, random_state=SEED)
    train_texts += tr;  train_labels += [label_id] * len(tr)
    val_texts   += va;  val_labels   += [label_id] * len(va)
    test_texts  += te;  test_labels  += [label_id] * len(te)

combined = list(zip(train_texts, train_labels))
random.shuffle(combined)
train_texts, train_labels = map(list, zip(*combined))

print(f"\nTrain: {len(train_texts):,}")
print(f"Val:   {len(val_texts):,}")
print(f"Test:  {len(test_texts):,}")

# ==============================================================================
# 6. Class weights
# ==============================================================================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array(sorted(LABEL2ID.values())),
    y=np.array(train_labels),
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

print("\nClass weights:")
for i, w in enumerate(class_weights):
    print(f"  {ID2LABEL[i]:12s}: {w:.4f}")

# ==============================================================================
# 7. Tokenise
# ==============================================================================
print(f"\nTokenising with {MODEL_CHECKPOINT} (max_length={MAX_LENGTH}) ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)


def make_hf_dataset(texts, labels):
    ds = Dataset.from_dict({"text": texts, "label": labels})
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True, padding="max_length", max_length=MAX_LENGTH,
        )
    ds = ds.map(tokenize, batched=True, batch_size=512)
    ds = ds.remove_columns(["text"])
    ds.set_format("torch")
    return ds


train_ds = make_hf_dataset(train_texts, train_labels)
val_ds   = make_hf_dataset(val_texts,   val_labels)
test_ds  = make_hf_dataset(test_texts,  test_labels)

print(f"Train: {train_ds}")
print(f"Val:   {val_ds}")

# ==============================================================================
# 8. Model + LoRA
# ==============================================================================
print(f"\nLoading {MODEL_CHECKPOINT} with {len(LABEL2ID)} labels ...")

base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(LABEL2ID),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

lora_config = LoraConfig(
    task_type        = TaskType.SEQ_CLS,
    r                = LORA_R,
    lora_alpha       = LORA_ALPHA,
    lora_dropout     = LORA_DROPOUT,
    target_modules   = LORA_TARGET_MODULES,
    bias             = "none",
    modules_to_save  = ["classifier", "pre_classifier"],
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# ==============================================================================
# 9. Custom Trainer with Weighted Loss
# ==============================================================================
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        weights = class_weights_tensor.to(logits.device)
        loss    = nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro")
    return {
        "accuracy" : round(float(acc), 4),
        "macro_f1" : round(float(f1), 4),
    }

# ==============================================================================
# 10. Training — H200 optimised
# ==============================================================================
total_steps  = (len(train_ds) // BATCH_SIZE) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = NUM_EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE * 2,
    learning_rate               = LEARNING_RATE,
    weight_decay                = WEIGHT_DECAY,
    warmup_steps                = warmup_steps,
    lr_scheduler_type           = "cosine",
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "accuracy",
    greater_is_better           = True,
    bf16                        = True,     # H200 native bf16 (was fp16)
    logging_steps               = 10,
    report_to                   = "none",
    push_to_hub                 = False,
    seed                        = SEED,
    dataloader_num_workers      = 4,        # H200 has plenty of CPU cores
    dataloader_pin_memory       = True,
)

trainer = WeightedLossTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_ds,
    eval_dataset    = val_ds,
    compute_metrics = compute_metrics,
)

print(f"\nTotal steps  : {total_steps}")
print(f"Warmup steps : {warmup_steps}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Precision    : bf16 (H200)")
print()

t_start = time.time()
trainer.train()
t_train = time.time() - t_start
print(f"\n✓ Training complete in {t_train:.1f}s")

# ==============================================================================
# 11. Evaluate on Test Set
# ==============================================================================
print("\n" + "=" * 60)
print("FINAL TEST SET RESULTS")
print("=" * 60)

preds_out  = trainer.predict(test_ds)
pred_ids   = np.argmax(preds_out.predictions, axis=-1)
true_ids   = preds_out.label_ids
label_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]

print(classification_report(true_ids, pred_ids, target_names=label_names))

cm = confusion_matrix(true_ids, pred_ids)
import pandas as pd
cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
print("Confusion Matrix:")
print(cm_df.to_string())

# ==============================================================================
# 12. Smoke Test — especially the misrouted queries
# ==============================================================================
print("\n" + "=" * 60)
print("SMOKE TEST")
print("=" * 60)

model.eval()
smoke_queries = [
    # Should be general (NOT medical)
    ("How do I open a tight jar?",                          "general"),
    ("What is the history of the internet?",                "general"),
    ("How does a microwave work?",                          "general"),
    ("What makes a good leader?",                           "general"),
    ("Tell me something interesting about dolphins.",        "general"),
    ("What is the chemical symbol for gold?",               "general"),
    ("Which planet has the most moons?",                    "general"),
    ("Explain the historical significance of the Magna Carta.", "general"),
    ("What is the tallest mountain in the solar system?",   "general"),
    ("Can you explain how transformers work in NLP?",       "general"),
    # Should be their respective domains
    ("Write a Python function to sort a list.",              "code"),
    ("Solve the integral of x^2 from 0 to 1.",              "math"),
    ("What are the symptoms of Type 2 diabetes?",           "medical"),
    ("Who was the first woman to win a Nobel Prize?",       "qa"),
    ("Implement Dijkstra's algorithm in Python.",            "code"),
    ("What is the derivative of sin(x)?",                   "math"),
    ("How is hypertension typically treated?",               "medical"),
    ("When did World War II end?",                          "qa"),
]

correct = 0
total = len(smoke_queries)
for query, expected in smoke_queries:
    enc = tokenizer(query, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = probs.argmax().item()
    pred_label = ID2LABEL[pred_id]
    conf = probs[pred_id].item()
    match = "✅" if pred_label == expected else "❌"
    if pred_label == expected:
        correct += 1
    print(f"  {match} {query[:55]:57s} → {pred_label:10s} ({conf:.3f})  expected={expected}")

print(f"\nSmoke test: {correct}/{total} ({100*correct/total:.0f}%)")

# ==============================================================================
# 13. Save & Push to HuggingFace Hub
# ==============================================================================
print("\n" + "=" * 60)
print("SAVING & PUSHING")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

merged_model = model.merge_and_unload()
merged_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

model_card = f"""---
language: en
license: apache-2.0
tags:
  - text-classification
  - distilbert
  - adaptroute
  - router
  - lora
pipeline_tag: text-classification
---

# gating-bert-adaptroute (v2 — 5-class)

A 5-class DistilBERT classifier acting as the gating network for AdaptRoute.

## Labels
| ID | Label | Route |
|----|-------|-------|
| 0 | code | → lora-code |
| 1 | math | → lora-math |
| 2 | qa | → lora-qa |
| 3 | medical | → lora-medical |
| 4 | general | → base model (no adapter) |

## Architecture
- Base: `distilbert-base-uncased` (LoRA merged)
- Training: {NUM_EPOCHS} epochs, lr={LEARNING_RATE}, batch={BATCH_SIZE}, bf16
- H200 optimised
"""

with open(f"{OUTPUT_DIR}/README.md", "w") as f:
    f.write(model_card)

if HF_TOKEN:
    api = HfApi()
    api.create_repo(repo_id=FULL_REPO_ID, exist_ok=True, token=HF_TOKEN)
    api.upload_folder(folder_path=OUTPUT_DIR, repo_id=FULL_REPO_ID, token=HF_TOKEN)
    print(f"✓ Pushed to https://huggingface.co/{FULL_REPO_ID}")
else:
    print(f"✓ Saved locally to {OUTPUT_DIR}/ (HF_TOKEN not set, skipped push)")

# ==============================================================================
# 14. Finish
# ==============================================================================


print("\n" + "=" * 60)
print("Done.")
print(f"  Hub   : https://huggingface.co/{FULL_REPO_ID}")
print(f"  Local : {OUTPUT_DIR}/")
print("=" * 60)
