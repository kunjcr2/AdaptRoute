# Yup, this is the newest one
# run this - https://colab.research.google.com/drive/1ouVcu3Nu2c2BXARJBCSqqrGm4PEso6Fw?usp=sharing

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import os
from huggingface_hub import snapshot_download
import time

# ==============================================================================
# CONFIGURATION
# Adjust these repository names based on your latest Hugging Face iterations (e.g. v3 vs v2)
# ==============================================================================
FIREWALL_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
GATING_MODEL = "kunjcr2/gating-bert-adaptroute"
BASE_MODEL = "Qwen/Qwen2.5-1.5B"

ADAPTER_REPOS = {
    "code": "kunjcr2/code-adaptroute-v3",     # e.g., Update to -v3 if needed
    "math": "kunjcr2/math-adaptroute-v3",
    "qa": "kunjcr2/qa-adaptroute-v3",
    "medical": "kunjcr2/medical-adaptroute-v3"
}

# Fix: In Colab/Jupyter, __file__ is not defined. Use os.getcwd() instead.
ADAPTERS_DIR = os.path.abspath(os.path.join(os.getcwd(), 'Adapters'))

# Single device — CUDA if available (A100 on Colab), else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def prepare():
    """
    Checks if the local Adapters folder has the required weights.
    If not, downloads them from Hugging Face into their respective subdirectories.
    """
    os.makedirs(ADAPTERS_DIR, exist_ok=True)
    # Check if there are subdirectories for the models
    existing_items = [name for name in os.listdir(ADAPTERS_DIR) if os.path.isdir(os.path.join(ADAPTERS_DIR, name))]

    if not existing_items:
        print(f"Adapters folder is empty. Downloading adapters from Hugging Face to {ADAPTERS_DIR}...")
        for domain, repo_id in ADAPTER_REPOS.items():
            local_path = os.path.join(ADAPTERS_DIR, domain)
            print(f"Downloading {repo_id} to {local_path}...")
            # Ignore some large tracking files or tensorboard logs to keep it fast
            snapshot_download(repo_id=repo_id, local_dir=local_path, ignore_patterns=["*.msgpack", "*.h5"])
        print("Finished downloading all adapters.")
    else:
        print(f"Adapters already exist locally in {ADAPTERS_DIR}.")

# Global dictionary to keep models loaded in RAM
global_systems = {
    "firewall_model": None,
    "firewall_tokenizer": None,
    "gating_model": None,
    "gating_tokenizer": None,
    "base_model": None,
    "base_tokenizer": None,
}

def load_all_models():
    """
    Loads the Firewall, Gating Network, Base Model (quantized), and all Adapters into RAM.
    """
    global global_systems

    # 1. Load Firewall (ProtectAI's pre-trained prompt injection detector)
    # Labels: 0 = SAFE, 1 = INJECTION
    print("Loading Firewall Model...")
    global_systems["firewall_tokenizer"] = AutoTokenizer.from_pretrained(FIREWALL_MODEL)
    global_systems["firewall_model"] = AutoModelForSequenceClassification.from_pretrained(FIREWALL_MODEL).to(DEVICE)
    global_systems["firewall_model"].eval()

    # 2. Load Gating Network
    print("Loading Gating Network...")
    global_systems["gating_tokenizer"] = AutoTokenizer.from_pretrained(GATING_MODEL)
    global_systems["gating_model"] = AutoModelForSequenceClassification.from_pretrained(GATING_MODEL).to(DEVICE)
    global_systems["gating_model"].eval()

    # 3. Load Base Model — use PyTorch built-in SDPA (Flash Attention without any extra install)
    # attn_implementation="sdpa" uses torch.nn.functional.scaled_dot_product_attention
    # which hits Flash Attention kernels natively on A100 (PyTorch 2.0+, already on Colab)
    print("Loading Base Model with SDPA attention (bfloat16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(DEVICE)
    # CRITICAL FIX for generation speed: use_cache must be True for inference
    base_model.config.use_cache = True

    global_systems["base_tokenizer"] = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if global_systems["base_tokenizer"].pad_token is None:
        global_systems["base_tokenizer"].pad_token = global_systems["base_tokenizer"].eos_token
    global_systems["base_tokenizer"].padding_side = "right"

    global_systems["base_model"] = base_model
    global_systems["base_model"].eval()

    print("All models loaded successfully! (Adapters deferred until queried)")

def process_query(query: str) -> dict:
    """
    Passes a single query through the firewall, gating network, soft-merges adapters,
    generates a response, and returns a structured dict for the API layer.
    """
    if any(m is None for m in global_systems.values()):
        return {"status": "error", "message": "Models are not loaded. Please call load_all_models() first."}

    import time
    t_start = time.time()

    # ---------------------------------------------------------
    # 1. Firewall Check
    # ---------------------------------------------------------
    fw_tokenizer = global_systems["firewall_tokenizer"]
    fw_model = global_systems["firewall_model"]

    fw_inputs = fw_tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(fw_model.device)
    with torch.no_grad():
        fw_outputs = fw_model(**fw_inputs)

    predicted_fw_class_id = fw_outputs.logits.argmax(dim=-1).item()
    fw_label = fw_model.config.id2label.get(predicted_fw_class_id, "SAFE")

    # ProtectAI labels: "SAFE" = pass, "INJECTION" = block
    if fw_label == "INJECTION":
        return {
            "status": "blocked",
            "message": "Your query was flagged as a potential prompt injection attempt and could not be processed. Please rephrase your request.",
            "firewall_label": fw_label
        }

    t_fw = time.time()
    # ---------------------------------------------------------
    # 2. Gating Network (Task Routing - Hard Routing)
    # ---------------------------------------------------------
    gating_tokenizer = global_systems["gating_tokenizer"]
    gating_model = global_systems["gating_model"]

    gate_inputs = gating_tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(gating_model.device)
    with torch.no_grad():
        gate_outputs = gating_model(**gate_inputs)

    # Get highest probability domain directly
    probs = torch.softmax(gate_outputs.logits, dim=-1).squeeze()
    best_idx = probs.argmax().item()
    best_prob = probs[best_idx].item()
    gate_id2label = gating_model.config.id2label

    # THRESHOLD CHECK: If confidence is below 0.9, use base model without adapter
    if best_prob < 0.9:
        print(f"Gating confidence {best_prob:.4f} below threshold (0.9). Using base model without adapter.")
        winning_domain = None
        winning_label = "base_model"
    else:
        winning_label = gate_id2label.get(best_idx, "").lower()
        winning_domain = None

        # Identify which local adapter to use based on the winning label
        for domain in ADAPTER_REPOS.keys():
            if domain.lower() in winning_label:
                winning_domain = domain
                break

        print(f"Using adapter: {winning_domain}")

        if not winning_domain:
            return {"status": "error", "message": f"Could not map gating network output to a known adapter. Got: {winning_label}"}

    t_gate = time.time()
    # ---------------------------------------------------------
    # 3. Dynamic Adapter Loading (Direct Use) - Skip if below threshold
    # ---------------------------------------------------------
    base_model = global_systems["base_model"]
    
    if winning_domain is not None:
        # Load adapter only if above threshold
        local_adapter_path = os.path.join(ADAPTERS_DIR, winning_domain)

        if not isinstance(base_model, PeftModel):
            # First time loading an adapter
            base_model = PeftModel.from_pretrained(base_model, local_adapter_path, adapter_name=winning_domain)
            global_systems["base_model"] = base_model
        else:
            # Load if not already present in memory
            if winning_domain not in base_model.peft_config:
                base_model.load_adapter(local_adapter_path, adapter_name=winning_domain)

            # Switch to the requested adapter directly
            base_model.set_adapter(winning_domain)

    t_adapter = time.time()
    # ---------------------------------------------------------
    # 4. Generation (BOTH: Base Model Alone vs. Base Model + Adapter)
    # ---------------------------------------------------------
    base_tokenizer = global_systems["base_tokenizer"]

    # Dynamic max_new_tokens based on domain
    max_tokens_map = {
        "medical": 256,
        "code": 256,
        "math": 128,
        "qa": 64
    }
    max_new_tokens = max_tokens_map.get(winning_domain, 128) if winning_domain else 128

    # Standard format wrapper used for QA/Tasks
    formatted_prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    enc = base_tokenizer(formatted_prompt, return_tensors="pt").to(base_model.device)

    # Identify stop tokens nicely for ChatML (Qwen usually uses <|im_end|>)
    stop_tokens = [base_tokenizer.eos_token_id]
    im_end_id = base_tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != base_tokenizer.unk_token_id:
        stop_tokens.append(im_end_id)

    # Math proofs repeat symbols/terms naturally — loosen the constraint for that domain
    ngram_size = 5 if winning_domain == "math" else 3

    # Generate response with adapter if available, otherwise base model only
    if winning_domain is not None:
        base_model.set_adapter(winning_domain)
        base_model.merge_adapter()
    
    with torch.inference_mode():
        out = base_model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            no_repeat_ngram_size=ngram_size,
            pad_token_id=base_tokenizer.pad_token_id,
            eos_token_id=stop_tokens,
        )
    
    if winning_domain is not None:
        base_model.unmerge_adapter()

    t_gen = time.time()

    response = base_tokenizer.decode(
        out[0][enc["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # For code domain, remove trailing comments
    if winning_domain == "code":
        parts = response.split("#")
        if len(parts) > 1:
            response = "#".join(parts[:-1]).strip()
        # Clean up any trailing whitespace/artifacts
        response = response.rstrip()
      
    if winning_domain == "medical" or winning_domain == "math":
        parts = response.split(".")
        if len(parts) > 1:
            response = ".".join(parts[:-1]).strip()
        parts = response.split("\n")
        if len(parts) > 1:
            response = "\n".join(parts[:-1]).strip()
        response = response.rstrip()

    t_total = t_gen - t_start

    gating_scores = {
        gate_id2label.get(i, str(i)).lower(): round(probs[i].item(), 4)
        for i in range(len(probs))
    }

    return {
        "status": "success",
        "response": response,
        "adapter_used": winning_domain if winning_domain else "base_model",
        "gating_confidence": round(best_prob, 4),
        "gating_scores": gating_scores,
        "firewall_label": fw_label,
        "time_seconds": round(t_total, 2),
    }