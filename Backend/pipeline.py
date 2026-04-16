import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
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

    # 3. Load Base Model (Native bfloat16 for A100 speed!)
    print("Loading Base Model natively in bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
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
    gate_id2label = gating_model.config.id2label

    winning_label = gate_id2label.get(best_idx, "").lower()
    winning_domain = None

    # Identify which local adapter to use based on the winning label
    for domain in ADAPTER_REPOS.keys():
        if domain.lower() in winning_label:
            winning_domain = domain
            break

    print(f"Using the adapter: {winning_domain}")

    if not winning_domain:
        return {"status": "error", "message": f"Could not map gating network output to a known adapter. Got: {winning_label}"}

    t_gate = time.time()
    # ---------------------------------------------------------
    # 3. Dynamic Adapter Loading (Direct Use)
    # ---------------------------------------------------------
    base_model = global_systems["base_model"]
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
    # 4. Generation
    # ---------------------------------------------------------
    base_tokenizer = global_systems["base_tokenizer"]

    # Standard format wrapper used for QA/Tasks
    formatted_prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    enc = base_tokenizer(formatted_prompt, return_tensors="pt").to(base_model.device)

    # Identify stop tokens nicely for ChatML (Qwen usually uses <|im_end|>)
    stop_tokens = [base_tokenizer.eos_token_id]
    im_end_id = base_tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != base_tokenizer.unk_token_id:
        stop_tokens.append(im_end_id)

    with torch.no_grad():
        out = base_model.generate(
            **enc,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.5,
            pad_token_id=base_tokenizer.pad_token_id,
            eos_token_id=stop_tokens
        )

    response = base_tokenizer.decode(
        out[0][enc["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    t_total = t_gen - t_start

    print("\n--- INFERENCE PROFILING ---")
    print(f"Firewall Check:    {t_fw - t_start:.2f}s")
    print(f"Gating Network:    {t_gate - t_fw:.2f}s")
    print(f"Adapter Switching: {t_adapter - t_gate:.2f}s")
    generated_tokens = out.shape[1] - enc['input_ids'].shape[1]
    print(f"Text Generation:   {t_gen - t_adapter:.2f}s ({generated_tokens} tokens, {generated_tokens/(t_gen - t_adapter + 0.0001):.2f} tok/s)")
    print(f"Total Time:        {t_total:.2f}s")
    print("---------------------------\n")

    gating_scores = {
        gate_id2label.get(i, str(i)).lower(): round(probs[i].item(), 4)
        for i in range(len(probs))
    }

    return {
        "status": "success",
        "response": response,
        "adapter_used": winning_domain,
        "gating_scores": gating_scores,
        "time_taken_seconds": round(t_total, 2),
    }

prepare()
load_all_models()

test_query = ""
result = process_query(test_query)
print(result)