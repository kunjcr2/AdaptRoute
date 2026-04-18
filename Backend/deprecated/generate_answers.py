"""
Script to generate answers for all questions in query_log.jsonl using the pipeline.
Updates answers and removes timestamps.
"""

import json
import re
import sys
import os
from pathlib import Path

# Add the current directory to path to import pipeline
sys.path.insert(0, str(Path(__file__).parent))

# Import pipeline utilities
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel

# Import your pipeline - handle different possible import names
# Import your pipeline
try:
    import pipeline
    from pipeline import (
        load_all_models, global_systems, FIREWALL_MODEL, BASE_MODEL, 
        ADAPTER_REPOS, ADAPTERS_DIR, DEVICE, prepare
    )
    print("[Pipeline] Pipeline module loaded")
except ImportError as e:
    print(f"[ERROR] Could not import 'pipeline.py': {e}")
    print("  Make sure your pipeline file is in the same directory and all dependencies are installed.")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def fix_json_line(line):
    """Fix common JSON escape sequence issues."""
    # Fix invalid escape sequences like \* or \**
    line = re.sub(r'\\(\*+)', r'\\u002a\1', line)  # Replace with unicode asterisk
    # Fix other common issues
    line = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', line)  # Escape invalid escapes
    return line


def process_query_with_domain(query: str, domain: str) -> dict:
    """
    Process a query directly using the specified domain adapter.
    Skips the gating network and routes directly to the domain's adapter.
    
    Args:
        query (str): The question to process
        domain (str): The domain to use (code, math, medical, etc)
    
    Returns:
        dict: Result containing status and response
    """
    if any(m is None for m in global_systems.values()):
        return {"status": "error", "message": "Models not loaded. Call load_all_models() first."}
    
    import time
    t_start = time.time()
    
    # ──────────────────────────────────────────────────────────────
    # 1. Firewall Check
    # ──────────────────────────────────────────────────────────────
    fw_tokenizer = global_systems["firewall_tokenizer"]
    fw_model = global_systems["firewall_model"]
    
    fw_inputs = fw_tokenizer(
        query, return_tensors="pt", truncation=True, max_length=512
    ).to(fw_model.device)
    
    with torch.no_grad():
        fw_outputs = fw_model(**fw_inputs)
    
    predicted_fw_class_id = fw_outputs.logits.argmax(dim=-1).item()
    fw_label = fw_model.config.id2label.get(predicted_fw_class_id, "SAFE")
    
    if fw_label == "INJECTION":
        return {
            "status": "blocked",
            "message": "Your query was flagged as a potential prompt injection attempt.",
            "firewall_label": fw_label,
        }
    
    # ──────────────────────────────────────────────────────────────
    # 2. Direct Domain Routing (skip gating network)
    # ──────────────────────────────────────────────────────────────
    domain = domain.lower().strip()
    valid_domains = list(ADAPTER_REPOS.keys())
    
    if domain not in valid_domains:
        print(f"[Route] ⚠️  Domain '{domain}' not found, falling back to base model")
        routing_mode = "base"
        winning_domain = None
    else:
        routing_mode = "hard"
        winning_domain = domain
        print(f"[Route] DIRECT → {winning_domain} (based on domain field)")
    
    # ──────────────────────────────────────────────────────────────
    # 3. Adapter Loading (if domain specified)
    # ──────────────────────────────────────────────────────────────
    base_model = global_systems["base_model"]
    base_tokenizer = global_systems["base_tokenizer"]
    
    if routing_mode == "hard":
        local_path = os.path.join(ADAPTERS_DIR, winning_domain)
        adapter_source = local_path if os.path.exists(local_path) else ADAPTER_REPOS[winning_domain]
        
        if not isinstance(base_model, PeftModel):
            base_model = PeftModel.from_pretrained(
                base_model, adapter_source, adapter_name=winning_domain
            )
            global_systems["base_model"] = base_model
        else:
            if winning_domain not in base_model.peft_config:
                base_model.load_adapter(adapter_source, adapter_name=winning_domain)
            base_model.set_adapter(winning_domain)
    
    # ──────────────────────────────────────────────────────────────
    # 4. Generation
    # ──────────────────────────────────────────────────────────────
    max_tokens_map = {"medical": 256, "code": 256, "math": 128}
    max_new_tokens = max_tokens_map.get(winning_domain, 128) if winning_domain else 128
    
    formatted_prompt = (
        f"<start_of_turn>user\n{query}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    enc = base_tokenizer(formatted_prompt, return_tensors="pt").to(base_model.device)
    
    stop_tokens = [base_tokenizer.eos_token_id]
    end_of_turn_id = base_tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if end_of_turn_id and end_of_turn_id != base_tokenizer.unk_token_id:
        stop_tokens.append(end_of_turn_id)
    
    ngram_size = 5 if winning_domain == "math" else 3
    
    if routing_mode == "hard":
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
    
    if routing_mode == "hard":
        base_model.unmerge_adapter()
    
    # ──────────────────────────────────────────────────────────────
    # 5. Decode + Post-process
    # ──────────────────────────────────────────────────────────────
    response = base_tokenizer.decode(
        out[0][enc["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    
    if winning_domain == "code":
        parts = response.split("#")
        if len(parts) > 1:
            response = "#".join(parts[:-1]).strip()
        response = response.rstrip()
    else:
        parts = response.split(".")
        if len(parts) > 1:
            response = ".".join(parts[:-1]).strip()
        parts = response.split("\n")
        if len(parts) > 1:
            response = "\n".join(parts[:-1]).strip()
        response = response.rstrip()
    
    t_total = time.time() - t_start
    
    return {
        "status": "success",
        "response": response,
        "adapter_used": winning_domain if winning_domain else "base_model",
        "routing_mode": routing_mode,
        "firewall_label": fw_label,
        "time_seconds": round(t_total, 2),
    }


def validate_record(record):
    """Validate that a record has required fields."""
    required_fields = ["question", "domain"]
    for field in required_fields:
        if field not in record:
            return False, f"Missing required field: {field}"
    return True, "OK"


def generate_answers_for_query_log(input_file="query_log.jsonl", output_file="query_log.jsonl", 
                                   backup_original=True, max_questions=None):
    """
    Reads questions from query_log.jsonl, generates answers using the pipeline,
    and writes back the updated records without timestamps.
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSONL file (defaults to overwriting input)
        backup_original (bool): Whether to backup the original file
        max_questions (int): Maximum number of questions to process (for testing)
    """
    records = []
    
    # Backup original file if requested
    if backup_original and Path(input_file).exists():
        backup_file = f"{input_file}.backup"
        print(f"[Backup] Creating backup: {backup_file}")
        try:
            import shutil
            shutil.copy2(input_file, backup_file)
            print(f"[Backup] Backup created at {backup_file}")
        except Exception as e:
            print(f"[WARNING] Could not create backup: {e}")
    
    # Read all records from the JSONL file
    print(f"\n[Read] Reading {input_file}...")
    try:
        with open(f"./{input_file}", 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    # Try to parse JSON
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        # Try to fix common escape sequence issues
                        fixed_line = fix_json_line(line)
                        try:
                            record = json.loads(fixed_line)
                            records.append(record)
                            print(f"  [OK] [Line {line_num}] Fixed JSON issue: {e}")
                        except json.JSONDecodeError as e2:
                            print(f"  [ERROR] [Line {line_num}] Cannot parse, skipping: {e2}")
                            continue
        print(f"[Read] Loaded {len(records)} records")
    except FileNotFoundError:
        print(f"[ERROR] {input_file} not found")
        return
    except Exception as e:
        print(f"[ERROR] Error reading file: {e}")
        return
    
    if not records:
        print("[ERROR] No valid records found in file")
        return
    
    # Load models once before processing
    print("\n[Setup] Loading models...")
    try:
        print("  Running pipeline.prepare()...")
        prepare()
        load_all_models()
        print("[Pipeline] All models loaded and adapters prepared")
    except Exception as e:
        print(f"[ERROR] Error loading models: {e}")
        return
    
    # Limit questions if specified
    if max_questions and max_questions < len(records):
        print(f"\n[WARNING] Limiting to first {max_questions} questions (of {len(records)})")
        records = records[:max_questions]
    
    # Process each record
    print(f"\n[Model] Generating answers for {len(records)} questions using domain-based routing...")
    print("-" * 60)
    
    updated_records = []
    success_count = 0
    error_count = 0
    
    for idx, record in enumerate(records, 1):
        question = record.get("question", "")
        domain = record.get("domain", "general")
        
        # Validate record
        is_valid, error_msg = validate_record(record)
        if not is_valid:
            print(f"  [{idx}/{len(records)}] [WARNING] {error_msg}")
            # Keep record but remove timestamp
            record_copy = {k: v for k, v in record.items() if k != "timestamp"}
            updated_records.append(record_copy)
            error_count += 1
            continue
        
        if not question:
            print(f"  [{idx}/{len(records)}] [WARNING] Empty question, skipping")
            record_copy = {k: v for k, v in record.items() if k != "timestamp"}
            updated_records.append(record_copy)
            error_count += 1
            continue
        
        try:
            print(f"  [{idx}/{len(records)}] Domain: {domain} | {question[:50]}...")
            
            # Call the pipeline with direct domain routing (bypass gating network)
            result = process_query_with_domain(question, domain)
            
            if result.get("status") == "success":
                generated_answer = result.get("response", "")
                
                # Create updated record without timestamp
                updated_record = {
                    "question": question,
                    "answer": generated_answer,
                    "domain": domain,
                }
                
                updated_records.append(updated_record)
                success_count += 1
                
                # Show a preview of the answer
                answer_preview = generated_answer[:80].replace('\n', ' ')
                print(f"    [OK] Answer: {answer_preview}...")
                
            else:
                # If generation failed, keep original answer if exists
                error_msg = result.get('message', 'Unknown error')
                print(f"    [ERROR] Generation failed: {error_msg}")
                
                record_copy = {k: v for k, v in record.items() if k != "timestamp"}
                updated_records.append(record_copy)
                error_count += 1
                
        except Exception as e:
            print(f"    [ERROR] Exception: {type(e).__name__}: {e}")
            # Keep original answer without timestamp
            record_copy = {k: v for k, v in record.items() if k != "timestamp"}
            updated_records.append(record_copy)
            error_count += 1
    
    # Write updated records back to file
    print("\n" + "-" * 60)
    print(f"💾 Writing {len(updated_records)} records to {output_file}...")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in updated_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"[Save] Successfully wrote to {output_file}")
        
        # Print statistics
        print("\n[Stats] Statistics:")
        print(f"  Total processed: {len(records)}")
        print(f"  OK: {success_count}")
        print(f"  ERRORS: {error_count}")
        print(f"  Success rate: {success_count/len(records)*100:.1f}%")
        
    except IOError as e:
        print(f"[ERROR] Error writing to {output_file}: {e}")
        return
    
    print("\n[Done] All done!")


def preview_file(file_path="query_log.jsonl", num_lines=5):
    """Preview the first few lines of the JSONL file."""
    print(f"\n[Preview] {file_path} (first {num_lines} lines):")
    print("-" * 60)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                try:
                    data = json.loads(line)
                    print(f"  {i+1}. Question: {data.get('question', '')[:50]}...")
                    print(f"     Domain: {data.get('domain', 'unknown')}")
                    print(f"     Answer length: {len(data.get('answer', ''))} chars")
                except:
                    print(f"  {i+1}. {line[:80]}...")
    except FileNotFoundError:
        print(f"  File {file_path} not found")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate answers for query_log.jsonl")
    parser.add_argument("--input", default="query_log.jsonl", help="Input JSONL file")
    parser.add_argument("--output", default="query_log.jsonl", help="Output JSONL file")
    parser.add_argument("--no-backup", action="store_true", help="Don't backup original file")
    parser.add_argument("--max", type=int, help="Maximum number of questions to process (for testing)")
    parser.add_argument("--preview", action="store_true", help="Preview file before processing")
    
    args = parser.parse_args()
    
    if args.preview:
        preview_file(args.input)
    else:
        print("=" * 60)
        print("  AdaptRoute - Answer Generator for Query Log")
        print("=" * 60)
        
        generate_answers_for_query_log(
            input_file=args.input,
            output_file=args.output,
            backup_original=not args.no_backup,
            max_questions=args.max
        )