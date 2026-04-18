"""
Script to generate answers for all questions in query_log.jsonl using the pipeline.
Updates answers and removes timestamps.
"""

import json
import re
import sys
from pathlib import Path

# Add the current directory to path to import pipeline
sys.path.insert(0, str(Path(__file__).parent))

# Import your pipeline - handle different possible import names
try:
    from pipeline_v4 import process_query
    print("✓ Using pipeline_v4.py")
except ImportError:
    try:
        from pipeline import process_query
        print("✓ Using pipeline.py")
    except ImportError:
        print("✗ Error: Could not find pipeline_v4.py or pipeline.py")
        print("  Make sure your pipeline file is in the same directory")
        sys.exit(1)


def fix_json_line(line):
    """Fix common JSON escape sequence issues."""
    # Fix invalid escape sequences like \* or \**
    line = re.sub(r'\\(\*+)', r'\\u002a\1', line)  # Replace with unicode asterisk
    # Fix other common issues
    line = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', line)  # Escape invalid escapes
    return line


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
        print(f"📋 Creating backup: {backup_file}")
        try:
            import shutil
            shutil.copy2(input_file, backup_file)
            print(f"✓ Backup created at {backup_file}")
        except Exception as e:
            print(f"⚠️ Warning: Could not create backup: {e}")
    
    # Read all records from the JSONL file
    print(f"\n📖 Reading {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
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
                            print(f"  ✓ [Line {line_num}] Fixed JSON issue: {e}")
                        except json.JSONDecodeError as e2:
                            print(f"  ✗ [Line {line_num}] Cannot parse, skipping: {e2}")
                            continue
        print(f"✓ Loaded {len(records)} records")
    except FileNotFoundError:
        print(f"✗ Error: {input_file} not found")
        return
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return
    
    if not records:
        print("✗ No valid records found in file")
        return
    
    # Limit questions if specified
    if max_questions and max_questions < len(records):
        print(f"\n⚠️  Limiting to first {max_questions} questions (of {len(records)})")
        records = records[:max_questions]
    
    # Process each record
    print(f"\n🤖 Generating answers for {len(records)} questions...")
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
            print(f"  [{idx}/{len(records)}] ⚠️  {error_msg}")
            # Keep record but remove timestamp
            record_copy = {k: v for k, v in record.items() if k != "timestamp"}
            updated_records.append(record_copy)
            error_count += 1
            continue
        
        if not question:
            print(f"  [{idx}/{len(records)}] ⚠️  Empty question, skipping")
            record_copy = {k: v for k, v in record.items() if k != "timestamp"}
            updated_records.append(record_copy)
            error_count += 1
            continue
        
        try:
            print(f"  [{idx}/{len(records)}] Processing: {question[:60]}...")
            
            # Call the pipeline to generate answer
            result = process_query(question)
            
            if result.get("status") == "success":
                generated_answer = result.get("response", "")
                
                # Create updated record without timestamp
                updated_record = {
                    "question": question,
                    "answer": generated_answer,
                    "domain": domain,
                }
                
                # Optional: Add metadata if available
                if "routing_mode" in result:
                    updated_record["routing_mode"] = result["routing_mode"]
                if "adapter_used" in result:
                    updated_record["adapter_used"] = result["adapter_used"]
                
                updated_records.append(updated_record)
                success_count += 1
                
                # Show a preview of the answer
                answer_preview = generated_answer[:80].replace('\n', ' ')
                print(f"    ✓ Answer: {answer_preview}...")
                
            else:
                # If generation failed, keep original answer if exists
                error_msg = result.get('message', 'Unknown error')
                print(f"    ✗ Generation failed: {error_msg}")
                
                record_copy = {k: v for k, v in record.items() if k != "timestamp"}
                updated_records.append(record_copy)
                error_count += 1
                
        except Exception as e:
            print(f"    ✗ Exception: {type(e).__name__}: {e}")
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
        print(f"✓ Successfully wrote to {output_file}")
        
        # Print statistics
        print("\n📊 Statistics:")
        print(f"  Total processed: {len(records)}")
        print(f"  ✅ Successful: {success_count}")
        print(f"  ❌ Errors: {error_count}")
        print(f"  Success rate: {success_count/len(records)*100:.1f}%")
        
    except IOError as e:
        print(f"✗ Error writing to {output_file}: {e}")
        return
    
    print("\n✨ All done!")


def preview_file(file_path="query_log.jsonl", num_lines=5):
    """Preview the first few lines of the JSONL file."""
    print(f"\n📄 Preview of {file_path} (first {num_lines} lines):")
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