"""
Script to generate answers for all questions in query_log.jsonl using the pipeline.
Updates answers and removes timestamps.
"""

import json
from pipeline import process_query


def generate_answers_for_query_log(input_file="query_log.jsonl", output_file="query_log.jsonl"):
    """
    Reads questions from query_log.jsonl, generates answers using the pipeline,
    and writes back the updated records without timestamps.
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSONL file (defaults to overwriting input)
    """
    records = []
    
    # Read all records from the JSONL file
    print(f"Reading {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    records.append(json.loads(line))
        print(f"✓ Read {len(records)} records")
    except FileNotFoundError:
        print(f"✗ Error: {input_file} not found")
        return
    except json.JSONDecodeError as e:
        print(f"✗ Error parsing JSON at line {line_num}: {e}")
        return
    
    # Process each record
    print("\nGenerating answers...")
    updated_records = []
    
    for idx, record in enumerate(records, 1):
        question = record.get("question", "")
        domain = record.get("domain", "unknown")
        
        if not question:
            print(f"  [{idx}/{len(records)}] Skipping record with empty question")
            # Keep the record but remove timestamp
            record_copy = {k: v for k, v in record.items() if k != "timestamp"}
            updated_records.append(record_copy)
            continue
        
        try:
            print(f"  [{idx}/{len(records)}] Processing: {question[:50]}...")
            
            # Call the pipeline to generate answer
            result = process_query(question)
            
            if result.get("status") == "success":
                generated_answer = result.get("response", "")
                routing_mode = result.get("routing_mode", "unknown")
                adapter_used = result.get("adapter_used", "base_model")
                
                # Create updated record without timestamp
                updated_record = {
                    "question": question,
                    "answer": generated_answer,
                    "domain": domain,
                }
                
                updated_records.append(updated_record)
                print(f"    ✓ Generated (router: {routing_mode}, adapter: {adapter_used})")
            else:
                # If generation failed, keep original answer and remove timestamp
                print(f"    ✗ Generation failed: {result.get('message', 'Unknown error')}")
                record_copy = {k: v for k, v in record.items() if k != "timestamp"}
                updated_records.append(record_copy)
                
        except Exception as e:
            print(f"    ✗ Error processing question: {e}")
            # Keep original answer without timestamp
            record_copy = {k: v for k, v in record.items() if k != "timestamp"}
            updated_records.append(record_copy)
    
    # Write updated records back to file
    print(f"\nWriting {len(updated_records)} records to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in updated_records:
                f.write(json.dumps(record) + '\n')
        print(f"✓ Successfully wrote to {output_file}")
    except IOError as e:
        print(f"✗ Error writing to {output_file}: {e}")
        return
    
    print("\n✓ All done!")


if __name__ == "__main__":
    generate_answers_for_query_log()
