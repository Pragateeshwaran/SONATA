import os
import json
import tiktoken

# Set up the base directory and storage location
base_dir = r'dev-clean\LibriSpeech\dev-clean'
storage_dir = r'./data'

# Initialize tiktoken encoder
encoder = tiktoken.get_encoding("gpt2")  # Using OpenAI's GPT-2 encoding

# Initialize the final dictionary to store the mappings
audio_text_mapping = {}

def process_folder(folder_path):
    """Process a folder containing .flac and .txt files."""
    try:
        # Get the list of files in the folder
        files = os.listdir(folder_path)
        
        print(f"\nProcessing folder: {folder_path}")
        
        # Find the .txt file (should be the trans.txt file)
        txt_files = [f for f in files if f.endswith('.trans.txt')]
        if not txt_files:
            print(f"No transcript file found in {folder_path}")
            return
            
        txt_file = txt_files[0]
        txt_path = os.path.join(folder_path, txt_file)
        
        # Find all .flac files
        flac_files = [f for f in files if f.endswith('.flac')]
        if not flac_files:
            print(f"No .flac files found in {folder_path}")
            return
            
        print(f"Found {len(flac_files)} .flac files and transcript file: {txt_file}")
        
        # Read the transcript file
        with open(txt_path, 'r', encoding='utf-8') as f:
            transcripts = {}
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    file_id, text = parts
                    # Encode the text using tiktoken - returns a list of integers
                    encoded_text = encoder.encode(text)
                    # Convert to regular list right away since encoded_text is already list-like
                    transcripts[file_id] = {
                        "text": text,  # Original text
                        "tokens": list(encoded_text)  # Convert to regular list - no need for tolist()
                    }

        # Process each .flac file
        for flac_file in flac_files:
            file_id = os.path.splitext(flac_file)[0]  # Remove .flac extension
            if file_id in transcripts:
                audio_path = os.path.join(folder_path, flac_file)
                audio_text_mapping[audio_path] = transcripts[file_id]
                print(f"Mapped: {flac_file} -> Text: {transcripts[file_id]['text'][:50]}...")
                print(f"        Tokens: {transcripts[file_id]['tokens'][:10]}... (total tokens: {len(transcripts[file_id]['tokens'])})")
            else:
                print(f"Warning: No transcript found for {flac_file}")
                
    except Exception as e:
        print(f"Error processing folder {folder_path}: {str(e)}")

# Create storage directory if it doesn't exist
os.makedirs(storage_dir, exist_ok=True)

# Process the folders
total_processed = 0
print("\nProcessing folders...")

for first_folder in os.listdir(base_dir):
    first_folder_path = os.path.join(base_dir, first_folder)
    
    if os.path.isdir(first_folder_path):
        print(f"\nProcessing speaker folder: {first_folder}")
        
        for second_folder in os.listdir(first_folder_path):
            second_folder_path = os.path.join(first_folder_path, second_folder)
            
            if os.path.isdir(second_folder_path):
                total_processed += 1
                print(f"\nProcessing chapter folder {total_processed}: {second_folder}")
                process_folder(second_folder_path)

# Save the mapping to JSON file
if audio_text_mapping:
    output_json_path = os.path.join(storage_dir, 'audio_text_mapping.json')
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(audio_text_mapping, json_file, ensure_ascii=False, indent=4)
    
    # Print some statistics
    total_tokens = sum(len(v['tokens']) for v in audio_text_mapping.values())
    total_texts = len(audio_text_mapping)
    print(f"\nSuccess! JSON file has been saved to {output_json_path}")
    print(f"Total mappings created: {total_texts}")
    print(f"Total tokens across all texts: {total_tokens}")
    print(f"Average tokens per text: {total_tokens/total_texts:.2f}")
else:
    print("\nError: No mappings were created. Check the error messages above.")