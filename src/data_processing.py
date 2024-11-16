import os
import json
import tiktoken

# Set up the base directory and storage location
base_dir = r'dev-clean\LibriSpeech\dev-clean'
storage_dir = r'./data'

# Initialize tiktoken encoder
encoder = tiktoken.get_encoding("gpt2")  # Using OpenAI's GPT-2 encoding

def process_folder(folder_path):
    """Process a folder containing .flac and .txt files."""
    try:
        # Get the list of files in the folder
        files = os.listdir(folder_path)
        
        print(f"\nProcessing folder: {folder_path}")
        
        # Extract speaker and chapter IDs from the path
        path_parts = folder_path.split(os.sep)
        speaker_id = path_parts[-2]
        chapter_id = path_parts[-1]
        
        # Find the .txt file (should be the trans.txt file)
        txt_files = [f for f in files if f.endswith('.trans.txt')]
        if not txt_files:
            print(f"No transcript file found in {folder_path}")
            return 0  # Return count of processed files
            
        txt_file = txt_files[0]
        txt_path = os.path.join(folder_path, txt_file)
        
        # Find all .flac files
        flac_files = [f for f in files if f.endswith('.flac')]
        if not flac_files:
            print(f"No .flac files found in {folder_path}")
            return 0
            
        print(f"Found {len(flac_files)} .flac files and transcript file: {txt_file}")
        
        # Read the transcript file
        with open(txt_path, 'r', encoding='utf-8') as f:
            transcripts = {}
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    file_id, text = parts
                    # Encode the text using tiktoken
                    encoded_text = encoder.encode(text)
                    transcripts[file_id] = {
                        "text": text,
                        "tokens": list(encoded_text)
                    }

        # Process each .flac file and create individual JSON files
        successful_processed = 0
        for flac_file in flac_files:
            file_id = os.path.splitext(flac_file)[0]
            if file_id in transcripts:
                # Create mapping for this specific audio file
                audio_path = os.path.join(folder_path, flac_file)
                mapping = {
                    "audio_path": audio_path,
                    "text": transcripts[file_id]["text"],
                    "tokens": transcripts[file_id]["tokens"],
                    "metadata": {
                        "speaker_id": speaker_id,
                        "chapter_id": chapter_id,
                        "file_id": file_id
                    }
                }
                
                # Create JSON file name with speaker and chapter info to ensure uniqueness
                json_filename = f"{speaker_id}-{chapter_id}-{file_id}.json"
                json_path = os.path.join(storage_dir, json_filename)
                
                # Save individual JSON file
                with open(json_path, 'w', encoding='utf-8') as json_file:
                    json.dump(mapping, json_file, ensure_ascii=False, indent=4)
                
                successful_processed += 1
                print(f"Created: {json_filename}")
                print(f"        Text: {mapping['text'][:50]}...")
                print(f"        Tokens: {mapping['tokens'][:10]}... (total tokens: {len(mapping['tokens'])})")
            else:
                print(f"Warning: No transcript found for {flac_file}")
        
        return successful_processed
                
    except Exception as e:
        print(f"Error processing folder {folder_path}: {str(e)}")
        return 0

# Create storage directory if it doesn't exist
os.makedirs(storage_dir, exist_ok=True)

# Process the folders
total_processed = 0
total_files = 0
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
                files_processed = process_folder(second_folder_path)
                total_files += files_processed

# Print final statistics
if total_files > 0:
    print(f"\nSuccess! Processing complete.")
    print(f"Total chapters processed: {total_processed}")
    print(f"Total individual JSON files created: {total_files}")
    print(f"Files are stored in: {storage_dir}")
else:
    print("\nError: No files were processed successfully. Check the error messages above.")