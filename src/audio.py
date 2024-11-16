import os
import json
 
base_dir = r'F:\works\A-important\A-neurals\New folder\dev-clean\LibriSpeech\dev-clean'
storage_dir = r'F:\works\A-important\A-neurals\New folder\data'
 
audio_text_mapping = {}

def process_folder(folder_path):
    """Process a folder containing .flac and .txt files."""
    try: 
        files = os.listdir(folder_path)
        
        print(f"\nProcessing folder: {folder_path}")
         
        txt_files = [f for f in files if f.endswith('.trans.txt')]
        if not txt_files:
            print(f"No transcript file found in {folder_path}")
            return
            
        txt_file = txt_files[0]
        txt_path = os.path.join(folder_path, txt_file)
         
        flac_files = [f for f in files if f.endswith('.flac')]
        if not flac_files:
            print(f"No .flac files found in {folder_path}")
            return
            
        print(f"Found {len(flac_files)} .flac files and transcript file: {txt_file}")
         
        with open(txt_path, 'r', encoding='utf-8') as f:
            transcripts = {}
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    file_id, text = parts 
                    transcripts[file_id] = text
 
        for flac_file in flac_files:
            file_id = os.path.splitext(flac_file)[0]  
            if file_id in transcripts:
                audio_path = os.path.join(folder_path, flac_file)
                audio_text_mapping[audio_path] = transcripts[file_id]
                print(f"Mapped: {flac_file} -> {transcripts[file_id][:50]}...")
            else:
                print(f"Warning: No transcript found for {flac_file}")
                
    except Exception as e:
        print(f"Error processing folder {folder_path}: {str(e)}")
 
os.makedirs(storage_dir, exist_ok=True)
 
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
 
if audio_text_mapping:
    output_json_path = os.path.join(storage_dir, 'audio_text_mapping.json')
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(audio_text_mapping, json_file, ensure_ascii=False, indent=4)
    print(f"\nSuccess! JSON file has been saved to {output_json_path}")
    print(f"Total mappings created: {len(audio_text_mapping)}")
else:
    print("\nError: No mappings were created. Check the error messages above.")