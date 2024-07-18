import os
import zipfile
import json
import pandas as pd

# Paths and variables
data_path = r'C:\Users\Dilfina\OneDrive\Desktop\dataset'  # Update this with the actual path to your zip folders
output_csv = 'output.csv'

# Initialize a list to store all data
data_list = []

# Function to process each JSON file and extract relevant data
def process_json(json_content, video_id):
    frames = json_content['frames']
    data = []
    for frame_id, frame_data in frames.items():
        arousal = frame_data['arousal']
        valence = frame_data['valence']
        landmarks = frame_data['landmarks']
        data.append([video_id, frame_id, arousal, valence, landmarks])
    return data

# Extract data from zip files
for zip_filename in os.listdir(data_path):
    if zip_filename.endswith('.zip'):
        with zipfile.ZipFile(os.path.join(data_path, zip_filename), 'r') as zip_ref:
            zip_ref.extractall('temp_extracted')
            for root, _, files in os.walk('temp_extracted'):
                for file in files:
                    if file.endswith('.json'):
                        with open(os.path.join(root, file), 'r') as json_file:
                            json_content = json.load(json_file)
                            video_id = json_content['video_id']
                            data = process_json(json_content, video_id)
                            data_list.extend(data)
            # Clean up the temporary extraction directory
            for root, dirs, files in os.walk('temp_extracted', topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))

# Create DataFrame from the data list
df = pd.DataFrame(data_list, columns=['video_id', 'frame', 'arousal', 'valence', 'landmarks'])

# Save the DataFrame to a CSV file
df.to_csv(output_csv, index=False)

