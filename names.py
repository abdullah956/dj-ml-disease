import pandas as pd
import json

# Load the CSV file
def extract_diseases(csv_file, json_file):
    df = pd.read_csv(csv_file)
    
    # Extract unique disease names (assuming the second column contains diseases)
    disease_names = df.iloc[:, 0].unique().tolist()
    
    # Save to JSON
    with open(json_file, "w") as f:
        json.dump(disease_names, f, indent=4)

# Example usage
csv_file = "dataset.csv"  # Update with your file path
json_file = "disease_names.json"
extract_diseases(csv_file, json_file)

print(f"Disease names extracted and saved to {json_file}")
