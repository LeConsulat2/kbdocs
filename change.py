import pandas as pd

# Define the file paths
json_file_path = r"C:\Users\Jonathan\Documents\sql-master/movies-data.json"  # Path to the input JSON file
csv_file_path = r"C:\Users\Jonathan\Documents\sql-master/movies-data.csv"  # Path to save the output CSV file

# Read the JSON file into a pandas DataFrame
df = pd.read_json(json_file_path)

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False, encoding="utf-8-sig")

print(f"Data successfully converted and saved to {csv_file_path}")
