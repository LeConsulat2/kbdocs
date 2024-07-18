import os
import pandas as pd
from docx import Document

# Load the Excel file from the correct local path
file_path = r"C:\Users\Jonathan\Documents\kbdocs\all-item.xlsx"
excel_data = pd.read_excel(file_path)

# Define the directory where the files should be saved
save_directory = r"C:\Users\Jonathan\Documents\kbdocs\here"

# Check if the directory exists, if not, create it
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Iterate through each row in the DataFrame and save the Word documents in the specified directory
for index, row in excel_data.iterrows():
    # Get the content and name from the respective columns
    content = str(row["Content"]) if pd.notna(row["Content"]) else ""
    name = str(row["Name"]) if pd.notna(row["Name"]) else "Unnamed"

    # Create a new Document
    doc = Document()

    # Add the content to the Document
    doc.add_paragraph(content)

    # Ensure the filename is valid
    valid_filename = "".join(x for x in name if x.isalnum() or x in "._- ")
    file_path = os.path.join(save_directory, f"{valid_filename}.docx")

    # Save the Document with the name from the 'Name' column
    doc.save(file_path)

print("Word documents created successfully.")
