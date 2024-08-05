import os
import pandas as pd
from docx import Document

# Load the Excel file from the provided path
file_path = r"C:\Users\Jonathan\Documents\kbdocs\3-items.xlsx"
excel_data = pd.read_excel(file_path)

# Define the base save directory
base_save_directory = r"C:\Users\Jonathan\Documents\kbdocs\staff"

# Iterate through each row in the DataFrame and save the Word documents in the specified directories
for index, row in excel_data.iterrows():
    # Get the content, name, and business owner from the respective columns
    content = str(row["Content"]) if pd.notna(row["Content"]) else ""
    name = str(row["Name"]) if pd.notna(row["Name"]) else "Unnamed"
    business_owner = (
        str(row["Business owner"]) if pd.notna(row["Business owner"]) else "Unknown"
    )

    # Create a directory for the business owner if it doesn't exist
    business_owner_directory = os.path.join(base_save_directory, business_owner)
    if not os.path.exists(business_owner_directory):
        os.makedirs(business_owner_directory)

    # Create a new Document
    doc = Document()

    # Add the content to the Document
    doc.add_paragraph(content)

    # Ensure the filename is valid
    valid_filename = "".join(x for x in name if x.isalnum() or x in "._- ")
    file_path = os.path.join(business_owner_directory, f"{valid_filename}.docx")

    # Save the Document with the name from the 'Name' column
    doc.save(file_path)

print("Word documents created successfully.")
