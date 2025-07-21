import pandas as pd

# Load your dataset (replace 'your_file.csv' with your actual filename)
df = pd.read_csv('/home/patrick/ssd/discover-hidden-visual-concepts/PatrickProject/testdata.csv')

# Extract unique class names
unique_classes = df['Class'].dropna().unique()

# Convert to DataFrame
unique_df = pd.DataFrame(unique_classes, columns=['Class'])

# Save to a new CSV file
unique_df.to_csv('unique_class_names.csv', index=False)

print("Unique class names saved to 'unique_class_names.csv'")
