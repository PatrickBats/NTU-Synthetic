# fix_labels.py
import pandas as pd

# 1) Load your raw labels CSV
#    Adjust the path if needed.
df = pd.read_csv('/home/patrick/ssd/discover-hidden-visual-concepts/PatrickProject/testdata.csv')

# 2) Define mappings for the known typos
color_mapping = {
    'muticolored': 'multicolored',
    'multicoloredred': 'multicolored',
    'blak': 'black', 'gray': 'grey'
}

texture_mapping = {
    'texured': 'textured'
}

# 3) Apply the mappings
df['Color']   = df['Color'].str.lower().replace(color_mapping)
df['Texture'] = df['Texture'].str.lower().replace(texture_mapping)

# 4) (Optional) Capitalize the first letter of each color for consistency
df['Color'] = df['Color'].str.capitalize()

# 5) Write out the cleaned CSV
df.to_csv('PatrickProject/cleaned_labels.csv', index=False)

print("Cleaned labels written to PatrickProject/cleaned_labels.csv")
