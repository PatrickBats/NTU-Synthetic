#!/usr/bin/env python3
"""
UNIVERSAL RENAMING SCRIPT FOR SYNTHETIC KONKLE DATASET
======================================================

TO USE THIS SCRIPT:
1. Change OLD_NAME to what you want to replace (e.g., "breadloaf", "jack-o-lantern")
2. Change NEW_NAME to what you want it to become (e.g., "bread", "pumpkin")
3. Run: python universal_rename_script.py

The script will:
- Rename PNG files in the color folder (e.g., breadloaf_color -> bread_color)
- Rename PNG files in the bases folder (e.g., breadloaf_bases -> bread_bases)
- Update all entries in master_labels.csv
- Handle the folder renaming if needed
"""

# ============================================
# CHANGE THESE TWO LINES FOR YOUR NEEDS:
# ============================================
OLD_NAME = "breadloaf"  # <-- CHANGE THIS to what you want to replace
NEW_NAME = "bread"  # <-- CHANGE THIS to what it should become

# ============================================
# DON'T CHANGE ANYTHING BELOW THIS LINE
# ============================================

import os
import shutil

def rename_folders():
    """Rename the folders themselves if needed"""
    base_path = 'data/SyntheticKonkle'
    folders_renamed = []
    
    # Check for color folder
    old_color_folder = os.path.join(base_path, f"{OLD_NAME}_color")
    new_color_folder = os.path.join(base_path, f"{NEW_NAME}_color")
    
    if os.path.exists(old_color_folder):
        print(f"Renaming folder: {OLD_NAME}_color -> {NEW_NAME}_color")
        os.rename(old_color_folder, new_color_folder)
        folders_renamed.append((old_color_folder, new_color_folder))
    
    # Check for bases folder
    old_bases_folder = os.path.join(base_path, f"{OLD_NAME}_bases")
    new_bases_folder = os.path.join(base_path, f"{NEW_NAME}_bases")
    
    if os.path.exists(old_bases_folder):
        print(f"Renaming folder: {OLD_NAME}_bases -> {NEW_NAME}_bases")
        os.rename(old_bases_folder, new_bases_folder)
        folders_renamed.append((old_bases_folder, new_bases_folder))
    
    return folders_renamed

def rename_png_files():
    """Rename all PNG files from OLD_NAME to NEW_NAME"""
    base_path = 'data/SyntheticKonkle'
    
    # Look for the folders (they might have new names if we renamed them)
    folders_to_check = [
        os.path.join(base_path, f"{NEW_NAME}_color"),
        os.path.join(base_path, f"{NEW_NAME}_bases"),
        # Also check if they still have old names
        os.path.join(base_path, f"{OLD_NAME}_color"),
        os.path.join(base_path, f"{OLD_NAME}_bases"),
    ]
    
    total_renamed = 0
    
    for folder_path in folders_to_check:
        if not os.path.exists(folder_path):
            continue
            
        print(f"\nChecking folder: {folder_path}")
        folder_renamed = 0
        
        for filename in os.listdir(folder_path):
            if OLD_NAME in filename and filename.endswith('.png'):
                new_filename = filename.replace(OLD_NAME, NEW_NAME)
                
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_filename)
                
                print(f"  Renaming: {filename} -> {new_filename}")
                os.rename(old_path, new_path)
                
                folder_renamed += 1
                total_renamed += 1
                
                # Show first few examples
                if folder_renamed <= 2:
                    print(f"    Example: {OLD_NAME}_large_bumpy_01_red.png -> {NEW_NAME}_large_bumpy_01_red.png")
        
        if folder_renamed > 0:
            print(f"  Renamed {folder_renamed} files in this folder")
    
    return total_renamed

def update_csv():
    """Update master_labels.csv to replace OLD_NAME with NEW_NAME"""
    csv_path = 'data/SyntheticKonkle/master_labels.csv'
    
    # Read the CSV
    with open(csv_path, 'r') as f:
        content = f.read()
    
    # Count how many replacements we'll make
    count = content.count(OLD_NAME)
    
    if count == 0:
        print(f"No instances of '{OLD_NAME}' found in master_labels.csv")
        return 0
    
    print(f"Found {count} instances of '{OLD_NAME}' in master_labels.csv")
    
    # Replace all instances
    content = content.replace(OLD_NAME, NEW_NAME)
    
    # Write back
    with open(csv_path, 'w') as f:
        f.write(content)
    
    print(f"Replaced all {count} instances with '{NEW_NAME}'")
    
    # Show some examples
    print("\nExample CSV changes:")
    print(f"  {OLD_NAME}_color,{OLD_NAME}_large_bumpy_01_red.png,{OLD_NAME},red,large,bumpy,1")
    print(f"  becomes:")
    print(f"  {NEW_NAME}_color,{NEW_NAME}_large_bumpy_01_red.png,{NEW_NAME},red,large,bumpy,1")
    
    return count

def update_labels_csv_in_folders():
    """Update labels.csv files inside the color and bases folders"""
    base_path = 'data/SyntheticKonkle'
    updated_files = 0
    
    # Check both color and bases folders
    folders_to_check = [
        os.path.join(base_path, f"{NEW_NAME}_color"),
        os.path.join(base_path, f"{NEW_NAME}_bases"),
    ]
    
    for folder_path in folders_to_check:
        labels_file = os.path.join(folder_path, 'labels.csv')
        
        if os.path.exists(labels_file):
            print(f"\nUpdating labels.csv in {os.path.basename(folder_path)}")
            
            # Read the labels.csv
            with open(labels_file, 'r') as f:
                content = f.read()
            
            # Count replacements
            count = content.count(OLD_NAME)
            
            if count > 0:
                # Replace all instances
                content = content.replace(OLD_NAME, NEW_NAME)
                
                # Write back
                with open(labels_file, 'w') as f:
                    f.write(content)
                
                print(f"  Replaced {count} instances of '{OLD_NAME}' with '{NEW_NAME}'")
                updated_files += 1
            else:
                print(f"  No instances of '{OLD_NAME}' found")
    
    return updated_files

def main():
    print("="*70)
    print(f"UNIVERSAL RENAME SCRIPT")
    print(f"Replacing: '{OLD_NAME}' -> '{NEW_NAME}'")
    print("="*70)
    
    if OLD_NAME == "example_old" or NEW_NAME == "example_new":
        print("\n⚠️  ERROR: You need to change OLD_NAME and NEW_NAME at the top of the script!")
        print("Open this script and edit lines 20-21")
        return
    
    print(f"\nThis will:")
    print(f"1. Rename folders from {OLD_NAME}_color/bases to {NEW_NAME}_color/bases")
    print(f"2. Rename all PNG files from {OLD_NAME}_*.png to {NEW_NAME}_*.png")
    print(f"3. Update master_labels.csv to replace all {OLD_NAME} with {NEW_NAME}")
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Step 1: Rename folders
    print("\n📁 STEP 1: Renaming folders...")
    folders_renamed = rename_folders()
    if folders_renamed:
        for old, new in folders_renamed:
            print(f"  ✓ Renamed: {os.path.basename(old)} -> {os.path.basename(new)}")
    else:
        print(f"  No folders found with name '{OLD_NAME}'")
    
    # Step 2: Rename PNG files
    print("\n🖼️  STEP 2: Renaming PNG files...")
    files_renamed = rename_png_files()
    print(f"  ✓ Renamed {files_renamed} PNG files")
    
    # Step 3: Update master CSV
    print("\n📄 STEP 3: Updating master_labels.csv...")
    csv_updates = update_csv()
    print(f"  ✓ Updated {csv_updates} entries in master CSV")
    
    # Step 4: Update labels.csv in folders
    print("\n📄 STEP 4: Updating labels.csv files in folders...")
    labels_updated = update_labels_csv_in_folders()
    print(f"  ✓ Updated {labels_updated} labels.csv files")
    
    # Summary
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print(f"  - Folders renamed: {len(folders_renamed)}")
    print(f"  - PNG files renamed: {files_renamed}")
    print(f"  - Master CSV entries updated: {csv_updates}")
    print(f"  - Folder labels.csv files updated: {labels_updated}")
    print("="*70)
    
    # Verification
    print("\n🔍 Verifying...")
    
    # Check CSV for any remaining old names
    with open('data/SyntheticKonkle/master_labels.csv', 'r') as f:
        content = f.read()
        remaining = content.count(OLD_NAME)
        
    if remaining == 0:
        print(f"  ✓ No '{OLD_NAME}' remaining in CSV")
    else:
        print(f"  ⚠ Warning: {remaining} instances of '{OLD_NAME}' still in CSV")
    
    # Check for any remaining files
    remaining_files = []
    for root, dirs, files in os.walk('data/SyntheticKonkle'):
        for f in files:
            if OLD_NAME in f:
                remaining_files.append(f)
    
    if not remaining_files:
        print(f"  ✓ No files with '{OLD_NAME}' remaining")
    else:
        print(f"  ⚠ Warning: {len(remaining_files)} files still contain '{OLD_NAME}'")
        for f in remaining_files[:3]:
            print(f"    - {f}")

if __name__ == "__main__":
    main()