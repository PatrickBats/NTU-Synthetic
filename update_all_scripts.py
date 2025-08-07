#!/usr/bin/env python3
import os
import glob

def get_test_name_from_path(path):
    parts = path.split(os.sep)
    if 'Class' in parts[-2]:
        return 'Class'
    
    test_type = ''
    # Get main attribute type (Color, Size, Texture, or combinations)
    for part in parts:
        if any(x in part for x in ['Color', 'Size', 'Texture']):
            test_type = part
            break
    
    # Get whether it's Same-Class or Different-Class
    class_type = 'Same-Class' if 'SameClass' in path else 'Different-Class'
    
    # Get whether it's Different-X (where X is Color, Size, etc)
    diff_type = parts[-2].split('_')[-1]
    
    return f"{class_type}-{diff_type}"

def update_script(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Skip if already updated
    if 'check_existing_results' in content:
        return
        
    test_name = get_test_name_from_path(file_path)
    
    check_function = '''def check_existing_results(model_name, test_name):
    if os.path.exists(MASTER_CSV):
        df = pd.read_csv(MASTER_CSV)
        existing = df[(df['Model'] == model_name) & (df['Test'] == test_name)]
        return len(existing) > 0
    return False

'''
    
    # Add check_existing_results function
    content = content.replace('def main():', check_function + 'def main():')
    
    # Add test name and check logic
    if 'CVCL' in file_path:
        content = content.replace(
            '    # ─── load CSV & model ───\n    df = pd.read_csv(CSV_PATH)',
            f'''    # ─── load CSV & model ───
    test_name = '{test_name}'
    
    # Check if results already exist
    if check_existing_results(args.model, test_name):
        print(f"[ℹ️] Results for {{args.model}} on {{test_name}} already exist in {{MASTER_CSV}}. Skipping...")
        return
        
    df = pd.read_csv(CSV_PATH)'''
        )
        # Update test name in results
        content = content.replace(
            "'Test':     'Color-Prototype'",
            "'Test':     test_name"
        )
    else:  # CLIP files
        model_type = 'clip-resnext' if 'CLIP' in file_path else 'cvcl-resnext'
        content = content.replace(
            f"    # ─── force {model_type.split('-')[0].upper()} model ───\n    model_name = '{model_type}'",
            f'''    # ─── force {model_type.split('-')[0].upper()} model ───
    model_name = '{model_type}'
    test_name = '{test_name}'
    
    # Check if results already exist
    if check_existing_results(model_name, test_name):
        print(f"[ℹ️] Results for {{model_name}} on {{test_name}} already exist in {{MASTER_CSV}}. Skipping...")
        return'''
        )
        # Update test name in results
        content = content.replace(
            f"'Test':     '{test_name.split('-')[0]}-Prototype-{model_type.split('-')[0].upper()}'",
            "'Test':     test_name"
        )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Updated {file_path}")

def main():
    # Get all Python files in KonkLab_Classification
    base_path = os.path.join('PatrickProject', 'KonkLab_Classification')
    for py_file in glob.glob(os.path.join(base_path, '**', '*.py'), recursive=True):
        if not py_file.endswith('__init__.py'):
            update_script(py_file)

if __name__ == '__main__':
    main()