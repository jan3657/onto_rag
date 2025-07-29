import os
import argparse
import collections.abc

# We need the pyyaml library to generate YAML output.
# You can install it with: pip install pyyaml
try:
    import yaml
except ImportError:
    print("Error: The 'pyyaml' library is required to generate YAML output.")
    print("Please install it using: pip install pyyaml")
    exit(1)


# --- Configuration ---
# Files or directories to explicitly skip by their exact name or path (relative to root_dir)
EXCLUDE_ITEMS_EXACT = {
    ".git",
    "__pycache__",
    "data",               # <<< Exclude the entire data directory
    "docs",               # <<< Exclude the entire docs directory
    "ontologies",         # <<< Exclude the entire ontologies directory
    "api_key.json",       # Sensitive file
    ".env",               # Sensitive file (though .env.example is fine)
    "models",
    ".DS_Store",
    "tempCodeRunnerFile.py",
    "logs",
    ".devcontainer",
    ".pytest_cache",
    "evaluation_results_llama.json",
    "evaluation_results_gemini_1.5-flash.json",
    "export_files_contents.py",  # The script itself
    # README.md is now handled specially: its content is moved to the summary
    "README.md",
    "project_contents_for_llm.yaml", # The output file itself
    "project_contents_for_llm.txt",  # Another potential output file name
    # Add other specific files or directories if needed
    # e.g. "some_large_binary_asset.dat"
}

# File extensions to skip (typically binary or non-text files)
EXCLUDE_EXTENSIONS = {
    ".pyc", ".pyo", ".bin", ".exe", ".dll", ".so", ".o", ".a", ".lib",
    ".jar", ".war", ".ear", ".class", ".swo", ".swp",
    # Image/Media files
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",
    ".mp3", ".wav", ".ogg", ".mp4", ".avi", ".mkv", ".mov",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".rar", ".7z",
    # Other common binary formats
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".ods", ".odp", ".sqlite", ".db",
    ".pkl", ".joblib", # Python pickled objects
    ".pt", ".pth", ".pb", ".onnx", # Model files
    ".DS_Store" # macOS specific
}

# NEW: Files considered important for the project summary
KEY_FILES_FOR_SUMMARY = {
    "requirements.txt",
    "main.py",
    "src/main.py",
    "app.py",
    "config.py",
    "src/config.py",
    ".gitignore",
}

# --- End Configuration ---

def should_skip(item_path, root_dir, output_filename):
    """
    Determines if a file or directory should be skipped for CONTENT inclusion.
    This is now used to decide whether to read a file's content and to mark
    items as '(excluded)' in the directory tree.
    """
    # Always get the relative path from the true root of the scan
    relative_item_path = os.path.relpath(item_path, root_dir)
    # Normalize for cross-platform consistency (e.g., convert \ to /)
    normalized_relative_path = relative_item_path.replace(os.sep, "/")

    # Get just the basename (the file or directory name)
    item_basename = os.path.basename(item_path)

    # Skip the output file itself
    if item_basename == output_filename:
        return True

    # Check against exact items to exclude (handles both files and directories)
    for excluded in EXCLUDE_ITEMS_EXACT:
        normalized_excluded = excluded.replace(os.sep, "/")
        # Match if the path is exactly the excluded item,
        # or if the path starts with an excluded directory.
        if normalized_relative_path == normalized_excluded or \
           normalized_relative_path.startswith(normalized_excluded + "/"):
            return True

    # If it's a file, check its extension
    if os.path.isfile(item_path):
        _, ext = os.path.splitext(item_path)
        if ext.lower() in EXCLUDE_EXTENSIONS:
            return True

    return False

# NEW: Helper to insert data into a nested dictionary based on path
def set_nested_dict_value(d, path, value):
    """
    Sets a value in a nested dictionary based on a path list.
    e.g., set_nested_dict_value({}, ['src', 'pipeline', 'main.py'], 'content')
    """
    keys = path.split('/')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def generate_project_context(root_dir, output_filename="project_contents_for_llm.yaml"):
    """
    Generates a structured YAML file containing a project summary and
    a hierarchical representation of file contents. The directory structure
    will now include excluded files and directories for context.
    """
    root_dir_abs = os.path.abspath(root_dir)
    output_file_abs_path = os.path.join(root_dir_abs, output_filename)

    # Add the final output file to the exclusion list to be safe
    EXCLUDE_ITEMS_EXACT.add(output_filename)

    print(f"Starting project context generation for: {root_dir_abs}")
    print(f"Output will be saved to: {output_file_abs_path}")
    print("-" * 30)

    # --- Data structures for the enhanced YAML ---
    project_summary = {
        "project_name": os.path.basename(root_dir_abs),
        "readme_content": None,
        "key_files": {},
        "directory_structure": ""
    }
    file_contents_hierarchical = {}
    tree_lines = []
    readme_path = os.path.join(root_dir_abs, "README.md")
    collected_files_count = 0

    # --- Special handling for README.md ---
    if os.path.exists(readme_path):
        try:
            with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                project_summary["readme_content"] = f.read()
            print("Extracted content from README.md for summary.")
        except Exception as e:
            project_summary["readme_content"] = f"[Error reading README.md: {e}]"

    # --- MODIFIED: Main walk to collect ALL files/dirs for the tree, and filter for content ---
    for dirpath, dirnames, filenames in os.walk(root_dir_abs, topdown=True):
        # MODIFIED: We no longer prune the directories we walk into, with one exception.
        # We want to see that .git exists, but walking it is very slow and unnecessary.
        # So we remove it from `dirnames` to prevent `os.walk` from descending into it.
        if ".git" in dirnames:
            dirnames.remove(".git")

        # Sort for consistent order
        dirnames.sort()
        filenames.sort()
        
        # MODIFIED: Determine if the *current directory itself* is inside an excluded path.
        # This is the key to marking all contents of an excluded dir properly.
        is_current_dir_excluded = should_skip(dirpath, root_dir_abs, output_filename)

        level = dirpath.replace(root_dir_abs, '').count(os.sep)
        indent = ' ' * 4 * level
        
        # Display the current directory in the tree
        if level > 0 or dirpath == root_dir_abs:
            dir_display_name = os.path.basename(dirpath)
            # Don't add a name for the root directory itself in the tree
            if dir_display_name != os.path.basename(root_dir_abs):
                marker = " (excluded)" if is_current_dir_excluded else ""
                tree_lines.append(f"{indent}├── {dir_display_name}/{marker}")

        sub_indent = ' ' * 4 * (level + 1)
        # Combine directories and files to draw the tree correctly
        all_items = [d + "/" for d in dirnames] + filenames
        
        for i, item_name in enumerate(all_items):
            is_last = (i == len(all_items) - 1)
            prefix = "└── " if is_last else "├── "
            
            item_abs_path = os.path.join(dirpath, item_name.strip('/'))

            # Check if this specific item should be skipped.
            is_item_explicitly_excluded = should_skip(item_abs_path, root_dir_abs, output_filename)
            
            # An item's content is skipped if it's explicitly excluded OR if its parent directory is excluded.
            is_content_skipped = is_current_dir_excluded or is_item_explicitly_excluded

            # Add to tree view
            marker = " (excluded)" if is_content_skipped else ""
            tree_lines.append(f"{sub_indent}{prefix}{item_name}{marker}")

            # If it's a file and its content should be included, process it.
            if not item_name.endswith('/') and not is_content_skipped:
                relative_file_path = os.path.relpath(item_abs_path, root_dir_abs)
                display_path = relative_file_path.replace(os.sep, "/")
                print(f"Processing file: {display_path}")

                try:
                    with open(item_abs_path, "r", encoding="utf-8", errors="ignore") as infile:
                        content = infile.read()
                        # Add to hierarchical dict
                        set_nested_dict_value(file_contents_hierarchical, display_path, content)
                        # Check if it's a key file for the summary
                        if display_path in KEY_FILES_FOR_SUMMARY:
                            project_summary["key_files"][display_path] = content
                except Exception as e:
                    set_nested_dict_value(file_contents_hierarchical, display_path, f"[Error reading file: {e}]")

                collected_files_count += 1
                
    project_summary["directory_structure"] = "\n".join(tree_lines)

    # --- Assemble the final YAML structure ---
    final_yaml_data = {
        "project_summary": project_summary,
        "file_contents": file_contents_hierarchical
    }

    # --- Write to YAML file ---
    try:
        with open(output_file_abs_path, "w", encoding="utf-8") as outfile:
            # Using '|' style for multiline strings and ensuring proper indentation
            yaml.dump(final_yaml_data, outfile, sort_keys=False, default_flow_style=False, allow_unicode=True, indent=2, default_style='|')
    except Exception as e:
        print(f"\n---!!! An error occurred while writing the YAML file: {e} !!!---")

    print("-" * 30)
    print("Project context generation complete.")
    print(f"Scanned all files for the directory tree.")
    print(f"Collected content from {collected_files_count} files.")
    print(f"Output saved to: {output_file_abs_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scans a project directory and saves a structured summary and hierarchical file contents to a YAML file for optimal LLM context. The directory tree will show all files, marking excluded ones."
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=".",
        help="The root directory of the project to scan (default: current directory).",
    )
    parser.add_argument(
        "--output",
        default="project_contents_for_llm.yaml",
        help="The name of the output file (default: project_contents_for_llm.yaml).",
    )
    args = parser.parse_args()

    generate_project_context(args.root_dir, args.output)