import os
import argparse

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
    # Add other specific files or directories if needed
    # e.g. "some_large_binary_asset.dat"
}

# File extensions to skip (typically binary or non-text files)
EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".bin",
    ".exe",
    ".dll",
    ".so",
    ".o",
    ".a",
    ".lib",
    ".jar",
    ".war",
    ".ear",
    ".class",
    ".swo",
    ".swp",
    # Image/Media files
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",
    ".mp3", ".wav", ".ogg", ".mp4", ".avi", ".mkv", ".mov",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".rar", ".7z",
    # Other common binary formats
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".ods", ".odp",
    ".sqlite", ".db",
    ".pkl", ".joblib", # Python pickled objects
    ".pt", ".pth", ".pb", ".onnx", # Model files
    ".DS_Store" # macOS specific
}

# --- End Configuration ---

def should_skip(item_path, root_dir, output_filename):
    """
    Determines if a file or directory should be skipped.
    item_path: absolute path to the item.
    root_dir: absolute path to the project's root directory.
    output_filename: name of the script's output file.
    """
    relative_item_path = os.path.relpath(item_path, root_dir)

    # Skip the output file itself
    if os.path.basename(item_path) == output_filename:
        return True

    # Check against exact items to exclude (can be dir names or file names or relative paths)
    # Normalize paths for comparison
    normalized_relative_item_path = relative_item_path.replace(os.sep, "/")
    for excluded in EXCLUDE_ITEMS_EXACT:
        normalized_excluded = excluded.replace(os.sep, "/")
        if normalized_relative_item_path == normalized_excluded or \
           normalized_relative_item_path.startswith(normalized_excluded + "/"):
            # print(f"Skipping '{relative_item_path}' due to exact match/prefix with '{excluded}'")
            return True
        # Also check just the basename for simple directory/file names at the root level
        # if os.path.dirname(relative_item_path) == "" and os.path.basename(item_path) == excluded:
        # The above check for basename is now effectively covered by the first part of the condition
        # if normalized_relative_item_path (e.g., "data") == normalized_excluded (e.g., "data")


    # If it's a file, check its extension
    if os.path.isfile(item_path):
        _, ext = os.path.splitext(item_path)
        if ext.lower() in EXCLUDE_EXTENSIONS:
            # print(f"Skipping '{relative_item_path}' due to extension '{ext}'")
            return True

    return False

def generate_project_context(root_dir, output_filename="project_contents_for_llm.txt"):
    """
    Generates a text file containing the names and contents of project files.
    """
    root_dir_abs = os.path.abspath(root_dir)
    output_file_abs_path = os.path.join(root_dir_abs, output_filename)


    print(f"Starting project context generation for: {root_dir_abs}")
    print(f"Output will be saved to: {output_file_abs_path}")
    print(f"Excluded items (exact name/path relative to root): {EXCLUDE_ITEMS_EXACT}")
    print(f"Excluded extensions: {EXCLUDE_EXTENSIONS}")
    print("-" * 30)

    collected_files_count = 0
    skipped_files_count = 0
    skipped_dirs_count = 0

    with open(output_file_abs_path, "w", encoding="utf-8", errors="replace") as outfile:
        for dirpath, dirnames, filenames in os.walk(root_dir_abs, topdown=True):
            # Modify dirnames in-place to skip directories
            # This is important for os.walk(topdown=True)
            original_dirnames_count = len(dirnames)
            current_dir_relative_to_root = os.path.relpath(dirpath, root_dir_abs)

            # Filter out directories to be skipped
            dirs_to_keep = []
            for d in dirnames:
                dir_full_path = os.path.join(dirpath, d)
                if not should_skip(dir_full_path, root_dir_abs, output_filename):
                    dirs_to_keep.append(d)
                else:
                    # Print skipped directory relative to root for clarity
                    skipped_dir_relative_path = os.path.relpath(dir_full_path, root_dir_abs).replace(os.sep, "/")
                    print(f"Skipping directory (and its contents): {skipped_dir_relative_path}")
                    skipped_dirs_count += 1
            dirnames[:] = dirs_to_keep


            for filename in filenames:
                file_abs_path = os.path.join(dirpath, filename)
                relative_file_path = os.path.relpath(file_abs_path, root_dir_abs)
                # Normalize for display
                display_path = relative_file_path.replace(os.sep, "/")


                if should_skip(file_abs_path, root_dir_abs, output_filename):
                    # This check might be redundant for files if their parent dir was already skipped,
                    # but good for files directly in an otherwise included dir that match other skip criteria.
                    if not any(display_path.startswith(excluded_dir + "/") for excluded_dir in EXCLUDE_ITEMS_EXACT if os.path.isdir(os.path.join(root_dir_abs, excluded_dir))):
                         print(f"Skipping file: {display_path}") # Only print if not part of an already reported skipped dir
                    skipped_files_count += 1
                    continue

                print(f"Processing file: {display_path}")
                outfile.write(f"--- File: {display_path} ---\n")
                try:
                    with open(file_abs_path, "r", encoding="utf-8", errors="ignore") as infile:
                        content = infile.read()
                        outfile.write(content)
                except Exception as e:
                    outfile.write(f"[Error reading file: {e}]\n")
                outfile.write(f"\n--- END File: {display_path} ---\n\n")
                collected_files_count += 1

    print("-" * 30)
    print(f"Project context generation complete.")
    print(f"Collected content from {collected_files_count} files.")
    print(f"Skipped {skipped_files_count} files (may include files within explicitly skipped dirs).")
    print(f"Skipped {skipped_dirs_count} directories (and their contents).")
    print(f"Output saved to: {output_file_abs_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scans a project directory and saves file names and contents to a text file for LLM context."
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=".",
        help="The root directory of the project to scan (default: current directory).",
    )
    parser.add_argument(
        "--output",
        default="project_contents_for_llm.txt",
        help="The name of the output file (default: project_contents_for_llm.txt).",
    )
    args = parser.parse_args()

    generate_project_context(args.root_dir, args.output)