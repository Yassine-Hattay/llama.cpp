# print_file_name_recursive.py
import os

def scan_files(root_dir=".", extensions=(".py",)):
    """
    Recursively scan files in root_dir with given extensions
    and print their names and contents.
    
    :param root_dir: directory to start from
    :param extensions: tuple of file extensions (e.g. (".py", ".txt"))
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extensions):
                full_path = os.path.join(dirpath, filename)
                
                print("=" * 80)
                print(f"FILE: {full_path}")
                print("=" * 80)
                
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        print(f.read())
                except Exception as e:
                    print(f"[ERROR] Could not read file: {e}")


if __name__ == "__main__":
    # 🔧 Change extensions here
    extensions_to_scan = (".py",".json")  # e.g. (".py", ".txt", ".md")
    
    # 🔧 Change directory if needed
    root_directory = "."  # current working directory
    
    scan_files(root_directory, extensions_to_scan)