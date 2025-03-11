import os

def get_folder_size(path):
    """Calculate the total size of a folder in bytes"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if not os.path.islink(file_path):  # Skip symlinks
                total_size += os.path.getsize(file_path)
    return total_size

def human_readable_size(size_bytes):
    """Convert a size in bytes to a human-readable string"""
    if size_bytes == 0:
        return "0 B"
    size_names = ("B", "KB", "MB", "GB", "TB", "PB")
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1
    return f"{size_bytes:.2f} {size_names[i]}"
