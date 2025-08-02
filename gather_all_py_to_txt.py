import os
import sys
import subprocess

OUTPUT_FILE = "all_py_code.txt"
SELF_FILE = os.path.basename(__file__)


def get_project_tree(root_dir):
    """获取项目结构（tree /f），兼容Windows和Linux。"""
    try:
        if os.name == 'nt':
            # Windows
            result = subprocess.run(['tree', '/f'], cwd=root_dir, capture_output=True, text=True, shell=True)
            return result.stdout
        else:
            # Linux/macOS
            result = subprocess.run(['tree', '-af'], cwd=root_dir, capture_output=True, text=True)
            return result.stdout
    except Exception as e:
        return f"[项目结构获取失败: {e}]"

def gather_py_files(root_dir):
    py_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py") and filename != SELF_FILE:
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, root_dir)
                py_files.append((rel_path, full_path))
    return sorted(py_files)

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    tree_str = get_project_tree(root_dir)
    py_files = gather_py_files(root_dir)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        out_f.write("# 项目结构 (tree /f 或 tree -af)\n")
        out_f.write("# ===============================\n")
        out_f.write(tree_str)
        out_f.write("\n# ===============================\n\n")
        for rel_path, full_path in py_files:
            out_f.write(f"# ========== {rel_path} ==========" + "\n")
            out_f.write(f"# 相对路径: {rel_path}\n")
            out_f.write(f"# 在项目中的相对位置: ./{rel_path}\n\n")
            with open(full_path, "r", encoding="utf-8") as in_f:
                out_f.write(in_f.read())
                out_f.write("\n\n")
    print(f"已整合 {len(py_files)} 个py文件到 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
