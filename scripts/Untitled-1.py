# scripts/Untitled-1.py
import os
from pathlib import Path
from datetime import datetime

def scan_directory_by_ctime():
    # 您的目标路径
    target_path = Path(r"C:\Users\PC\OneDrive\桌面\Saarland Writing Sample\data_new")

    print(f"🔍 正在扫描 (按创建时间倒序): {target_path}")
    print("=" * 110)
    # 调整了列宽以容纳更多信息
    print(f"{'文件名 (File Name)':<50} | {'大小':<10} | {'创建时间 (Created)':<20} | {'修改时间 (Modified)'}")
    print("-" * 110)

    if not target_path.exists():
        print(f"❌ 错误: 找不到目录 {target_path}")
        return

    # 1. 获取所有文件
    files = [f for f in target_path.iterdir() if f.is_file()]
    
    # 2. 关键修改：按创建时间 (st_ctime) 排序，reverse=True 表示倒序 (最新的在前面)
    # 注意：在 Windows 上 st_ctime 是创建时间；在 Linux 上通常是元数据修改时间
    files.sort(key=lambda f: f.stat().st_ctime, reverse=True)
    
    csv_count = 0
    json_count = 0
    pdf_count = 0

    for file_path in files:
        stat = file_path.stat()
        size_bytes = stat.st_size
        size_str = f"{size_bytes / 1024:.1f} KB" if size_bytes > 1024 else f"{size_bytes} B"
        
        # 获取时间戳
        ctime = datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        # 打印 (高亮最近的文件)
        print(f"{file_path.name:<50} | {size_str:<10} | {ctime:<20} | {mtime}")

        if file_path.suffix.lower() == '.csv':
            csv_count += 1
        elif file_path.suffix.lower() == '.json':
            json_count += 1
        elif file_path.suffix.lower() == '.pdf':
            pdf_count += 1

    print("=" * 110)
    print(f"📊 统计摘要 (最新文件在顶部):")
    print(f"   - CSV 表格: {csv_count}")
    print(f"   - JSON 数据: {json_count}")
    print(f"   - PDF 图表: {pdf_count}")
    print("=" * 110)

if __name__ == "__main__":
    scan_directory_by_ctime()