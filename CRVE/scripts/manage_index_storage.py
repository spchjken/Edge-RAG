import os
import shutil
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_DIR = "/mnt/d/Edge-RAG-archive"

def report_and_move():
    print("=" * 60)
    print("Checking Index Storage Locations...")
    print("=" * 60)

    dense_src = os.path.join(BASE_DIR, "data", "cache", "dense_indices")
    splade_src = os.path.join(BASE_DIR, "data", "cache", "splade_indices")
    terrier_src = os.path.join(BASE_DIR, "data", "cache", "terrier_indices")

    dense_dst = os.path.join(ARCHIVE_DIR, "dense_indices")
    splade_dst = os.path.join(ARCHIVE_DIR, "splade_indices")
    terrier_archive = os.path.join(ARCHIVE_DIR, "terrier_indices")

    os.makedirs(dense_dst, exist_ok=True)
    os.makedirs(splade_dst, exist_ok=True)
    os.makedirs(terrier_src, exist_ok=True)

    # 1. Move dense indices to D:\
    if os.path.exists(dense_src):
        entries = sorted(os.listdir(dense_src))
        print(f"\n[Dense Indices] Found {len(entries)} entries in {dense_src}:")
        for e in entries:
            src_path = os.path.join(dense_src, e)
            dst_path = os.path.join(dense_dst, e)
            print(f"  Moving {e} -> {dense_dst}...", end="", flush=True)
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path, ignore_errors=True)
            shutil.move(src_path, dst_path)
            print(" done!", flush=True)

    # 2. Move splade indices to D:\
    if os.path.exists(splade_src):
        entries = sorted(os.listdir(splade_src))
        print(f"\n[SPLADE Indices] Found {len(entries)} entries in {splade_src}:")
        for e in entries:
            src_path = os.path.join(splade_src, e)
            dst_path = os.path.join(splade_dst, e)
            print(f"  Moving {e} -> {splade_dst}...", end="", flush=True)
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path, ignore_errors=True)
            shutil.move(src_path, dst_path)
            print(" done!", flush=True)

    # 3. Check and restore any remaining terrier indices from D:\
    if os.path.exists(terrier_archive):
        entries = sorted(os.listdir(terrier_archive))
        print(f"\n[Terrier Archive] Checking {len(entries)} entries in {terrier_archive}...")
        for e in entries:
            src_path = os.path.join(terrier_archive, e)
            dst_path = os.path.join(terrier_src, e)
            if not os.path.exists(dst_path):
                print(f"  Restoring {e} -> {terrier_src}...", end="", flush=True)
                shutil.move(src_path, dst_path)
                print(" done!", flush=True)
            else:
                print(f"  Already in cache: {e}")

    # Final summary
    print("\n" + "=" * 60)
    print("Storage Status Summary:")
    print("=" * 60)
    print(f"Dense in {dense_src}: {len(os.listdir(dense_src)) if os.path.exists(dense_src) else 0}")
    print(f"Dense in {dense_dst}: {len(os.listdir(dense_dst)) if os.path.exists(dense_dst) else 0}")
    print(f"SPLADE in {splade_src}: {len(os.listdir(splade_src)) if os.path.exists(splade_src) else 0}")
    print(f"SPLADE in {splade_dst}: {len(os.listdir(splade_dst)) if os.path.exists(splade_dst) else 0}")
    print(f"Terrier in {terrier_src}: {len(os.listdir(terrier_src)) if os.path.exists(terrier_src) else 0}")
    
    # Check D:\ and ext4 free space
    try:
        import shutil as sh
        c_use = sh.disk_usage("/mnt/c")
        d_use = sh.disk_usage("/mnt/d")
        ext4_use = sh.disk_usage(BASE_DIR)
        print(f"\nDisk Free Space:")
        print(f"  Windows C: {c_use.free / (1024**3):.1f} GB free")
        print(f"  Windows D: {d_use.free / (1024**3):.1f} GB free")
        print(f"  Linux ext4: {ext4_use.free / (1024**3):.1f} GB free")
    except Exception as e:
        print(f"Error checking free space: {e}")

if __name__ == "__main__":
    report_and_move()
