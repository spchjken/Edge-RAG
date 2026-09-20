import shutil
import os
import sys

def main():
    c_usage = shutil.disk_usage("/mnt/c")
    total_gb = c_usage.total / (1024**3)
    used_gb = c_usage.used / (1024**3)
    free_gb = c_usage.free / (1024**3)

    vhd_path = "/mnt/c/Users/dongh/AppData/Local/wsl/{3d11a2ab-604f-4fec-8401-c2b2583cf2a6}/ext4.vhdx"
    if os.path.exists(vhd_path):
        vhd_size_gb = os.path.getsize(vhd_path) / (1024**3)
    else:
        vhd_size_gb = -1

    print(f"HOST_C_TOTAL_GB: {total_gb:.2f}")
    print(f"HOST_C_USED_GB: {used_gb:.2f}")
    print(f"HOST_C_FREE_GB: {free_gb:.2f}")
    print(f"VHDX_SIZE_GB: {vhd_size_gb:.2f}")

    # Check if shrink_log.txt exists anywhere
    candidates = [
        "/mnt/c/Users/dongh/Desktop/shrink_log.txt",
        "/mnt/c/Users/dongh/shrink_log.txt",
        "/mnt/c/Windows/System32/shrink_log.txt",
        "/mnt/c/shrink_log.txt",
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"\nFound log at: {c}")
            with open(c, "r", encoding="utf-8", errors="ignore") as f:
                print(f.read())
            break
    else:
        print("\nNo shrink_log.txt found in common locations.")

if __name__ == "__main__":
    main()
