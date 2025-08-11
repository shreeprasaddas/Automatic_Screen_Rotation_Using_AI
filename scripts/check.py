import os

root_dir = r"D:\Chrome Download\auto_screen_rotation\dataset\BIWI\faces_0"

print(f"Checking dataset folder: {root_dir}")
if not os.path.exists(root_dir):
    print("ERROR: Root dataset folder does not exist!")
else:
    subjects = sorted(os.listdir(root_dir))
    print(f"Subjects found: {subjects}")

    for subject in subjects:
        subject_dir = os.path.join(root_dir, subject)
        if not os.path.isdir(subject_dir):
            print(f"Skipping {subject_dir} (not a directory)")
            continue

        files = sorted(os.listdir(subject_dir))
        print(f"Files in {subject}: {files[:10]}")  # show first 10 files
