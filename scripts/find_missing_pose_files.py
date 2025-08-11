import os

def find_missing_pose_files(root_dir):
    for subject in sorted(os.listdir(root_dir)):
        subject_path = os.path.join(root_dir, subject)
        if not os.path.isdir(subject_path):
            continue
        frame_files = sorted([f for f in os.listdir(subject_path) if f.endswith('.png')])
        for frame_file in frame_files:
            pose_file = frame_file.replace('.png', '_pose.txt')
            pose_path = os.path.join(subject_path, pose_file)
            if not os.path.isfile(pose_path):
                print(f"Missing pose file for image: {os.path.join(subject_path, frame_file)}")

if __name__ == "__main__":
    find_missing_pose_files("dataset/BIWI/faces_0")
