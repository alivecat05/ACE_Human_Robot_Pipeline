import os

data_root_path = "/media/ubuntu/My Passport/ACE-Datatool/"

for root, dirs, files in os.walk(data_root_path):
    for dir_name in dirs:
        if dir_name == "rgb":
            dir_path = os.path.join(root, dir_name)
            print(f"Removing directory: {dir_path}")
            os.system(f"rm -rf {dir_path}")