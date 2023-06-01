from moto3.s3_manager import S3Manager
import os
import zipfile
import io

def unzip_from_memory(zip_data, extract_dir):
    zip_file = zipfile.ZipFile(io.BytesIO(zip_data))

    for file in zip_file.namelist():
        zip_file.extract(file, extract_dir)

root = 'renders/'
if not os.path.exists(root):
    os.makedirs(root)
s3m = S3Manager("objaverse-db")
renders = s3m.list_all_files(prefix="renders/", max_files=1000)
for i in renders:
    print('unzipping:' + i)
    zip_file = s3m.read_file(i, decode =None)
    folder_name = os.path.splitext(os.path.basename(i))[0]
    # Create the folder
    extract_directory = os.path.join(os.getcwd(), root + folder_name)
    if not os.path.exists(extract_directory):
        os.makedirs(extract_directory)
    unzip_from_memory(zip_file, extract_directory)
