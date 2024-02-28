import os
from PIL import Image

def replace_string_from_files(directory, target_string, new_str):
    # Loop through all files in the specified directory
    for filename in os.listdir(directory):
        # Check if the file is a text file
        if filename.endswith(".txt"):
            # Construct the full path to the file
            filepath = os.path.join(directory, filename)
            # Open the file for reading
            with open(filepath, 'r', encoding='utf-8') as file:
                contents = file.read()
            # Remove the target string from the file's contents
            updated_contents = contents.replace(target_string,new_str) 
            # Open the file for writing
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(updated_contents)
            print(f"Processed {filename}")

def crop_to_1024(directory):
    for f in os.listdir(directory):
        f = os.path.join(directory, f)
        cropped = Image.crop((0,0,1024,1024))
        cropped.save(f+'.jpg')

# Specify the directory where your text files are located
dir = '/home/minjune/kohya_ss/dataset/xvd_noreg_as_minjune/img/40_minjune man '
# Specify the string you want to remove
a = "xvd"
b = 'minjune'
replace_string_from_files(dir, a, b)
# Call the function with your specified directory and string
#remove_string_from_files(directory_path, target_string)
