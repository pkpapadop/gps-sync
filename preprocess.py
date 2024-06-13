import os
import csv
import subprocess
import shutil
from PIL import Image
import argparse

def crop_and_save_images(source_folder, target_folder, crop_box):
    # Ensure the target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # List all files in the source folder
    files = os.listdir(source_folder)

    for file_name in files:
        # Check if the file is an image (you can customize the file extension check)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Load the image
            image_path = os.path.join(source_folder, file_name)
            img = Image.open(image_path)

            # Crop the image
            cropped_img = img.crop(crop_box)

            # Save the cropped image to the target folder
            target_path = os.path.join(target_folder, file_name)
            cropped_img.save(target_path)

def copy_and_rename_files(source_folder, destination_folder):
    # List all files in the source folder
    files = os.listdir(source_folder)

    # Iterate through each file and copy/rename it
    for old_name in files:
        # Construct the new name by adding a prefix
        #new_name = old_name[:10] + old_name[13:-4] + '.0.png'
        new_name = old_name[:-4] + '.0.png'


        # Build the full file paths
        old_path = os.path.join(source_folder, old_name)
        new_path = os.path.join(destination_folder, new_name)

        # Copy the file to the destination folder
        shutil.copy2(old_path, new_path)
        os.remove(old_path)

# Function to rename files in the folder based on the timestamps in the CSV file
def rename_files(csv_file, folder_path):
    # Read CSV file
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            timestamp = row['Timestamp']
            image_name = row['ImageUrl']
            old_path = os.path.join(folder_path, image_name)
            old_path = old_path + '.png'
            new_name = os.path.join(folder_path, timestamp + '.png')
            os.rename(old_path, new_name)


def extract_frames(input_video, output):
    # FFmpeg command to extract frames
    ffmpeg_command = [
        'ffmpeg',
        '-i', input_video,
        '-q:v', '0',
        f'{output}/frame%01d.png'
    ]

    # Run FFmpeg command
    subprocess.run(ffmpeg_command)



def preprocess(main_folder, output_folder, input_type='video', input_folder=None):
    crop_box = (0, 0, 968, 600)
    subfolder_name = r'HDFExtractData'
    # Construct the path to the HDFExtractData folder
    hdf_extract_data_folder = os.path.join(main_folder, subfolder_name)


    if os.path.exists(hdf_extract_data_folder):
        # Loop through the subfolders (0_video, 1_video, 2_video, 3_video)
        if input_type == 'video':
            #print('videdododododododod')
            for i in range(4):
                subfolder_name = f"{i}_video"
                subfolder_path = os.path.join(hdf_extract_data_folder, subfolder_name)
                video_path = os.path.join(subfolder_path, 'video.mp4')
                folder_create = os.path.join(output_folder, subfolder_name)
                folder_processed = os.path.join(output_folder, subfolder_name + '_processed')
                if not os.path.exists(folder_create):
                    os.makedirs(folder_create)
                if not os.path.exists(folder_processed):
                    os.makedirs(folder_processed)
                extract_frames(video_path, folder_create)
                csv_path = os.path.join(subfolder_path, 'video.csv')
                rename_files(csv_path, folder_create)
                crop_and_save_images(folder_create, folder_processed, crop_box)
                copy_and_rename_files(folder_processed, folder_processed)
        
        if input_type == 'images':
            #images need to be in a specific way
            #input_folder
                #0_video
                #1_video
                #2_video
                #3_video

            for i in range(4):
                subfolder_name = f"{i}_video"
                subfolder_path = os.path.join(hdf_extract_data_folder, subfolder_name)
                folder_create = os.path.join(input_folder, subfolder_name)
                folder_processed = os.path.join(output_folder, subfolder_name + '_processed')
                if not os.path.exists(folder_processed):
                    os.makedirs(folder_processed)
                csv_path = os.path.join(subfolder_path, 'video.csv')
                #rename_files(csv_path, folder_create)
                crop_and_save_images(folder_create, folder_processed, crop_box)
                copy_and_rename_files(folder_processed, folder_processed)


def main():
    parser = argparse.ArgumentParser(description='Extract (optional) and preprocess images')

    # Add command line arguments
    parser.add_argument('--main-folder', type=str, help='Path to the source folder containing images')
    parser.add_argument('--output-folder', type=str, help='Path to the target folder to save cropped images')
    parser.add_argument('--input-type', type=str, default='video', help='Type of input (video,images). If video extract images')
    parser.add_argument('--input-folder', type=str, default=None, help='Only use this argument if you selected images. It is the folder that contains them')


    # Parse the command line arguments
    args = parser.parse_args()

    # Call the preprocess function with command line arguments
    preprocess(args.main_folder, args.output_folder, args.input_type, args.input_folder)


if __name__ == "__main__":
    main()


