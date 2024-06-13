# GPS Synchronization Tool

This tool is designed to synchronize frames from videos with GPS information provided by FICOSA. This synchronization allows for accurate spatial-temporal alignment of the video frames with corresponding geographic locations. The output generated by this tool can then be utilized by COLMAP and NerfStudio.

## Prerequisites

#### FFmpeg: https://ffmpeg.org/download.html

#### Colmap: https://colmap.github.io/

#### Python Libraries
* Pillow
* Pandas
* Numpy
* Scipy

## Usage

1. Run to preprocess the images
   
 ```bash
 python preprocess.py --main-folder <path_to_main_folder> --output-folder <path_to_output_folder> --input-type {video, images} --input-folder <path_to_input_images_folder>
 ```
2. Run to create the csv files and the COLMAP folder

```bash
python gps_sync.py --main-folder <path_to_main_folder> --csv-folder <path_to_camera_folder> --total-frames <number_of_frames> --threshold-min <minimum_threshold> --threshold-max <maximum_threshold> --colmap-path <path_to_colmap_output_file>
```

## Help
#### preprocess
* main-folder: Path to the main folder ---> example: \FICOSA\VideoServer_DYMOS_VX\RealWorld\20240307_085658 
* output-folder: Path to the output folder to save the processed images
* input-type: {images, video}
   * video: Extract frames from the FICOSA's video
   * images: Assume that images are extracted and properly named
* input-folder: Folder that contains the input images (requires: input-type = images) 


#### gps_sync
* main-folder: Path to the main folder ---> example: \FICOSA\VideoServer_DYMOS_VX\RealWorld\20240307_085658 
* csv-folder: Path to the folder to write the camera CSV files
* total-frames: Number of frames to extract
* threshold-min: Minimum threshold for closest times (optional: if not set it takes all frames)
* threshold-max: Maximum threshold for closest times (optional: if not set it takes all frames)
* colmap-path: Path to the output COLMAP text file 



## Output
* __4 csv files__ (0_cam, 1_cam, 2_cam, 3_cam) each corresponding to 1 of the 4 cameras (front, right, left, back)
  
  ![Example of csv file](csv_example.png)
* __colmap folder__ that contains images.txt, cameras.txt, points3D.txt. These are needed in order to load the sparse model in COLMAP. 


## License
This project is licensed under the [MIT License](LICENSE).
