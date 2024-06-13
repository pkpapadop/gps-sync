import math
import os
import re
import pandas as pd
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def vector_length(v):
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def quaternion_length(q):
    return math.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])

def rotation_matrix_to_quaternion(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S=4*qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return np.array([qw, qx, qy, qz])


#Extract the extrinsic infromation of front,right,left,back cameras
def extract_values(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    patterns = {
        'rot_x': r'rot_x = ([\d.-]+)',
        'rot_y': r'rot_y = ([\d.-]+)',
        'rot_z': r'rot_z = ([\d.-]+)',
        'fSocketX': r'fSocketX = ([\d.-]+)',
        'fSocketY': r'fSocketY = ([\d.-]+)',
        'fSocketZ': r'fSocketZ = ([\d.-]+)'
    }
    extracted_values = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            extracted_values[key] = float(match.group(1))

    return extracted_values

def find_closest_timestamp_hdf(timestamp, timestamps_hdf):
    return min(timestamps_hdf, key=lambda x: abs(x - timestamp))

def main(main_folder, folder_cam, total_frames, threshold_min, threshold_max,colmap_path):

    create_folder(folder_cam)
    create_folder(colmap_path)
    images_path = os.path.join(colmap_path, 'images.txt')
    cameras_path = os.path.join(colmap_path, 'cameras.txt')
    points3D_path = os.path.join(colmap_path, 'points3D.txt')

    subfolder_name = r'HDFExtractData'
    odometry = None

    for file_name in os.listdir(main_folder):
        if file_name.endswith('.csv'):
            odometry = os.path.join(main_folder, file_name)
            break

    csv_files = []
    config_files = []
    # Construct the path to the HDFExtractData folder
    hdf_extract_data_folder = os.path.join(main_folder, subfolder_name)

    if os.path.exists(hdf_extract_data_folder):
        # Loop through the subfolders (0_video, 1_video, 2_video, 3_video)
        for i in range(4):
            subfolder_name = f"{i}_video"
            subfolder_path = os.path.join(hdf_extract_data_folder, subfolder_name)
            csv_file_path = os.path.join(subfolder_path, 'video.csv')
            if os.path.exists(csv_file_path):
                csv_files.append(csv_file_path)
            config_file_path = os.path.join(subfolder_path, 'camera.ini')
            if os.path.exists(config_file_path):
                config_files.append(config_file_path)

    for csv_name in range(4):
        first_df = pd.read_csv(csv_files[csv_name])
        second_df = pd.read_csv(odometry)
        def find_closest_timestamp_hdf_row(row):
            timestamp = (row['Timestamp'])
            timestamps_hdf = second_df['TimestampHDF']
            closest_timestamp_hdf = find_closest_timestamp_hdf(timestamp, timestamps_hdf)
            second_row = second_df[second_df['TimestampHDF'] == closest_timestamp_hdf].iloc[0]
            return pd.Series({
                'Timestamp': int(row['Timestamp']),
                'ClosestTimestampHDF': closest_timestamp_hdf,
                'X': second_row['x'], 
                'Y': second_row['y'],  
                'Z': second_row['z'], 
                'Heading': second_row['heading'],
                'Pitch': second_row['pitch'],  
                'Roll': second_row['roll'] 
            })

        result_df = first_df.apply(find_closest_timestamp_hdf_row, axis=1)
        result_df = result_df.sort_values(by='ClosestTimestampHDF')
        result_df['Timestamp'] = result_df['Timestamp'].astype('int64')
        result_df['ClosestTimestampHDF'] = result_df['ClosestTimestampHDF'].astype('int64')
        absolute_difference = abs(result_df['Timestamp'] - result_df['ClosestTimestampHDF'])
        #Filter rows within the specified range (2200 to 2900)
        mask = (absolute_difference >=threshold_min) & (absolute_difference <= threshold_max)
        # Apply the mask to the DataFrame
        result_df = result_df[mask]
        output_path = rf'{csv_name}_cam.csv'
        output_path = os.path.join(folder_cam, output_path)
        result_df.to_csv(output_path, index=False)



    for global_index in range(4):
        path2 = rf'{global_index}_cam.csv'
        input_csv_path = os.path.join(folder_cam, path2)
        input_df = pd.read_csv(input_csv_path)

        #Extract extrinsic values of the car's cameras
        extrinsics_path = config_files[global_index]
        extracted_values = extract_values(extrinsics_path)

        #From euler coordinates calculate the the quaternion both rotated and unrotated
        def calculate_quaternion(yaw,roll,pitch):
            heading = yaw
            pitch1 = pitch
            roll1 = roll
            rot_x = extracted_values['rot_x']
            rot_y = extracted_values['rot_y']
            rot_z = extracted_values['rot_z']
            yaw =  (heading -rot_z) * (math.pi / 180) # X-axis (rotates the camera left right)
            pitch = (roll1 + 180 - rot_x) * (math.pi / 180) # Y-axis (around axis)
            roll = (-90 + pitch1 - rot_y) * (math.pi / 180)  # Z-axis (up-down)
            yaw_unrot = (heading) * (math.pi / 180)
            pitch_unrot = (roll1) * (math.pi / 180)
            roll_unrot = (pitch1) * (math.pi / 180) 
            qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
            qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
            qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
            qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
            qx_unrot = np.sin(roll_unrot/2) * np.cos(pitch_unrot/2) * np.cos(yaw_unrot/2) - np.cos(roll_unrot/2) * np.sin(pitch_unrot/2) * np.sin(yaw_unrot/2)
            qy_unrot = np.cos(roll_unrot/2) * np.sin(pitch_unrot/2) * np.cos(yaw_unrot/2) + np.sin(roll_unrot/2) * np.cos(pitch_unrot/2) * np.sin(yaw_unrot/2)
            qz_unrot = np.cos(roll_unrot/2) * np.cos(pitch_unrot/2) * np.sin(yaw_unrot/2) - np.sin(roll_unrot/2) * np.sin(pitch_unrot/2) * np.cos(yaw_unrot/2)
            qw_unrot = np.cos(roll_unrot/2) * np.cos(pitch_unrot/2) * np.cos(yaw_unrot/2) + np.sin(roll_unrot/2) * np.sin(pitch_unrot/2) * np.sin(yaw_unrot/2)

            return [qw, qx, qy, qz, qw_unrot, qx_unrot, qy_unrot, qz_unrot]



        # Apply the function to calculate quaternions
        input_df['quaternion'] = input_df.apply(lambda row: calculate_quaternion(row['Heading'], row['Roll'], row['Pitch']), axis=1) 
        input_df = input_df.head(total_frames)
        # Create a new DataFrame with the desired format
        output_df = pd.DataFrame({
            'id': input_df.index , 
            'qw': input_df['quaternion'].apply(lambda x: x[0]),
            'qx': input_df['quaternion'].apply(lambda x: x[1]),
            'qy': input_df['quaternion'].apply(lambda x: x[2]),
            'qz': input_df['quaternion'].apply(lambda x: x[3]),
            'qw_unrot': input_df['quaternion'].apply(lambda x: x[4]),
            'qx_unrot': input_df['quaternion'].apply(lambda x: x[5]),
            'qy_unrot': input_df['quaternion'].apply(lambda x: x[6]),
            'qz_unrot': input_df['quaternion'].apply(lambda x: x[7]),
            'x': - input_df['X']  ,
            'y': input_df['Y'],
            'z': input_df['Z'],
        
            'camera_model': 1, 
            'frame_path':input_df['Timestamp']
            
        })

        outputs_df = pd.DataFrame()

        fSocketY = extracted_values['fSocketY']
        fSocketX = extracted_values['fSocketX']
        fSocketZ = extracted_values['fSocketZ']
        for index, row in output_df.iterrows():

            original_quaternion = np.array([row['qw'], row['qx'], row['qy'], row['qz']])
            q = np.array([row['qx_unrot'], row['qy_unrot'], row['qz_unrot'], row['qw_unrot']])
            direction_vector = np.array([1, 0, 0])
            distance_to_move_left = fSocketY
            rotated_direction_vector = R.from_quat(q).apply(direction_vector)
            scaled_direction_vector_left = distance_to_move_left * rotated_direction_vector

            direction_vector = np.array([0, 1, 0])
            distance_to_move_up = fSocketX
            rotated_direction_vector = R.from_quat(q).apply(direction_vector)
            scaled_direction_vector_up = distance_to_move_up * rotated_direction_vector

            direction_vector = np.array([0, 0, 1])
            distance_to_move_front = -fSocketZ
            rotated_direction_vector = R.from_quat(q).apply(direction_vector)
            scaled_direction_vector_front = distance_to_move_front * rotated_direction_vector

            translation_vector = np.array([row['x'], row['y'], row['z']])
            translation_vector = translation_vector + scaled_direction_vector_left + scaled_direction_vector_front + scaled_direction_vector_up
        
            rotation_matrix = quaternion_to_rotation_matrix(original_quaternion)
            transposed_rotation_matrix = np.transpose(rotation_matrix)
            result_vector = -np.dot(transposed_rotation_matrix, translation_vector)

            # Update the output DataFrame with new quaternion and translation values
            outputs_df = pd.concat([
                outputs_df,
                pd.DataFrame({
                    'id': int(row['id'])+ (total_frames * global_index),
                    'qw': [rotation_matrix_to_quaternion(transposed_rotation_matrix)[0]],
                    'qx': [rotation_matrix_to_quaternion(transposed_rotation_matrix)[1]],
                    'qy': [rotation_matrix_to_quaternion(transposed_rotation_matrix)[2]],
                    'qz': [rotation_matrix_to_quaternion(transposed_rotation_matrix)[3]],
                    'tx': [result_vector[0]],
                    'ty': [result_vector[1]],
                    'tz': [result_vector[2]],
                    'camera_model': int(row['camera_model']),
                    'frame_path':str(row['frame_path']) +'.png'
                })
            ], ignore_index=True)


        if global_index == 0 :
            with open(images_path, 'w') as file:
                for _, row in outputs_df.iterrows():
                    file.write(' '.join(map(str, row)) + '\n\n')
        else:
            with open(images_path, 'a') as file:
                for _, row in outputs_df.iterrows():
                    file.write(' '.join(map(str, row)) + '\n\n')

    with open(cameras_path, 'w') as file:
                file.write('1 SIMPLE_RADIAL 968 600 369.99475464818056 484 300 -0.0017993058548071376')
                
    with open(points3D_path, 'w') as file:
                file.write('')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--main-folder', type=str, help='Path to the main folder containing other required files')
    parser.add_argument('--csv-folder', type=str, help='Path to the folder to write the camera CSV files')
    parser.add_argument('--total-frames', type=int, help='Number of frames to extract')
    parser.add_argument('--threshold-min',default=0, type=int, help='Minimum threshold for closest times')
    parser.add_argument('--threshold-max',default=10000, type=int, help='Maximum threshold for closest times')
    parser.add_argument('--colmap-path', type=str, help='Path to the output Colmap text file')

    args = parser.parse_args()

    main(args.main_folder, args.csv_folder, args.total_frames, args.threshold_min, args.threshold_max, args.colmap_path)
