# calibration_params.json生成


import numpy as np
import dlib
import cv2
import glob
import json
import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation

# モニター設定
MONITOR_WIDTH_MM = 1329
MONITOR_HEIGHT_MM = 748
MONITOR_RESOLUTION_X = 1920
MONITOR_RESOLUTION_Y = 1080
MONITOR_TOP_LEFT = np.array([164 + 1329, 1810 - 1740, 0])   

# パスの設定
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))  # 現在のファイルからの相対パスに修正
stereo_calib_dir = os.path.join(PROJECT_DIR, 'stereocalib/0')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
person = 0
calib_file = os.path.join(DATA_DIR, f'person{person:02d}', 'calibration_params.json')

# キャリブレーションパラメータ
checkerboard = (9, 6)
square_size = 28.75  # mm



def main():
    if os.path.exists(calib_file):
        params = load_calibration_params(calib_file)
        if params:
            print(f"\nAlready calibrated.")
            print(f"Calibration date: {params['calibration_date']}")
    else:
        print("Performing new calibration...")
        left_images = sorted(glob.glob(os.path.join(stereo_calib_dir, 'left', '*.JPG')))
        right_images = sorted(glob.glob(os.path.join(stereo_calib_dir, 'right', '*.JPG')))
        
        if not left_images or not right_images:
            print("No stereo calibration images found")
        
        try:
            params = perform_calibration(left_images, right_images, checkerboard, square_size)
            save_calibration_params(params, calib_file)
        except Exception as e:
            print(f"Calibration failed: {e}")

    print_calibration_results(params)



def load_calibration_params(calib_file):
    """キャリブレーションパラメータを読み込む"""
    try:
        with open(calib_file, 'r') as f:
            params = json.load(f)
        
        return {
            'mtx_l': np.array(params['left_camera_matrix']),
            'dist_l': np.array(params['left_distortion']),
            'mtx_r': np.array(params['right_camera_matrix']),
            'dist_r': np.array(params['right_distortion']),
            'R': np.array(params['rotation_matrix']),
            'T': np.array(params['translation_vector']),
            'E': np.array(params['essential_matrix']),
            'F': np.array(params['fundamental_matrix']),
            'checkerboard': tuple(params['checkerboard_size']),
            'monitor_width_mm': int(params['monitor_width_mm']),
            'monitor_height_mm': int(params['monitor_height_mm']),
            'monitor_resolution_x': int(params['monitor_resolution_x']),
            'monitor_resolution_y': int(params['monitor_resolution_y']),
            'monitor_top_right': np.array(params['monitor_top_right']),
            'calibration_date': params['calibration_date']
        }
    except Exception as e:
        print(f"Error loading calibration parameters: {e}")
        return None

def perform_calibration(left_images, right_images, checkerboard, square_size):
    """ステレオカメラキャリブレーションを実行"""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # チェッカーボードの3D点を準備
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints_l = []
    imgpoints_r = []
    img_size = None
    
    for left_path, right_path in zip(left_images, right_images):
        img_l = cv2.imread(left_path)
        img_r = cv2.imread(right_path)
        
        if img_l is None or img_r is None:
            print(f"Failed to load images: {left_path} or {right_path}")
            continue
            
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        
        if img_size is None:
            img_size = gray_l.shape[::-1]
        
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, checkerboard, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, checkerboard, None)
        
        if ret_l and ret_r:
            objpoints.append(objp)
            corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), criteria)
            corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), criteria)
            imgpoints_l.append(corners2_l)
            imgpoints_r.append(corners2_r)
    
    if not objpoints:
        raise ValueError("No valid calibration images found")
    
    # 単眼キャリブレーション
    ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, img_size, None, None)
    ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, img_size, None, None)
    
    # ステレオキャリブレーション
    ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        mtx_l, dist_l, mtx_r, dist_r,
        img_size, None, None,
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    
    return {
        'mtx_l': mtx_l, 'dist_l': dist_l,
        'mtx_r': mtx_r, 'dist_r': dist_r,
        'R': R, 'T': T, 'E': E, 'F': F,
        'checkerboard': checkerboard,
        'calibration_date': datetime.now().isoformat()
    }

def save_calibration_params(params, calib_file):
    """キャリブレーションパラメータを保存"""
    save_params = {
        'left_camera_matrix': params['mtx_l'].tolist(),
        'left_distortion': params['dist_l'].tolist(),
        'right_camera_matrix': params['mtx_r'].tolist(),
        'right_distortion': params['dist_r'].tolist(),
        'rotation_matrix': params['R'].tolist(),
        'translation_vector': params['T'].tolist(),
        'essential_matrix': params['E'].tolist(),
        'fundamental_matrix': params['F'].tolist(),
        'checkerboard_size': params['checkerboard'],
        'monitor_width_mm': MONITOR_WIDTH_MM,
        'monitor_height_mm': MONITOR_HEIGHT_MM,
        'monitor_resolution_x': MONITOR_RESOLUTION_X,
        'monitor_resolution_y': MONITOR_RESOLUTION_Y,
        'monitor_top_left': MONITOR_TOP_LEFT.tolist(),
        'calibration_date': params['calibration_date']
    }
    
    try:
        os.makedirs(os.path.dirname(calib_file), exist_ok=True)
        with open(calib_file, 'w') as f:
            json.dump(save_params, f, indent=4)
        print(f"Calibration parameters saved to: {calib_file}")
    except Exception as e:
        print(f"Error saving calibration parameters: {e}")

def print_calibration_results(params):
    """キャリブレーション結果を表示"""
    print("\n=== Calibration Parameters ===")
    
    # カメラ内部パラメータの表示
    for camera in ['left', 'right']:
        print(f"\n--- {camera.capitalize()} Camera Intrinsic Matrix ---")
        mtx_key = f'mtx_{camera[0]}'
        dist_key = f'dist_{camera[0]}'
        
        if params[mtx_key] is not None:
            print(f"Camera Matrix (K_{camera}):")
            print(np.array2string(params[mtx_key], precision=3, suppress_small=True))
            print(f"\nDistortion Coefficients ({camera}):")
            print(np.array2string(params[dist_key], precision=3, suppress_small=True))
        else:
            print(f"{camera.capitalize()} camera parameters not available")
    
    # ステレオパラメータの表示
    print("\n--- Stereo Parameters ---")
    if all(params[key] is not None for key in ['R', 'T', 'E', 'F']):
        print("Rotation Matrix (R):")
        print(np.array2string(params['R'], precision=3, suppress_small=True))
        
        print("\nTranslation Vector (T):")
        print(np.array2string(params['T'], precision=3, suppress_small=True))
        
        # 基線長の計算
        baseline = np.linalg.norm(params['T'])
        print(f"\nBaseline (mm): {baseline:.2f}")
        
        # 回転角度の計算
        r = Rotation.from_matrix(params['R'])
        euler_angles = r.as_euler('xyz', degrees=True)
        print("\nRotation Angles (degrees):")
        print(f"X: {euler_angles[0]:.2f}")
        print(f"Y: {euler_angles[1]:.2f}")
        print(f"Z: {euler_angles[2]:.2f}")
    else:
        print("Stereo parameters not available")
        
if __name__ == '__main__':
    main()
    
    
    