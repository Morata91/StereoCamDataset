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


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))  # 現在のファイルからの相対パスに修正
DATA_DIR = os.path.join(PROJECT_DIR, 'data')



def main():
    



    # dlibの検出器を初期化
    try:
        detector = dlib.get_frontal_face_detector()  # 顔検出器
        predictor = dlib.shape_predictor("face_detector/shape_predictor_68_face_landmarks.dat")  # ランドマーク検出器
    except Exception as e:
        print(f"Error initializing face detectors: {e}")
        
    results = []   # ラベルデータ格納配列
        
    for person in range(0,3):
        person_data_dir = os.path.join(DATA_DIR, f'person{person:02d}')
        
        calib_file = os.path.join(person_data_dir, 'calibration_params.json')
        
        # キャリブレーションパラメータを読み込む
        if os.path.exists(calib_file):
            params = load_calibration_params(calib_file)
            if params:
                print(f"\nCalibration parameters loaded from: {calib_file}")
                print(f"Calibration date: {params['calibration_date']}")
        else:
            print("calibration_params.json is not found.")
            exit()
            
        # 射影行列を事前に計算
        try:
            P_l = np.dot(params['mtx_l'], np.hstack((np.eye(3), np.zeros((3,1)))))
            P_r = np.dot(params['mtx_r'], np.hstack((params['R'], params['T'])))
        except Exception as e:
            print(f"Error calculating projection matrices: {e}")
            

        # 画像ファイルのリストを取得
        left_images = sorted(glob.glob(os.path.join(person_data_dir, 'left', '*.JPG')))
        right_images = sorted(glob.glob(os.path.join(person_data_dir, 'right', '*.JPG')))

        print(f"Left images found: {len(left_images)}")
        print(f"Right images found: {len(right_images)}")

        if len(left_images) != len(right_images):
            print("Error: Number of left and right images don't match")
            exit()
        if not left_images:
            print("No images found")
            exit()
        print(f"Found {len(left_images)} image pairs")
        
        # 注視点座標の取得
        gaze_points_csv_path = os.path.join(person_data_dir, 'gaze_points.csv')
        try:
            df = pd.read_csv(gaze_points_csv_path)
            required_columns = ['x', 'y']
            
            # 必要なカラムが存在するか確認
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV file must contain columns: {required_columns}")
            
            # インデックスでソートして座標をリストに変換
            gaze_points = list(zip(df['x'], df['y']))
            
            if len(left_images) != len(gaze_points):
                print("Error: Number of images and gaze_points don't match")
                exit()
            
            print(f"Loaded {len(gaze_points)} monitor points from {gaze_points_csv_path}")
            
        except Exception as e:
            print(f"Error loading monitor points: {e}")

        

        for idx, (left_path, right_path, gaze_point) in enumerate(zip(left_images, right_images, gaze_points)):
            print(f"\nProcessing pair {idx+1}/{len(left_images)}")
            print(f"Left: {os.path.basename(left_path)}")
            print(f"Right: {os.path.basename(right_path)}")
            
            try:
                # 画像を読み込み
                img_l = cv2.imread(left_path)
                img_r = cv2.imread(right_path)
                
                if img_l is None or img_r is None:
                    print(f"Failed to load images:\nLeft: {left_path}\nRight: {right_path}")
                    continue
                    
                # グレースケール変換
                gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
                
                
                # 顔検出
                faces_l = detector(gray_l, 1)  # 第2引数は検出のアップサンプリング回数
                faces_r = detector(gray_r, 1)
                
                if len(faces_l) == 0 or len(faces_r) == 0:
                    print("No faces detected in one or both images")
                    continue
                
                face_l = max(faces_l, key=lambda rect: rect.width() * rect.height())
                face_r = max(faces_r, key=lambda rect: rect.width() * rect.height())
                
                # ランドマーク検出
                landmarks_l = predictor(gray_l, face_l)
                landmarks_r = predictor(gray_r, face_r)
                
                # 左目、右目の中点を計算
                left_eye_l = np.mean([[landmarks_l.part(i).x, landmarks_l.part(i).y] 
                                    for i in range(36, 42)], axis=0)
                right_eye_l = np.mean([[landmarks_l.part(i).x, landmarks_l.part(i).y] 
                                    for i in range(42, 48)], axis=0)
                
                left_eye_r = np.mean([[landmarks_r.part(i).x, landmarks_r.part(i).y] 
                                    for i in range(36, 42)], axis=0)
                right_eye_r = np.mean([[landmarks_r.part(i).x, landmarks_r.part(i).y] 
                                    for i in range(42, 48)], axis=0)
                
                # 両目の中点を計算
                eye_center_l = np.mean([left_eye_l, right_eye_l], axis=0)
                eye_center_r = np.mean([left_eye_r, right_eye_r], axis=0)
                
                # 三角測量
                point_4d = cv2.triangulatePoints(P_l, P_r,
                                                eye_center_l.astype(np.float32),
                                                eye_center_r.astype(np.float32))
                eye_center_3d = (point_4d[:3] / point_4d[3]).ravel()
                
                
                # モニター上のピクセル座標を物理的な座標に変換（左上原点）
                x_mm = (gaze_point[0] / params['monitor_resolution_x']) * params['monitor_width_mm']
                y_mm = (gaze_point[1] / params['monitor_resolution_y']) * params['monitor_height_mm']
                
                # 注視点のカメラ座標系3D位置を計算
                gaze_point_3d = np.array(params['monitor_top_left']) + np.array([-x_mm, y_mm, 0])
                
                # 視線ベクトルを計算 (モニターポイント - 目の中心)
                gaze_vector = gaze_point_3d - eye_center_3d
                
                # ベクトルを正規化
                gaze_vector_normalized = gaze_vector / np.linalg.norm(gaze_vector)
                
                
                # 結果を記録
                result = {
                    'left_image': os.path.basename(left_path),
                    'right_image': os.path.basename(right_path),
                    'm_gaze_point_x': float(gaze_point[0]),
                    'm_gaze_point_y': float(gaze_point[1]),
                    'gaze_point_x': float(gaze_point_3d[0]),
                    'gaze_point_y': float(gaze_point_3d[1]),
                    'gaze_point_z': float(gaze_point_3d[2]),
                    'left_eye_center_x': float(eye_center_l[0]),
                    'left_eye_center_y': float(eye_center_l[1]),
                    'right_eye_center_x': float(eye_center_r[0]),
                    'right_eye_center_y': float(eye_center_r[1]),
                    'eye_center_3d_x': float(eye_center_3d[0]),
                    'eye_center_3d_y': float(eye_center_3d[1]),
                    'eye_center_3d_z': float(eye_center_3d[2]),
                    'gaze_vector_x': float(gaze_vector_normalized[0]),
                    'gaze_vector_y': float(gaze_vector_normalized[1]),
                    'gaze_vector_z': float(gaze_vector_normalized[2]),
                }
                
                results.append(result)
                
                print('success')
                
                
            except Exception as e:
                print(f"Error processing pair: {e}")
                results.append({
                    'frame': idx,
                    'timestamp': datetime.now().isoformat(),
                    'left_image': os.path.basename(left_path),
                    'right_image': os.path.basename(right_path),
                    'processing_status': 'error',
                    'error_message': str(e)
                })
        
        
        # 処理したデータ数の確認
        print(f"Processed {len(results)} valid gaze vectors")
        
        # DataFrameを作成してCSVに保存
        df = pd.DataFrame(results)
        output_csv_path = os.path.join(DATA_DIR, 'label.csv')
        df.to_csv(output_csv_path, index=False)
        print(f"Gaze data saved to {output_csv_path}")
    




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
            'monitor_top_left': np.array(params['monitor_top_left']),
            'calibration_date': params['calibration_date']
        }
    except Exception as e:
        print(f"Error loading calibration parameters: {e}")
        return None
    
    
    
if __name__ == '__main__':
    main()
    
    