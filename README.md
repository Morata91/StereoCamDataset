# ステレオカメラを用いた視線推定データセット作成


## ディレクトリ構造
```
project/
├── face_detector/
│   └── shape_predictor_68_face_landmarks.dat   # dlib顔検出器
│── label.csv   # ラベルデータ
└── data/
    ├── person00/
    │   ├── left/
    │   ├── right/
    │   ├── gaze_points.csv
    │   └── calibration_params.json
    ├── person01/
    │   ├── left/
    │   ├── right/
    │   ├── gaze_points.csv
    │   └── calibration_params.json
    ├── ...
    └── stereocalib/    # ステレオキャリブレーション用画像ファイル


```

## 手順

1. パッケージのインストール
```bash
pip install -r requirements.txt
```

2. dlibの顔ランドマーク検出モデルをダウンロード:
```bash
wget -P face_detector http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 face_detector/shape_predictor_68_face_landmarks.dat.bz2
```

3. データセット解凍
```bash
unzip gaze_dataset_rawdata.zip
```

4. ラベリング
```bash
python label.py
```

## 出力されるラベルデータの形式

### label.csv
- left_image: 左カメラの画像ファイル名
- right_image: 右カメラの画像ファイル名
- m_gaze_point_x: 注視点のピクセルX座標
- m_gaze_point_x: 注視点のピクセルY座標
- gaze_point_x: 注視点のX座標
- gaze_point_y: 注視点のY座標
- gaze_point_z: 注視点のZ座標
- left_eye_center_x: 左目の中心X座標（左画像内）
- left_eye_center_y: 左目の中心Y座標（左画像内）
- right_eye_center_x: 左目の中心X座標（右画像内）
- right_eye_center_y: 左目の中心Y座標（右画像内）
- eye_center_3d_x: 両目の中点の3D位置X
- eye_center_3d_y: 両目の中点の3D位置Y
- eye_center_3d_z: 両目の中点の3D位置Z
- gaze_vector_x: 視線ベクトルのX成分
- gaze_vector_y: 視線ベクトルのY成分
- gaze_vector_z: 視線ベクトルのZ成分





## メモ

person00
MONITOR_TOP_LEFT = np.array([164 + 1329, 1810 - 1740, 0])

person01~
MONITOR_TOP_LEFT = np.array([250 + 1329, 1810 - 1740, 0])


person00　村田

person01　松浦

person02　沙

person03　酒井

