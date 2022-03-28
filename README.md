# Person-Detector-Web

## 1. Clone the repo

## 2. Open Anaconda Promt, navigate to the project folder. Open the project in VSCode from here using "code ." command.
## 3. For CPU or GPU, run the following commands on VSCode terminal.

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

## 4. Downloading Official Pre-trained Weights for YOLOv4 detection
 
YOLOv4 comes pre-trained and able to detect 80 classes. For easy demo purposes we will use the pre-trained weights.
Download pre-trained yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

Copy and paste yolov4.weights from your downloads folder into the 'data' folder of this repository.

## 5. YOLOv4 Using Tensorflow (tf, .pb model)
To implement YOLOv4 using TensorFlow, first we convert the .weights into the corresponding TensorFlow model files and then run the model.

```bash
## Convert darknet weights to tensorflow
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4
```

## 6. For windows, "Visual Studio" must be installed on the system along with "Desktop Development with C++".

## 7. Install face_recognition API.
```bash
## For Windows
pip install cmake dlib==19.18 face_recognition
## For Ubuntu
pip install cmake face_recognition
```

## 8. Install the Flask App Dependencies
```bash
pip install flask flask_bootstrap flask_wtf wtforms flask_sqlalchemy werkzeug flask_login email-validator
```
## 9. Install sqlite3 on windows 10.
https://www.youtube.com/watch?v=wXEZZ2JT3-k&ab_channel=ProgrammingKnowledge

## 10. Install sqlite3 on Ubuntu.
https://www.youtube.com/watch?v=C16QgidWZsU&ab_channel=ProgrammingKnowledge

```
sudo apt-get install sqlite3 libsqlite3-dev
sqlite3
.quit

## navigate to the project directory an run following commands
sqlite3 database.db
.databases
.exit
```

## 11. Create sqlite database for the project
```bash
## In the VSCode termial, run:
python
from app import db
db.create_all()
exit()
sqlite3 database.db
.tables
.exit
```
## 12. run "app.py".
