# Stolen Car Detection System

This repository contains a GUI-based application for detecting stolen cars by processing vehicle detection data from both video footage and a CSV file of known stolen license plates. The system uses YOLO object detection models and EasyOCR to extract and match license plate numbers from vehicles captured in a video.

## Features

* GUI interface for easy interaction.
* Upload and analyze videos containing vehicle footage.
* Upload a CSV file containing a list of known stolen license plates.
* Real-time detection of vehicles and license plates using YOLOv8.
* OCR-based recognition of license plate numbers using EasyOCR.
* Matches detected plates with stolen ones and saves results.
* Stores data in a MySQL database and generates annotated output video.

## Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

**Requirements include**:

* OpenCV
* Tkinter
* easyocr
* ultralytics
* numpy
* pandas
* mysql-connector-python
* scipy

You also need:

* Pre-trained YOLO models:

  * `car_detector.pt`
  * `license_plate_detector.pt`
* A running MySQL instance with appropriate tables:

  * `stolenlicenseplates(license_plate_numbers)`
  * `videolicenseplates(license_plate_numbers)`

## Folder Structure

```
.
├── data/
│   ├── car_detector.pt
│   └── license_plate_detector.pt
├── output/
│   ├── detected_cars_from_video.csv
│   ├── final_data.csv
│   ├── out.mp4
│   └── stolen_cars.csv
├── main.py
├── util1.py
├── util2.py
├── README.md
```

## How to Use

1. **Run the App:**

```bash
python main.py
```

2. **Insert CSV File:**

   * Click "Insert CSV" and select a `.csv` file containing stolen license plates.

3. **Insert Video:**

   * Click "Insert Video" and select a `.mp4` file to analyze.

4. **Execute Detection:**

   * Click "Execute" to run the full pipeline.
   * The system will:

     * Detect vehicles and license plates.
     * Recognize license plate numbers.
     * Match against stolen plates.
     * Update the database.
     * Generate annotated video and CSVs in the `output/` folder.

## CSV Formats

**Input CSV (`Insert CSV`)**

```
stolen_car_license_plates
ABC1234
XYZ5678
```

**Output CSVs**

* `detected_cars_from_video.csv`: Raw detections
* `final_data.csv`: Interpolated and structured detections
* `stolen_cars.csv`: Matched stolen plates

## Notes

* Ensure MySQL credentials are set correctly (`user=root`, `passwd=1234`, `database=police`).
* OCR accuracy may vary depending on video quality.
* Detected cars will be highlighted with license plate annotations in the output video.

## Credits

* YOLOv8 via [Ultralytics](https://github.com/ultralytics/ultralytics)
* OCR by [EasyOCR](https://github.com/JaidedAI/EasyOCR)
* SORT tracking algorithm
* GUI built with Tkinter

## License

MIT License
