from tkinter import filedialog, messagebox
from ultralytics import YOLO
from sort.sort import *
from pathlib import Path
from util2 import get_car, read_license_plate, write_csv
from scipy.interpolate import interp1d
import cv2
import tkinter as tk
import csv
import mysql.connector
import mysql
import numpy as np
import ast
import pandas as pd


# Video Playing Label
video_label = None

# Buttons
video_button = None
CSV_button = None
execution_button = None

# Indicator Labels
CSV_indicator_label = None
video_indicator_label = None
execution_indicator_label = None

# Windows
main_window = None
video_window = None

# Frames
button_frame = None
detection_frame = None

# Found Booleans
video_found = None
CSV_found = None
executing = None

# Video
video = None

# Database Connection
conn = None

# Cursor
cursor = None

# Models
license_plate_detector = None
car_detector = None

# Paths
CSV_path = None
video_path = None
video_CSV_path = r'output\detected_cars_from_video.csv'

# CSV File Strings
input_CSV_string = r'data\test csv file 2.csv'
video_CSV_string = r'C:\Users\laptop zone\Desktop\Program Output\detected_cars_from_video.csv'

# Database Password and Instance
password = '1234'
instance = 'police'

# Main Function


def main(): #Riskless Function

    global main_window

    main_window = tk.Tk()

    initialiseConnectionAndCursor()
    setMainWindow()
    addAllFrames()

    main_window.mainloop()

# Button Related Functions


def addCSVButton(x_value, y_value, width, height): 

    global CSV_button

    CSV_button = tk.Button(button_frame, text='Insert CSV', font=('Arial', 10, 'bold'), command= insertCSVFile, background='#C4CA8D')
    CSV_button.config(padx=width, pady=height)

    CSV_button.place(x=x_value, y=y_value)


def addVideoButton(x_value, y_value, width, height): 

    global video_button

    video_button = tk.Button(button_frame, text='Insert Video', font=('Arial', 10, 'bold'), command=searchVideo, background='#C4CA8D')
    video_button.config(padx=width, pady=height)

    video_button.place(x=x_value, y=y_value)


def addExecuteButton(x_value, y_value, width, height): 

    global execution_button

    execution_button = tk.Button(button_frame, text='Execute', font=('Arial', 10, 'bold'), command=executeProcess, background='#C4CA8D')
    execution_button.config(padx=width, pady=height)

    execution_button.place(x=x_value, y=y_value)


def addAllButtons(): 

    addCSVButton(x_value=15, y_value=75, width=40, height=30)
    addVideoButton(x_value=210, y_value=75, width=34, height=30)
    addExecuteButton(x_value=405, y_value=75, width=47, height=29)

# Miscellaneous Functions


def setMainWindow(): 

    global main_window

    screen_width = main_window.winfo_screenwidth()
    screen_height = main_window.winfo_screenheight()

    x = (screen_width - 600) // 2
    y = (screen_height - 295) // 2

    main_window.geometry(f'600x295+{x}+{y}')
    main_window.resizable(False, False)
    main_window.title('Stolen Car Detector')


def addIndicatorLabels(root_frame): 

    global CSV_indicator_label, video_indicator_label, execution_indicator_label

    CSV_indicator_label = tk.Label(root_frame, text='CSV Added', foreground='red', background='#3d3a3a', font=('Arial', 12), width=10)
    video_indicator_label = tk.Label(root_frame, text='Video Added', foreground='red', background='#3d3a3a', font=('Arial', 12), width=11)
    execution_indicator_label = tk.Label(root_frame, text='Executing', foreground='red', background='#3d3a3a', font=('Arial', 12), width=8)

    CSV_indicator_label.place(x=0, y=0)
    video_indicator_label.place(x=475, y=0)
    execution_indicator_label.place(x=245, y=0)


def updateIndicatorLabels(): 

    global video_found, CSV_indicator_label, video_indicator_label , execution_indicator_label , video_found , \
        CSV_found, executing

    if video_found:
        video_indicator_label.config(foreground= 'green')
    if CSV_found:
        CSV_indicator_label.config(foreground= 'green')
    if executing:
        execution_indicator_label.config(foreground= 'green')


def resetIndicatorLabels(label_number): 

    global video_found, CSV_indicator_label, video_indicator_label, execution_indicator_label, video_found, \
        CSV_found, executing

    if label_number == 1:

        video_found = False
        video_indicator_label.config(foreground='red')

    elif label_number == 2:

        CSV_found = False
        CSV_indicator_label.config(foreground='red')

    elif label_number == 3:

        resetIndicatorLabels(1)
        resetIndicatorLabels(2)

        executing = False
        execution_indicator_label.config(foreground='red')

    else:

        tk.messagebox.showerror('Error!', 'Wrong Label Number Was Input')

def executeProcess(): 

    global video_found, CSV_found, video_path, main_window, executing

    if video_found and CSV_found:

        executing = True
        updateIndicatorLabels()

        tk.messagebox.showinfo('Info!', 'The Process Is Starting, This Will Take A While, DO NOT CLOSE')

        parseFiles(video_path)
        matchLicensePlates()

        tk.messagebox.showinfo('Info! ', 'CSVs Have Been Added To The Database')

    else:

        tk.messagebox.showerror('Error!', 'Both Video and CSV Files Must Be Input!')


def searchVideo(): 

    global video_found, video_path

    file_path = tk.filedialog.askopenfilename()

    try:

        if file_path.lower().endswith('.mp4'):

            video_found = True
            video_path = file_path

            updateIndicatorLabels()

        elif not file_path:

            pass

        else:

            resetIndicatorLabels(1)

            tk.messagebox.showerror('Error!', 'Invalid File Selected')


    except IOError as err:

        tk.messagebox.showerror('Error!', f'File IO Error! {err}')

    except Exception as err:

        tk.messagebox.showerror('Error!', f'An Exception Occurred! {err}')


def parseFiles(video_path):

    results = {}

    mot_tracker = Sort()

    # load models
    coco_model = YOLO(r"data/car_detector.pt")
    license_plate_detector = YOLO(r"data/license_plate_detector.pt")

    # load video
    cap = cv2.VideoCapture(video_path)

    vehicles = [2, 5, 7]

    # read frames
    frame_nmr = -1
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}
            # detect vehicles
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])

            # track vehicles
            detections_array = np.asarray(detections_)
            if detections_array.size == 0:
                track_ids = np.array([])  # Handle empty detection case
            else:
                track_ids = mot_tracker.update(detections_array)

            # detect license plates
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:

                    # crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                    # process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                      'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}

    # write results
    write_csv(results, r'output\detected_cars_from_video.csv')
    tk.messagebox.showinfo('Info!', 'The CSVs Have Been Written')

    addMissingData(r'output\detected_cars_from_video.csv')
    tk.messagebox.showinfo('Info!', 'The Missing Data Has Been Added, Now Writing The Video')

    writeVideo(video_path, r'output\final_data.csv')
    tk.messagebox.showinfo('Info!', 'The Video Has Been Written')

    resetIndicatorLabels(3)

# Initialising Functions

def initialiseConnection(): 

    global conn

    try:

        conn = conn = mysql.connector.connect(
                host='localhost',
                user='root',
                passwd=password,
                database=instance
            )

    except Exception as err:

        tk.messagebox.showerror('Error!', f'Exception Occurred While Trying To Establish A Database Connection: {err}')


def initialiseCursor(): 

    global cursor, conn

    try:

        cursor = conn.cursor()

    except mysql.connector.Error as err:

        tk.messagebox.showerror('Error!', f'Database Error: {err}')

    except Exception as err:

        tk.messagebox.showerror('Error!', f'Exception Occurred While Trying To Create A Cursor: {err}')


def initialiseConnectionAndCursor():

    initialiseConnection()
    initialiseCursor()


def initialiseLicensePlateDetector():

    global license_plate_detector

    license_plate_detector = YOLO(r'data/license_plate_detector.pt')

# CSV Related Functions


def readCSVFile(file_path): 

    if isinstance(file_path, Path):

        file_path = str(file_path)

    data_tuples = []

    try:

        with open(file_path, 'r') as csv_file:

            csv_reader = csv.reader(csv_file)
            next(csv_reader)

            # Iterate over each row in the CSV file
            for row in csv_reader:

                if row:  # Check if the row is not empty
                    # Create a tuple with the single entry and append to the list
                    data_tuples.append((row[0],))

    except FileNotFoundError:

        tk.messagebox.showerror('Error!', f"Error: The file at {file_path} was not found.")

    except Exception as err:

        tk.messagebox.showerror('Error!', f'Exception Occurred While Trying To Read The CSV File: {err}')

    return data_tuples


def readLicensePlateCSV(file_path): 

    data = []

    try:

        with open(file_path, 'r') as csv_file:

            csv_reader = csv.reader(csv_file)
            next(csv_reader)

            for row in csv_reader:

                if len(row) > 6:

                    column1 = row[1]
                    column5 = row[5]
                    column6 = row[6]

                    data.append((column1, column5, column6))

    except FileNotFoundError:

        tk.messagebox.showerror('Error!', f'File At {file_path} Was Not Found!')

    except Exception as err:

        tk.messagebox.showerror('Error!', f'An Exception Occurred While Trying To Read The CSV File: {err}')

    plate_dict = {}
    license_plates = {}
    confidence_score = None

    for row in data:

        plate_id = row[0]
        plate_value = row[1]
        score = row[2]

        if plate_id in plate_dict:

            if score > confidence_score:

                license_plates[plate_id] = plate_value
                confidence_score = score
        else:

            plate_dict[plate_id] = score
            confidence_score = score
            license_plates[plate_id] = plate_value

    license_plates_values = license_plates.values()
    license_plates_values = list(license_plates_values)

    output_list = [(item,) for item in license_plates_values]

    return output_list

def writeCSVFile(tuples_list): 

    filename = r'output\stolen_cars.csv'

    try:

        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Write each tuple to the CSV file
            writer.writerow(['stolen_car_license_plates'])
            writer.writerows(tuples_list)

    except FileNotFoundError:

        tk.messagebox.showerror('Error!', f'File At {filename} Was Not Found')

    except Exception as err:

        tk.messagebox.showerror('Error!', f'An Exception Occurred While Trying To Write A CSV File')


def insertCSVFile(): 

    global CSV_path, CSV_found

    file_path = tk.filedialog.askopenfilename()

    try:

        if file_path.lower().endswith('.csv'):

            CSV_found = True
            CSV_path = file_path

            updateIndicatorLabels()

        elif not file_path:

            pass

        else:

            resetIndicatorLabels(2)

            tk.messagebox.showerror('Error!', 'Invalid File Selected')


    except IOError as err:

        tk.messagebox.showerror('Error!', f'File IO Error! {err}')

    except Exception as err:

        tk.messagebox.showerror('Error!', f'An Exception Occurred! {err}')

# Frame Related Functions

def addButtonFrame(x_value, y_value, width, height): 

    global button_frame

    button_frame = tk.Frame(main_window, bg='#3681bf', bd=10, relief=tk.RAISED, background='#3B7B7A')
    button_frame.config(width=width, height=height)

    addAllButtons()

    button_frame.place(x=x_value, y=y_value)


def addDetectionFrame(x_value, y_value, width, height): 

    global detection_frame

    detection_frame = tk.Frame(main_window, bg='#7321ad', bd=10, relief=tk.RIDGE, background='#0B536B')
    detection_frame.config(width=width, height=height)

    addIndicatorLabels(detection_frame)

    detection_frame.place(x=x_value, y=y_value)


def addAllFrames(): 

    addButtonFrame(0, 0, 600, 250)
    addDetectionFrame(0, 250, 600, 45)

# SQL Related Functions

def matchLicensePlates(): 

    global CSV_path, video_CSV_path

    video_license_plates = readLicensePlateCSV(video_CSV_path)
    input_license_plates = readCSVFile(CSV_path)

    addToVideoTable(video_license_plates)
    addToStolenTable(input_license_plates)

    joined_tables = innerJoinOutputList()
    writeCSVFile(joined_tables)


def innerJoinOutputList(): 

    global cursor
    results = []

    try:

        query = '''
            SELECT * 
            FROM stolenlicenseplates
            INNER JOIN videolicenseplates 
            ON videolicenseplates.license_plate_numbers = stolenlicenseplates.license_plate_numbers
        '''

        cursor.execute(query)

        results = cursor.fetchall()

    except mysql.connector.Error as err:

        tk.messagebox.showerror('Error!', f"Database Error: {err}")

    except Exception as err:

        tk.messagebox.showerror('Error!', f'An Exception Occured While Trying To Join The Tables: {err}')

    output_list = [(t[0],) for t in results]

    return output_list

def addToStolenTable(CSV_list): 

    global cursor
    query = 'INSERT INTO stolenlicenseplates(license_plate_numbers) VALUES (%s)'

    try:

        cursor.executemany(query, CSV_list)
        conn.commit()

    except mysql.connector.Error as err:

        conn.rollback()
        tk.messagebox.showerror('Error!', f"Database Error: {err}")

    except Exception as err:

        tk.messagebox.showerror('Error!', f'An Exception Occured While Trying To Add Values To The stolenlicenseplates'
                                          f'Table: {err}')


def addToVideoTable(CSV_list): 

    global cursor

    print(f'REMOVE LATER, {CSV_list}')

    try:

        query = 'INSERT INTO videolicenseplates(license_plate_numbers) VALUES (%s)'
        cursor.executemany(query, CSV_list)
        conn.commit()

    except mysql.connector.Error as err:

        conn.rollback()
        tk.messagebox.showerror('Error!', f"Database Error: {err}")

    except Exception as err:

        tk.messagebox.showerror('Error!', f'An Exception Occured While Trying To Add Values To The videolicenseplates '
                                          f'Table: {err}')

def interpolate_bounding_boxes(data):
    # Extract necessary data columns from input data
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    for car_id in unique_car_ids:

        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]
        print(frame_numbers_, car_id)

        # Filter data for a specific car ID
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = car_frame_numbers[0]
        last_frame_number = car_frame_numbers[-1]

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i-1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['frame_nmr'] = str(frame_number)
            row['car_id'] = str(car_id)
            row['car_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
            row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

            if str(frame_number) not in frame_numbers_:
                # Imputed row, set the following fields to '0'
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                # Original row, retrieve values from the input data if available
                original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][0]
                row['license_plate_bbox_score'] = original_row['license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else '0'
                row['license_number'] = original_row['license_number'] if 'license_number' in original_row else '0'
                row['license_number_score'] = original_row['license_number_score'] if 'license_number_score' in original_row else '0'

            interpolated_data.append(row)

    return interpolated_data

def addMissingData(CSV_file_path):

    # Load the CSV file
    with open(CSV_file_path, 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    # Interpolate missing data
    interpolated_data = interpolate_bounding_boxes(data)

    # Write updated data to a new CSV file
    header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
    with open(r'output\final_data.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(interpolated_data)

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

def writeVideo(video_path, CSV_path):

    results = pd.read_csv(CSV_path)

    # load video
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(r'output\out.mp4', fourcc, fps, (width, height))

    license_plate = {}
    for car_id in np.unique(results['car_id']):
        max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
        license_plate[car_id] = {'license_crop': None,
                                 'license_plate_number': results[(results['car_id'] == car_id) &
                                                                 (results['license_number_score'] == max_)]['license_number'].iloc[0]}
        cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                                 (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
        ret, frame = cap.read()

        x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                                  (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

        license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

        license_plate[car_id]['license_crop'] = license_crop

    frame_nmr = -1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # read frames
    ret = True
    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        if ret:
            df_ = results[results['frame_nmr'] == frame_nmr]
            for row_indx in range(len(df_)):
                # draw car
                car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                            line_length_x=200, line_length_y=200)

                # draw license plate
                x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

                # crop license plate
                license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']

                H, W, _ = license_crop.shape

                try:
                    frame[int(car_y1) - H - 100:int(car_y1) - 100,
                          int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                    frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                          int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                    (text_width, text_height), _ = cv2.getTextSize(
                        license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4.3,
                        17)

                    cv2.putText(frame,
                                license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                                (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                4.3,
                                (0, 0, 0),
                                17)
                except:
                    pass

            out.write(frame)
            frame = cv2.resize(frame, (1280, 720))

            # cv2.imshow('frame', frame)
            # cv2.waitKey(0)

    out.release()
    cap.release()
