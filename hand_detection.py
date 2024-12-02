from datetime import datetime  # Import datetime class explicitly
import cv2
import argparse
import orien_lines
from imutils.video import VideoStream
from utils import detector_utils as detector_utils
import xlrd
from xlwt import Workbook
from xlutils.copy import copy
import numpy as np

lst1 = []
lst2 = []
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                default=1, help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())

detection_graph, sess = detector_utils.load_inference_graph()

def save_data(no_of_time_hand_detected, no_of_time_hand_crossed):
    try:
        # Get current date and time in format: YYYY-MM-DD HH:MM:SS
        current_date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        rb = xlrd.open_workbook('result.xls')
        sheet = rb.sheet_by_index(0)

        # Loop through the rows to find if the current date already exists
        date_found = False
        for row in range(1, sheet.nrows):
            existing_date = sheet.cell_value(row, 1)

            if existing_date == current_date_time.split()[0]:  # If the date is the same
                date_found = True
                break

        # Open the workbook for writing
        wb = copy(rb)
        w_sheet = wb.get_sheet(0)

        if date_found:
            # If the date exists, move to the next row and add the data
            new_row = sheet.nrows
            w_sheet.write(new_row, 0, new_row)  # Sl.No
            w_sheet.write(new_row, 1, current_date_time.split()[0])  # Date only
            w_sheet.write(new_row, 2, current_date_time.split()[1])  # Time only
            w_sheet.write(new_row, 3, no_of_time_hand_detected)  # No of times hand detected
            w_sheet.write(new_row, 4, no_of_time_hand_crossed)  # No of times hand crossed
            wb.save('result.xls')
        else:
            # If the date doesn't exist, add a new row with the current date and time
            new_row = sheet.nrows
            w_sheet.write(new_row, 0, new_row)  # Sl.No
            w_sheet.write(new_row, 1, current_date_time.split()[0])  # Date only
            w_sheet.write(new_row, 2, current_date_time.split()[1])  # Time only
            w_sheet.write(new_row, 3, no_of_time_hand_detected)  # No of times hand detected
            w_sheet.write(new_row, 4, no_of_time_hand_crossed)  # No of times hand crossed
            wb.save('result.xls')

    except FileNotFoundError:
        # If the file doesn't exist, create a new one with the required columns
        current_date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        wb = Workbook()
        sheet = wb.add_sheet('Sheet 1')

        sheet.write(0, 0, 'Sl.No')
        sheet.write(0, 1, 'Date')
        sheet.write(0, 2, 'Time')
        sheet.write(0, 3, 'Number of times hand detected')
        sheet.write(0, 4, 'Number of times hand crossed')

        m = 1
        sheet.write(1, 0, m)
        sheet.write(1, 1, current_date_time.split()[0])  # Date
        sheet.write(1, 2, current_date_time.split()[1])  # Time
        sheet.write(1, 3, no_of_time_hand_detected)
        sheet.write(1, 4, no_of_time_hand_crossed)

        wb.save('result.xls')

if __name__ == '__main__':
    score_thresh = 0.80
    vs = VideoStream(0).start()

    Orientation = 'bt'
    Line_Perc1 = float(15)
    Line_Perc2 = float(30)

    num_hands_detect = 2

    start_time = datetime.now()  # Use the corrected datetime class
    num_frames = 0

    im_height, im_width = (None, None)
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)

    def count_no_of_times(lst):
        x = y = cnt = 0
        for i in lst:
            x = y
            y = i
            if x == 0 and y == 1:
                cnt = cnt + 1
        return cnt

    try:
        while True:
            frame = vs.read()
            frame = np.array(frame)

            if im_height is None:
                im_height, im_width = frame.shape[:2]

            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")

            boxes, scores, classes = detector_utils.detect_objects(frame, detection_graph, sess)
            Line_Position2 = orien_lines.drawsafelines(frame, Orientation, Line_Perc1, Line_Perc2)

            a, b = detector_utils.draw_box_on_image(
                num_hands_detect, score_thresh, scores, boxes, classes, im_width, im_height, frame, Line_Position2, Orientation)
            lst1.append(a)
            lst2.append(b)

            no_of_time_hand_detected = no_of_time_hand_crossed = 0

            num_frames += 1
            elapsed_time = (datetime.now() - start_time).total_seconds()
            fps = num_frames / elapsed_time

            if args['display']:
                detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)
                cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    vs.stop()
                    break

        no_of_time_hand_detected = count_no_of_times(lst2)
        no_of_time_hand_crossed = count_no_of_times(lst1)

        save_data(no_of_time_hand_detected, no_of_time_hand_crossed)
        print("Average FPS: ", str("{0:.2f}".format(fps)))

    except KeyboardInterrupt:
        no_of_time_hand_detected = count_no_of_times(lst2)
        no_of_time_hand_crossed = count_no_of_times(lst1)
        save_data(no_of_time_hand_detected, no_of_time_hand_crossed)
        print("Average FPS: ", str("{0:.2f}".format(fps)))
