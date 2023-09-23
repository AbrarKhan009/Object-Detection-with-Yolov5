import cv2
import csv

# Stop Sign Cascade Classifier xml data
#stop_sign = cv2.CascadeClassifier('stop_data.xml')
stop_sign = cv2.CascadeClassifier(r"C:\Users\abrar\OneDrive\Desktop\Research Project\OpenCv task 0\task0\__pycache__\task0_stopSign_Detection\stop_data.xml")


output_file = r"C:/Users/abrar/OneDrive/Desktop/Research Project/OpenCv task 0/task0/output5(SF_1.4).txt"
#C:\Users\abrar\OneDrive\Desktop\Research Project\OpenCv task 0\task0\task0_txt_csv_abrar.py
#output_file = r"F:\Abrar_Task\Stop-Sign-Detection\output.csv"

def main():
    param_list = []

    cap = cv2.VideoCapture("StopSign.mp4")
    while cap.isOpened():
        _, img = cap.read()

        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            stop_sign_scaled = stop_sign.detectMultiScale(gray, 1.4, 5)

            #open below link for more detailed instruction about detectMultiScale() method parameters
            #https://stackoverflow.com/questions/36218385/parameters-of-detectmultiscale-in-opencv-using-python

            for (x, y, w, h) in stop_sign_scaled:

                # Syntax: cv2.rectangle(image, start_point, end_point, color, thickness)
                stop_sign_rectangle = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                stop_sign_text = cv2.putText(img=stop_sign_rectangle,
                                             text="Stop Sign",
                                             org=(x, y + h + 30),
                                             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                             fontScale=1, color=(0, 0, 255),
                                             thickness=1, lineType=cv2.LINE_4)


                param_list.append([x, y,w, h])
                print(param_list)
                print(x.dtype.name)
                print(x.size)
                print(x.itemsize)
                print(type(x))

            cv2.imshow("img", img)
            key = cv2.waitKey(10)

        else:
            break


    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)

        for values in param_list:
            writer.writerow(values)

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()