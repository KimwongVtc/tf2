import tensorflow as tf
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox


class SampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self._frame = None
        self.switch_frame(StartPage)
        self.geometry("1000x600")
        self.title("Stem Platform")
        self.configure(bg="blue")

    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()


class StartPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Frame.configure(self, bg='blue')
        tk.Label(self, text="Stem Platform", font=('Helvetica', 48, "bold")).pack(side="top", fill="x", pady=150)
        tk.Button(self, text="Start",
                  command=lambda: master.switch_frame(PageOne)).pack(side="top", pady=10)
        tk.Button(self, text="Go to page two",
                  command=lambda: master.switch_frame(PageTwo)).pack()


class PageOne(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Frame.configure(self, bg='blue')
        tk.Label(self, text="Level", font=('Helvetica', 48, "bold")).pack(padx=300, pady=150)
        tk.Button(self, text="Grade 3",
                  command=lambda: master.switch_frame(PageTwo)).pack(padx=150, pady=5)
        tk.Button(self, text="Grade 4",
                  command=lambda: master.switch_frame(PageThree)).pack(padx=150, pady=5)
        tk.Button(self, text="Grade 5",
                  command=lambda: master.switch_frame(StartPage)).pack(padx=150, pady=5)
        tk.Button(self, text="Grade 6",
                  command=lambda: master.switch_frame(StartPage)).pack(padx=150, pady=5)


class PageTwo(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Frame.configure(self,bg='red')
        tk.Label(self, text="Page two", font=('Helvetica', 18, "bold")).pack(side="top", fill="x", pady=5)
        tk.Button(self, text="Go back to start page",
                  command=chk).pack()


def button_event():
    MsgBox = tk.messagebox.askquestion('Success', 'Next Step')
    if MsgBox == 'yes':
#        cap.release()
        cv2.destroyAllWindows()
    tk.Button(cv2, text='yes', command=lambda: master.switch_frame(PageThree)).pack()


def chk():
    # Using laptop’s webcam as the source of video
    cap = cv2.VideoCapture(0)
    cap.set(3, 480)  # ID 3 = width
    cap.set(4, 320)  # ID 4 = height

    # Labels — The various outcome possibilities
    labels = ["t1", "iphone"]

    # Loading the model weigths we just downloaded
    model = tf.keras.models.load_model("test.h5", compile=False)
    while True:
        success, image = cap.read()
        if not success :
            break

        # Necessary to avoid conflict between left and right
        image = cv2.flip(image, 1)
        cv2.imshow("Check", image)

        # The model takes an image of dimensions (224,224) as input so let’s
        # reshape our image to the same.
        img = cv2.resize(image, (224, 224))

        # Convert the image to a numpy array
        img = np.array(img, dtype=np.float32)
        img = np.expand_dims(img, axis=0)

        # Normalizing input image
        img = img / 255

        # Predict the class
        prediction = model.predict(img)

        # Map the prediction to the labels
        # Rnp.argmax returns the indices of the maximum values along an axis.
        predicted_labels = labels[np.argmax(prediction[0], axis=-1)]
        #    print(predicted_labels)
        print(predicted_labels, np.argmax(prediction[0], axis=-1), prediction[0])
        t1, iphone = prediction[0]  # 取得預測結果
        if t1 > 0.9:
            messagebox.askquestion('Fail', 'Retry')
        if iphone > 0.9:
            time.sleep(10)
            button_event()
        cv2.imshow('oxxostudio', img)
        if cv2.waitKey(500) == ord('q'):
            break  # 按下 q 鍵停止


class PageThree(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Frame.configure(self,bg='red')
        tk.Label(self, text="Page three", font=('Helvetica', 18, "bold")).pack(side="top", fill="x", pady=5)
        tk.Button(self, text="Go back to start page",
                  command=chk).pack()


if __name__ == "__main__":
   app = SampleApp()
   app.mainloop()

# Function
# def start():

# image
# img = tk.PhotoImage(file="")

#btn = Button(text="Click me")
# btn.config(bg="skyblue")
# btn.config(width=10, height=5)
# btn.config(image=img)
# btn.config(command=start)




# model = tf.keras.models.load_model('20230306_2keras_model.h5', compile=False)
# data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


# def text(text):
#    global img
#    org = (0,50)
#    fontFace = cv2.FONT_HERSHEY_SIMPLEX
#    fontScale = 2.5
#    color = (255,255,255)
#    thickness = 5
#    lineType = cv2.LINE_AA
#    cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType) # 放入文字

# root = tk.Tk()
# root.geometry("350x400+200+300")
# root.title('cuteluluWindow')
# root.configure(bg="#7AFEC6")
# root.iconbitmap('heart_green.ico')
# root.geometry('300x300')

# cap = cv2.VideoCapture(1)
# if not cap.isOpened():
#    messagebox.showerror('My messagebox', 'Error')
#    print("Cannot open camera")
#    exit()
# while True:
#   ret, frame = cap.read()
#    if not ret:
#     messagebox.showerror('My messagebox', 'Error')
#        print("Cannot receive frame")
#     break
#    img = cv2.resize(frame , (398, 224))
#    img = img[0:224, 80:304]
#    image_array = np.asarray(img)
#    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
#    data[0] = normalized_image_array
#    prediction = model.predict(data)
#    angle,ledg,light,bg,button = prediction[0]
#    if angle >0.9:
#        messagebox.showerror('My messagebox', 'Error')
#        text('angle')
#    if ledg >0.9:
#        messagebox.showerror('My messagebox', 'Error')
#       text('ledg')
#    if light >0.9:
#        messagebox.showerror('My messagebox', 'Error')
#       text('light')
#    if button >0.9:
#        messagebox.showerror('My messagebox', 'Error')
#        text('button')
#    cv2.imshow('oxxostudio', img)
#    if cv2.waitKey(1) == ord('q'):
#       break    # 按下 q 鍵停止

#    root.mainloop()
#cap.release()
#cv2.destroyAllWindows()

