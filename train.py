from sys import path
from tkinter import*
from tkinter import ttk
from PIL import Image,ImageTk
import os
import mysql.connector
import cv2
import numpy as np
from tkinter import messagebox
from trainer import VanillaDL, TRAINERMAIN, CREATE_MODEL
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import classification_report






class Train:

    def __init__(self,root):
        self.root=root
        self.root.geometry("1366x768+0+0")
        self.root.title("RIMT-IITD Attendance System | Train Pannel")

        # This part is image labels setting start 
        # first header image  
        img=Image.open(r"./Images_GUI/banner.jpg")
        img=img.resize((1366,130),Image.LANCZOS)
        self.photoimg=ImageTk.PhotoImage(img)

        # set image as lable
        f_lb1 = Label(self.root,image=self.photoimg)
        f_lb1.place(x=0,y=0,width=1366,height=130)

        # backgorund image 
        bg1=Image.open(r"./Images_GUI/t_bg1.jpg")
        bg1=bg1.resize((1366,768),Image.LANCZOS)
        self.photobg1=ImageTk.PhotoImage(bg1)

        # set image as lable
        bg_img = Label(self.root,image=self.photobg1)
        bg_img.place(x=0,y=130,width=1366,height=768)


        #title section
        title_lb1 = Label(bg_img,text="Training Panel | RIMT-IITD Attendance System",font=("verdana",30,"bold"),bg="white",fg="navyblue")
        title_lb1.place(x=0,y=0,width=1366,height=45)

        # Create buttons below the section 
        # ------------------------------------------------------------------------------------------------------------------- 
        # Training button 1
        std_img_btn=Image.open(r"./Images_GUI/t_btn1.png")
        std_img_btn=std_img_btn.resize((180,180),Image.LANCZOS)
        self.std_img1=ImageTk.PhotoImage(std_img_btn)

        std_b1 = Button(bg_img,command=self.call_trainer,image=self.std_img1,cursor="hand2")
        std_b1.place(x=600,y=170,width=180,height=180)

        std_b1_1 = Button(bg_img,command=self.call_trainer,text="Train Dataset",cursor="hand2",font=("tahoma",15,"bold"),bg="navyblue",fg="white")
        std_b1_1.place(x=600,y=350,width=180,height=45)

    
    
    # ==================Create Function of Traing===================
    def call_trainer(self):
        try:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            DATA_DIR = "./data_images/train"
            trainerX= TRAINERMAIN()
            DATA_FOR_INCEPTION = VanillaDL(16,DATA_DIR,299, split = 0.8) 
            NUM_CLASSES = len(DATA_FOR_INCEPTION.class_names)
            LEARNING_RATE = 0.01
            EPOCHS = 5
            criterion = nn.CrossEntropyLoss()
            inceptionv3 = CREATE_MODEL('inceptionv3',NUM_CLASSES)
            inceptionv3.model = inceptionv3.model.to(device)
            optimizer= optim.SGD(inceptionv3.model.fc.parameters(), lr=LEARNING_RATE, momentum=0.9)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            TRAIN_ACCURACY, TRAIN_LOSS, VAL_ACCURACY, VAL_LOSS, inceptionv3.model = trainerX.train_model(DATA_FOR_INCEPTION, inceptionv3.model, criterion, optimizer, exp_lr_scheduler, num_epochs=EPOCHS, is_inception = True)
            #Confusion Matrix on Validation Data
            CM = inceptionv3.get_confusion_matrix(DATA_FOR_INCEPTION.train_dataloader, NUM_CLASSES)

            #Evaluation Params on Validation Data
            y_ground, y_predicted = inceptionv3.evaluation_params(DATA_FOR_INCEPTION.val_dataloader)
            print(classification_report(y_ground,y_predicted, target_names=DATA_FOR_INCEPTION.class_names))
            #Saving Model
            torch.save(inceptionv3.model, "./inceptionv3.pth")
            messagebox.showinfo("Result","Training Dataset Completed!", parent=self.root )
        except Exception as e:
            messagebox.showinfo("Result","Training NOT Completed! Reason: "+str(e),parent=self.root)



if __name__ == "__main__":
    root=Tk()
    obj=Train(root)
    root.mainloop()