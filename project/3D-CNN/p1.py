import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
import time
from tensorflow.keras.models import Model, load_model
import subprocess


if __name__ == "__main__":
    
    targets_name = ['backhand',
                'backhand2hands',
                'backhand_slice',
                'backhand_volley',
                'forehand_flat',
                'forehand_openstands',
                'forehand_slice',
                'forehand_volley',
                'smash'
            ]


    #   Loading model
    model1_path = "model"
    model1 = load_model(model1_path)

    # Setting up window
    font = cv2.FONT_HERSHEY_SIMPLEX
    quietMode = False
    img_rows,img_cols=64, 64 
    cap = cv2.VideoCapture('p55_smash_s3.avi')
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original',720,720)

    # Initialising varibales
    framecount = 0
    fps = ""
    start = time.time()
    frames = []
    num=[5]
    max =1
    real_index = 5
    instruction = 'No Gesture'
    pre =0
    prev = None
    prev2 = None
    prev3 = None
    black = np.zeros((100, 400, 3), dtype = "uint8")
    num_classes = 8

    # Running videostream
    while(1):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 3)
        frame = cv2.resize(frame, (640,480))
        

        framecount = framecount + 1
        end  = time.time()
        timediff = (end - start)
        if( timediff >= 1):
            fps = 'FPS:%s' %(framecount)
            start = time.time()
            framecount = 0

        cv2.putText(frame,fps,(10,20), font, 0.7,(0,255,0),2,1)
        X_tr=[]
             
        image=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)
        input=np.array(frames)
        
        if input.shape[0]==35:
            frames = frames[2:]
            X_tr.append(input)
            X_train= np.array(X_tr)
            train_set = np.zeros((1, 35, img_cols,img_rows,3))
            train_set[0][:][:][:][:]=X_train[0,:,:,:,:]
            train_set = train_set.astype('float32')
            train_set -= 108.26149
            train_set /= 146.73851
            result_1 = model1.predict(train_set)
            print(result_1)
            num = np.argmax(result_1,axis =1)
            instruction = targets_name[num[0]]
            print(instruction)
                        
        cv2.putText(frame, instruction, (450, 50), font, 0.7, (0, 255, 0), 2, 1)
        cv2.putText(black, "Quiet Mode  "+instruction, (0,50), font, 0.8, (255, 255, 255), 2, 1)
        if not quietMode:
            cv2.resizeWindow('Original',720,720)
            cv2.imshow('Original',frame)
        if quietMode:
            cv2.resizeWindow('Original',100,400)
            cv2.imshow('Original',black)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('q'):
            quietMode = not quietMode
    cap.release()
    cv2.destroyAllWindows()