import streamlit as st
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os

st.title('Vehicle parking slot detection')
def load_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255. 

    return img_tensor
def save_uploadedfile(uploadedfile):
    tempDir=os.getcwd()
    with open(os.path.join(tempDir,uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())



    # load model
model = load_model("first_cnn.h5", compile=False)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['acc'])

uploaded_files = st.file_uploader("Choose a image", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    st.image(uploaded_file)
        
    save_uploadedfile(uploaded_file)
    img=load_image(uploaded_file.name)
    pred = model.predict(img)
    classes = ['full','free']
    
    pred = pred[:][0].tolist()
 
    max_prob=max(pred)
    x=pred.index(max_prob)
    cls = classes[x]
        
    st.write("The parking is", cls)