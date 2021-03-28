import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import base64
import os

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='', page_icon="./f.png")
st.title('Convolutional Neural Network - MNIST Classifier')
st.subheader('By [Francisco Tarantuviez](https://www.linkedin.com/in/francisco-tarantuviez-54a2881ab/) -- [Other Projects](https://franciscot.dev/portfolio)')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.write('---')

class CNN(nn.Module): 
  # constructor
  def __init__(self, out_1=16, out_2=32):
    super(CNN,self).__init__()
    self.cnn1=nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
    self.maxpool1=nn.MaxPool2d(kernel_size=2)
    
    self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=2)
    self.maxpool2=nn.MaxPool2d(kernel_size=2)
    self.fc1 = nn.Linear(out_2 * 4 * 4, 10)
  
  # prediction
  def forward(self, x):
    x=self.cnn1(x)
    x=torch.relu(x)
    x=self.maxpool1(x)
    x=self.cnn2(x)
    x=torch.relu(x)
    x=self.maxpool2(x)
    
    x = x.view(x.size(0), -1)
    x = self.fc1(x)
    return x

def plot_parameters(W, number_rows=1, name="", i=0):
    W = W.data[:, i, :, :]
    n_filters = W.shape[0]
    w_min = W.min().item()
    w_max = W.max().item()
    fig, axes = plt.subplots(number_rows, n_filters // number_rows)
    fig.subplots_adjust(hspace=0.4)

    for i, ax in enumerate(axes.flat):
      if i < n_filters:
        # Set the label for the sub-plot.
        ax.set_xlabel("kernel:{0}".format(i + 1))

        # Plot the image.
        ax.imshow(W[i, :], vmin=w_min, vmax=w_max, cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle(name, fontsize=10)    
    st.pyplot(fig)

def get_binary_file_downloader_html(bin_file, file_label='File'):
  with open(bin_file, 'rb') as f:
    data = f.read()
  bin_str = base64.b64encode(data).decode()
  href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download</a>'
  return href

cnn_trained = CNN()
cnn_trained.load_state_dict(torch.load("cnn.pth"))

st.write("This application is made to predict hand-written numbers with the image format of [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.")
st.write("Through a Neural Network (a Convolutional Neural Network) this app predict which number you uploaded. Here you just get fun, if you wanna the code, at the end is the Github repository.")
st.warning("""
**[WARNING]**: The images should follow the next prerequisites
- The number has to be an integer
- The number can be in the range [0, 10)
- The format should be similar to the following example:
""")
col1, col2 = st.beta_columns(2)

example_img = Image.open("example.png")
col1.image(example_img, caption="Example of image")

col2.write("### Choose an image")
col2.write("You can search and upload your own images, or you can download one from here and upload to the model!")
img_to_download = col2.number_input("Download image:", 0, 9)
col2.markdown(get_binary_file_downloader_html("images_to_download/{}.png".format(img_to_download), f'MNIST_{img_to_download}'), unsafe_allow_html=True)



image_user = st.file_uploader("Upload file", type=["png", "jpg", "jpeg"])
image_uploaded = Image.open(image_user) if image_user else None
IMAGE_SIZE = 16
if image_uploaded:
  col3, col4 = st.beta_columns(2)
  col3.write("### **Image uploaded**")
  col3.image(image_uploaded, caption="Example of image")
  transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
  img_tensor = transform(image_uploaded.convert('L'))
  col4.write("### **Prediction**")
  col4.dataframe(pd.DataFrame(pd.Series(torch.max(cnn_trained(img_tensor[None, ...]).data,1).indices.item()), columns=["Value predicted"]))


# This app repository

st.write("""
## App repository

[Github](https://github.com/ftarantuviez/Convolutional-Neural-Network-MNIST-Classification)
""")
# / This app repository