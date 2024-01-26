import streamlit as st # ok
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import io
import torch # ok
import matplotlib.pyplot as plt #ok
from torchvision import transforms as T
import torch.nn as nn
import cv2
import torch.optim as optim
import os
    
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import streamlit as st
    
from denoiser_MODEL import UNet, Down, Up, ConvBlock
from semantic_seg_MODEL import SemUNet
import torchvision.transforms.functional as TF
# import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from semantic_seg_MODEL import SemUNet
from torchvision import transforms
  


def main():
    # Боковая панель с навигацией
    menu = ["Детекция опухолей головного мозга", "Очистка документов", "Семантическая сегментация", "Семантическая сегментация 2"]
    choice = st.sidebar.radio("Навигация", menu)

    # Отображение контента в зависимости от выбранной страницы
    if choice == "Детекция опухолей головного мозга":
        page_tumor_detection()
    elif choice == "Очистка документов":
        page_document_cleanup()
    elif choice == "Семантическая сегментация":
        page_semantic_segmentation()
    elif choice == "Семантическая сегментация 2":
        page_semantic_segmentation2()

def page_tumor_detection():
    st.subheader("Детекция опухолей головного мозга")
    # Добавьте контент для главной страницы
    

    @st.cache_data

    def load_model():
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='github')
        return model

    model = load_model()

    # Streamlit page configuration
    st.title("YOLOv5 Object Detection")

    # File uploader 
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    def draw_boxes(image, results):
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        n = len(labels)
        x_shape, y_shape = image.shape[1], image.shape[0]
        for i in range(n):
            row = cord[i]
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            bgr = (0, 255, 0) # color of the box
            classes = model.names  # Get class names from model
            label_font = cv2.FONT_HERSHEY_SIMPLEX  # Font for the label
            cv2.rectangle(image, (x1, y1), (x2, y2), bgr, 2)  # Draw the rectangle
            cv2.putText(image, f'{classes[int(labels[i].item())]} {row[4]:.2f}', (x1, y1), label_font, 0.9, bgr, 2)  # Put the class name and confidence
        return image

    if uploaded_file is not None:
        # Convert the file to an image
        image = Image.open(BytesIO(uploaded_file.read())).convert('RGB')
        #вывод на печать изначальной фотографии
        #st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Transform the image for the model
        transform = T.Compose([T.ToTensor()])
        input_img = transform(image).unsqueeze(0)

        # Run the model
        results = model(image)

        # Draw boxes on the image
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_boxed = draw_boxes(image_np, results)

        # Convert back to RGB and display
        image_boxed = cv2.cvtColor(image_boxed, cv2.COLOR_BGR2RGB)
        #st.image(image, caption='Uploaded Image', use_column_width=True)
        st.image(image_boxed, caption='Processed Image', use_column_width=True)
 

def page_document_cleanup():
    
    # Добавьте контент для страницы 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet().to(device)

    @st.cache_data
    def load_model(model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNet()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model.to(device)

    def transform_image(image):
        transform = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.Grayscale(num_output_channels=1),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        return transform(image).unsqueeze(0).to(device)

    def show_output_image(output_tensor):
        output_image = output_tensor.squeeze().cpu().detach().numpy()
        output_image = (output_image * 255).astype(np.uint8)  # Convert to uint8
        output_image = Image.fromarray(output_image)  # Convert to PIL Image
        return output_image

    st.title('Image Denoiser')
    model_path = 'denoising_unet.pth'
    model = load_model(model_path)

    uploaded_file = st.file_uploader("Choose an image...", type="png")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')  # Convert to RGB
        st.write("Denoising in progress...")

        # Process the image and obtain the denoised version
        input_tensor = transform_image(image)
        with torch.no_grad():
            output_tensor = model(input_tensor)
        denoised_image = show_output_image(output_tensor)

        # Create two columns for displaying the images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)

        with col2:
            st.image(denoised_image, caption='Denoised Image', use_column_width=True)

                

def page_semantic_segmentation():
    st.subheader("Семантическая сегментация")
    # Добавьте контент для страницы 2
            
    class UNet(nn.Module):
        def __init__(self):
            super(UNet, self).__init__()

            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

            self.decoder = nn.Sequential(
                nn.Conv2d(64, 3, kernel_size=3, padding=1),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x



    # Загрузка предварительно обученной модели U-Net

    # Загрузка предварительно обученной модели U-Net
    def load_model():
        loaded_model = UNet()
        loaded_model.load_state_dict(torch.load('unet_model.pth', map_location=torch.device('cpu')))
        loaded_model.eval()
        return loaded_model

    model = load_model()

    # ... (остальной код остается без изменений)




    image_path = st.file_uploader("Загрузи свое изображение, и я попробую избавиться от шумов!", type=['jpg', 'png'])
    img = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor(),
    ])

    img_tensor = transform(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)


    threshold = st.slider("Выберите пороговое значение", 0.85, 1.0, 0.97, step=0.01)
    predicted_mask = (output > threshold).float()
    predicted_image = transforms.ToPILImage()(predicted_mask.squeeze())

    #Отображение изображений в Streamlit
    st.image([img, predicted_image], caption=['Original Image', 'Predicted Mask'],
    use_column_width=True, channels='RGB')
    # predicted_mask = (output > threshold).float()
    # predicted_image = transforms.ToPILImage()(predicted_mask.squeeze())
    # col1, col2 = st.beta_columns(2)

    # # Отображение оригинального изображения в первом столбце
    # col1.image(img, caption='Original Image', use_column_width=True, channels='RGB')

    # # Отображение предсказанной маски во втором столбце
    # col2.image(predicted_image, caption='Predicted Mask', use_column_width=True, channels='RGB')
    
    
def page_semantic_segmentation2():
    st.subheader("Семантическая сегментация 2")
    # Добавьте контент для страницы 2



    in_channels = 3  
    out_channels = 1  
    model = SemUNet(in_channels, out_channels)
    model.load_state_dict(torch.load('semseg_best.pth'))
    model.eval()
    

    # Streamlit webpage
    st.title('Semantic Segmentation with U-Net')

    # Threshold slider
    threshold = st.slider('Select a threshold value', 
                        min_value=0.30, 
                        max_value=0.60, 
                        value=0.45,  
                        step=0.00001, 
                        format='%f')

    def transform_image(image):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),   
            transforms.ToTensor(),
        ])
        return transform(image).unsqueeze(0) # Ensures a 4D tensor is returned

    def process_image(image_path, threshold):
        image = Image.open(image_path).convert("RGB")
        original_image_np = np.array(image)
        image_tensor = transform_image(image)  # Now returns a 4D tensor

        with torch.no_grad():
            prediction = model(image_tensor)
        predicted_mask = torch.sigmoid(prediction).data.numpy()
        predicted_mask = (predicted_mask > threshold).astype(np.uint8)
        predicted_mask = np.squeeze(predicted_mask, axis=(0, 1))

        predicted_mask = Image.fromarray(predicted_mask)  # Convert to PIL Image for resizing
        predicted_mask = predicted_mask.resize((original_image_np.shape[1], original_image_np.shape[0]), resample=Image.NEAREST)
        predicted_mask = np.array(predicted_mask)  # Convert back to numpy array

        violet_mask = np.zeros_like(original_image_np)
        violet_mask[:, :, 0] = predicted_mask * 238
        violet_mask[:, :, 2] = predicted_mask * 130
        overlay_image_np = np.where(predicted_mask[..., None], violet_mask, original_image_np).astype(np.uint8)

        return original_image_np, overlay_image_np



    # File uploader
    uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'png', 'jpeg'])

    # Check if a file is uploaded and process it
    if uploaded_file is not None:
        st.session_state['uploaded_file'] = uploaded_file

    if 'uploaded_file' in st.session_state:
        original_image, overlay_image = process_image(st.session_state['uploaded_file'], threshold)

        # Display images
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, use_column_width=True)
            st.caption('Original Image')
        with col2:
            st.image(overlay_image, use_column_width=True)
            st.caption('Segmented Image')

if __name__ == "__main__":
    main()





    











