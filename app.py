import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label, Button
import numpy as np
import cv2
from PIL import Image, ImageTk, UnidentifiedImageError
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.msd import MSDNet  # Assuming MSDNet is defined in model/msd.py
from common.logger import Logger
from common.evaluation import Evaluator
from common.vis import Visualizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Segmentation App')
        
        # Initialize model and paths
        self.model = None
        self.image_path = None
        self.weights_path = None
        
        # UI Elements
        self.load_weights_button = Button(root, text="Load Pretrained Weights", command=self.load_weights)
        self.load_weights_button.pack(pady=10)

        self.load_image_button = Button(root, text="Load Image", command=self.load_image)
        self.load_image_button.pack(pady=10)

        self.segment_button = Button(root, text="Segment Image", command=self.segment_image, state=tk.DISABLED)
        self.segment_button.pack(pady=10)

        self.image_label = Label(root)
        self.image_label.pack(pady=10)
    
    def load_weights(self):
        self.weights_path = filedialog.askopenfilename(title="Select Pretrained Weights", filetypes=[("PyTorch Model", "*.pt")])
        if self.weights_path:
            try:
                logger.info(f"Loading model weights from {self.weights_path}")
                # Load the model architecture and weights
                self.model = self.load_model(self.weights_path)
                messagebox.showinfo("Info", "Model loaded successfully!")
                self.segment_button["state"] = tk.NORMAL
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def load_image(self):
        self.image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            try:
                logger.info(f"Loading image from {self.image_path}")
                img = Image.open(self.image_path)
                img = img.convert('RGB')  # Ensure the image is in RGB format
                img.thumbnail((400, 400))
                img = ImageTk.PhotoImage(img)
                self.image_label.configure(image=img)
                self.image_label.image = img
            except UnidentifiedImageError:
                logger.error("Failed to open image. The file may be corrupted or in an unsupported format.")
                messagebox.showerror("Error", "Failed to open image. The file may be corrupted or in an unsupported format.")
    
    def segment_image(self):
        if not self.model:
            messagebox.showerror("Error", "Please load a pretrained model first!")
            return
        if not self.image_path:
            messagebox.showerror("Error", "Please load an image first!")
            return
        
        logger.info("Starting image segmentation")
        # Read and preprocess image
        image = cv2.imread(self.image_path)
        if image is None:
            logger.error("Failed to read the image. Please check the file format and try again.")
            messagebox.showerror("Error", "Failed to read the image. Please check the file format and try again.")
            return
        original_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (473, 473)) / 255.0
        image_tensor = torch.tensor(image_resized, dtype=torch.float).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Perform segmentation using similar approach to original code
        with torch.no_grad():
            self.model.eval()
            try:
                logger.info("Running model prediction using original testing approach")
                # Using the predict_mask method if it exists or similar logic
                batch = {'query_img': image_tensor, 'support_imgs': image_tensor, 'support_masks': torch.ones_like(image_tensor)}
                logit_mask = self.model(batch['query_img'], batch['support_imgs'], batch['support_masks'])
                pred_mask = (logit_mask > 0.5).float().squeeze(1).cpu().numpy()[0]
                logger.info("Model prediction completed")
            except IndexError as e:
                logger.error(f"Model prediction failed: {str(e)}. Please check the input dimensions.")
                messagebox.showerror("Error", f"Model prediction failed: {str(e)}. Please check the input dimensions.")
                return
            
        # Ensure mask values are either 0 or 1 for correct overlaying
        mask_resized = cv2.resize(pred_mask, (original_image.shape[1], original_image.shape[0]))
        mask_binary = (mask_resized > 0.7).astype(np.uint8)
        mask_rgb = np.zeros_like(original_image, dtype=np.uint8)
        mask_rgb[mask_binary == 1] = [0, 0, 255]  # Red color for the mask
        
        # Overlay the segmentation mask on the original image
        overlaid_image = cv2.addWeighted(original_image, 0.7, mask_rgb, 0.3, 0)
        
        # Convert to ImageTk format for display
        overlaid_image = cv2.cvtColor(overlaid_image, cv2.COLOR_BGR2RGB)
        segmented_image = Image.fromarray(overlaid_image)
        segmented_image.thumbnail((400, 400))
        segmented_image = ImageTk.PhotoImage(segmented_image)
        
        # Display segmented image
        self.image_label.configure(image=segmented_image)
        self.image_label.image = segmented_image
        logger.info("Segmentation completed and displayed")

    def load_model(self, weights_path):
        # Load the MSDNet model
        logger.info("Initializing MSDNet model")
        model = MSDNet(layers=50, shot=1, reduce_dim=64)  # Assuming MSDNet is correctly defined and imported
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        logger.info("Model loaded and moved to device")
        return model

    def convert_segmentation_to_rgb(self, segmentation):
        # Map each class label to a color (example: two classes - background and object)
        colors = np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8)  # Black for background, Red for object
        rgb_image = colors[segmentation.astype(int)]
        return rgb_image

if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationApp(root)
    root.mainloop()
