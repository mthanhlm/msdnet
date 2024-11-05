import logging
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
from model.msd import MSDNet
import numpy as np
from torchvision import transforms
import os

class FSSTestingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FSS Testing Application")

        # Initialize logging
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("FSSTestingApp initialized.")
        
        # Khởi tạo mô hình
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((473, 473)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Tạo GUI
        self.create_widgets()
        
        # Lưu trữ ảnh
        self.support_img = None
        self.support_mask = None
        self.query_img = None
        
    def create_widgets(self):
        # Frame cho model loading
        model_frame = ttk.LabelFrame(self.root, text="Model Configuration", padding=10)
        model_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        ttk.Label(model_frame, text="Model Path:").grid(row=0, column=0, sticky="w")
        self.model_path = ttk.Entry(model_frame, width=50)
        self.model_path.grid(row=0, column=1, padx=5)
        
        ttk.Button(model_frame, text="Browse", command=self.browse_model).grid(row=0, column=2, padx=5)
        ttk.Button(model_frame, text="Load Model", command=self.load_model).grid(row=0, column=3, padx=5)
        
        # Frame cho support image
        support_frame = ttk.LabelFrame(self.root, text="Support Image & Mask", padding=10)
        support_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        ttk.Label(support_frame, text="Support Image:").grid(row=0, column=0, sticky="w")
        self.support_path = ttk.Entry(support_frame, width=50)
        self.support_path.grid(row=1, column=0, padx=5)
        ttk.Button(support_frame, text="Browse", command=self.browse_support_image).grid(row=1, column=1, padx=5)
        
        ttk.Label(support_frame, text="Support Mask:").grid(row=2, column=0, sticky="w")
        self.mask_path = ttk.Entry(support_frame, width=50)
        self.mask_path.grid(row=3, column=0, padx=5)
        ttk.Button(support_frame, text="Browse", command=self.browse_support_mask).grid(row=3, column=1, padx=5)
        
        ttk.Button(support_frame, text="Load Support", command=self.load_support).grid(row=4, column=0, pady=5)
        
        self.support_preview = ttk.Label(support_frame)
        self.support_preview.grid(row=5, column=0, columnspan=2)
        
        # Frame cho query image
        query_frame = ttk.LabelFrame(self.root, text="Query Image", padding=10)
        query_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
        ttk.Label(query_frame, text="Query Image:").grid(row=0, column=0, sticky="w")
        self.query_path = ttk.Entry(query_frame, width=50)
        self.query_path.grid(row=1, column=0, padx=5)
        ttk.Button(query_frame, text="Browse", command=self.browse_query_image).grid(row=1, column=1, padx=5)
        
        ttk.Button(query_frame, text="Load Query", command=self.load_query).grid(row=2, column=0, pady=5)
        
        self.query_preview = ttk.Label(query_frame)
        self.query_preview.grid(row=3, column=0, columnspan=2)
        
        # Frame cho kết quả
        result_frame = ttk.LabelFrame(self.root, text="Results", padding=10)
        result_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        ttk.Button(result_frame, text="Run Prediction", command=self.run_prediction).grid(row=0, column=0, columnspan=2, pady=5)
        
        # Tạo hai label cho hai kết quả
        self.support_result = ttk.Label(result_frame)
        self.support_result.grid(row=1, column=0, padx=5)
        
        self.query_result = ttk.Label(result_frame)
        self.query_result.grid(row=1, column=1, padx=5)

    def create_overlay(self, image, mask, color):
        """
        Tạo overlay mask với màu được chỉ định lên ảnh
        color: tuple (R,G,B) cho màu của mask
        """
        # Chuyển mask về cùng kích thước với ảnh
        mask = mask.resize(image.size, Image.NEAREST)
        
        # Tạo ảnh màu từ mask
        colored_mask = Image.new('RGB', image.size, color)
        
        # Tạo alpha channel từ mask
        alpha = Image.fromarray(np.array(mask) * 0.5)  # 50% transparency
        
        # Chuyển ảnh gốc sang RGBA
        image_rgba = image.convert('RGBA')
        colored_mask_rgba = colored_mask.convert('RGBA')
        
        # Blend ảnh với mask
        result = Image.composite(colored_mask_rgba, image_rgba, alpha)
        return result

    def browse_model(self):
        filename = filedialog.askopenfilename(
            title="Select model file",
            filetypes=(("PyTorch files", "*.pt"), ("All files", "*.*"))
        )
        if filename:
            self.model_path.delete(0, tk.END)
            self.model_path.insert(0, filename)

    def browse_support_image(self):
        filename = filedialog.askopenfilename(
            title="Select support image",
            filetypes=(("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*"))
        )
        if filename:
            self.support_path.delete(0, tk.END)
            self.support_path.insert(0, filename)

    def browse_support_mask(self):
        filename = filedialog.askopenfilename(
            title="Select support mask",
            filetypes=(("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*"))
        )
        if filename:
            self.mask_path.delete(0, tk.END)
            self.mask_path.insert(0, filename)

    def browse_query_image(self):
        filename = filedialog.askopenfilename(
            title="Select query image",
            filetypes=(("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*"))
        )
        if filename:
            self.query_path.delete(0, tk.END)
            self.query_path.insert(0, filename)
    
    def load_model(self):
        try:
            model_path = self.model_path.get()
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            self.model = MSDNet(layers=50, shot=1, reduce_dim=64)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def load_support(self):
        try:
            # Load support image
            support_path = self.support_path.get()
            mask_path = self.mask_path.get()
            
            if not os.path.exists(support_path):
                raise FileNotFoundError(f"Support image not found: {support_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Support mask not found: {mask_path}")
            
            self.support_img = Image.open(support_path).convert('RGB')
            preview = self.support_img.resize((200, 200))
            photo = ImageTk.PhotoImage(preview)
            self.support_preview.configure(image=photo)
            self.support_preview.image = photo
            
            # Load support mask
            self.support_mask = Image.open(mask_path).convert('L')
            # Convert mask to binary
            self.support_mask = Image.fromarray((np.array(self.support_mask) > 128).astype(np.uint8) * 255)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load support images: {str(e)}")
    
    def load_query(self):
        try:
            query_path = self.query_path.get()
            if not os.path.exists(query_path):
                raise FileNotFoundError(f"Query image not found: {query_path}")
                
            self.query_img = Image.open(query_path).convert('RGB')
            preview = self.query_img.resize((200, 200))
            photo = ImageTk.PhotoImage(preview)
            self.query_preview.configure(image=photo)
            self.query_preview.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load query image: {str(e)}")
    
    def prepare_batch(self):
        # Prepare support image
        support_tensor = self.transform(self.support_img)
        support_mask_tensor = transforms.ToTensor()(self.support_mask)
        
        # Prepare query image
        query_tensor = self.transform(self.query_img)
        
        # Create batch dictionary
        batch = {
            'support_imgs': support_tensor.unsqueeze(0).to(self.device),
            'support_masks': support_mask_tensor.unsqueeze(0).to(self.device),
            'query_img': query_tensor.unsqueeze(0).to(self.device),
            'org_query_imsize': torch.tensor(self.query_img.size[::-1]).to(self.device)
        }
        
        return batch
    
    def run_prediction(self):
        if not all([self.model, self.support_img, self.support_mask, self.query_img]):
            messagebox.showerror("Error", "Please load all required components first!")
            return
            
        try:
            with torch.no_grad():
                batch = self.prepare_batch()
                logit_mask = self.model.predict_mask(batch)
                pred_mask = (logit_mask > 0.5).float().squeeze()
                
                # Convert prediction to image
                pred_mask_np = pred_mask.cpu().numpy() * 255
                pred_mask_img = Image.fromarray(pred_mask_np.astype(np.uint8))
                
                # Create overlays
                support_overlay = self.create_overlay(
                    self.support_img, 
                    self.support_mask, 
                    (0, 255, 0)  # Green for support mask
                )
                query_overlay = self.create_overlay(
                    self.query_img,
                    pred_mask_img,
                    (255, 0, 0)  # Red for prediction mask
                )
                
                # Resize for display
                support_preview = support_overlay.resize((400, 400))
                query_preview = query_overlay.resize((400, 400))
                
                # Convert to PhotoImage and display
                support_photo = ImageTk.PhotoImage(support_preview)
                query_photo = ImageTk.PhotoImage(query_preview)
                
                self.support_result.configure(image=support_photo)
                self.support_result.image = support_photo
                
                self.query_result.configure(image=query_photo)
                self.query_result.image = query_photo
                
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FSSTestingApp(root)
    root.mainloop()