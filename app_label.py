import os
import tkinter as tk
from PIL import Image, ImageTk

class ImageLabeler:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_list = os.listdir(image_folder)
        self.current_index = 0
        
        self.root = tk.Tk()
        self.root.title("Image Labeler")

        self.canvas = tk.Canvas(self.root, width=600, height=600)
        self.canvas.pack()
        
        self.image_label = self.canvas.create_image(0, 0, anchor="nw")
        self.load_image()
        
        self.label_var = tk.StringVar(value="unlabeled")
        self.label_entry = tk.Entry(self.root, textvariable=self.label_var)
        self.label_entry.pack()
        
        self.next_button = tk.Button(self.root, text="Next", command=self.next_image)
        self.next_button.pack()
        
        self.save_button = tk.Button(self.root, text="Save", command=self.save_label)
        self.save_button.pack()
        
        self.root.mainloop()
    
    def load_image(self):
        image_path = os.path.join(self.image_folder, self.image_list[self.current_index])
        image = Image.open(image_path)
        image = image.resize((600, 600))
        self.photo_image = ImageTk.PhotoImage(image)
        self.canvas.itemconfigure(self.image_label, image=self.photo_image)
        
    def next_image(self):
        self.current_index += 1
        if self.current_index >= len(self.image_list):
            self.current_index = 0
        self.load_image()
        
    def save_label(self):
        label = self.label_var.get()
        image_path = os.path.join(self.image_folder, self.image_list[self.current_index])
        with open("labels.txt", "a") as f:
            f.write(f"{image_path} {label}\n")
            
            
if __name__ == '__main__':
    print("Start: ")
    ImageLabeler('/Users/trantuandat/Downloads/scp-result/images')