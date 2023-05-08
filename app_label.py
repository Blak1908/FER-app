import os
import tkinter as tk
import pandas as pd
from PIL import Image, ImageTk

class ImageLabeler:
    def __init__(self, image_folder,data_csv_path):
        self.categories = []
        self.image_folder = image_folder
        self.image_list = os.listdir(image_folder)
        self.current_index = 0
        
        self.dataframe = pd.read_csv(data_csv_path)
        self.root = tk.Tk()
        self.root.title("Image Labeler")
        self.canvas = tk.Canvas(self.root, width=1080, height=600)
        # self.canvas.pack()
        
        self.image_label = self.canvas.create_image(0, 0, anchor="nw")
        self.load_image()

        # Add category buttons
        cats = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection', \
            'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear','Happiness', \
            'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
        
        #Creating a List
        self.cat_box = tk.Listbox(self.root, selectmode=tk.MULTIPLE)
        for i in range(0, len(cats)):
            self.cat_box.insert(i, cats[i])
        
        # selected_item function
        self.btn_add_cats = tk.Button(self.root, text='Add Selected', command=self.selected_item)
        self.btn_add_cats.grid(column=1, row=1)
        
        # Placing the button and listbox
        # self.btn_add_cats.pack(side='bottom')
        self.cat_box.grid(column=1, row=0, sticky=tk.W, padx=650, pady=5)
        # self.cat_box.pack()

        # Add gender radio buttons
        self.gender_var = tk.StringVar(value="male")
        self.gender_label = tk.Label(self.root, text="Gender:")
        self.gender_label.grid(column=2,row=0, sticky=tk.NW, padx=5, pady=5)
        # self.gender_label.pack()
        self.male_radio = tk.Radiobutton(self.root, text="Male", variable=self.gender_var, value="male")
        self.male_radio.grid(column=2,row=1, sticky=tk.NW, padx=5, pady=5)
        # self.male_radio.pack()
        self.male_radio = tk.Radiobutton(self.root, text="Female", variable=self.gender_var, value="female")
        self.male_radio.grid(column=2,row=2, sticky=tk.NW, padx=5, pady=5)
        # self.female_radio.pack()

        # Add radio buttons for age
        self.age_var = tk.StringVar(value="kid")
        self.age_label = tk.Label(self.root, text="Age:")
        self.age_label.grid(column=1,row=2, sticky=tk.S, padx=5, pady=5)
        # self.age_label.pack()

        self.age_kid = tk.Radiobutton(self.root, text="Kid", variable=self.age_var, value="kid")
        self.age_label.grid(column=1,row=3, sticky=tk.NW, padx=5, pady=5)
        # self.age_kid.pack()

        self.age_teen = tk.Radiobutton(self.root, text="Teenager", variable=self.age_var, value="teen")
        self.age_label.grid(column=1,row=4, sticky=tk.NW, padx=5, pady=5)
        # self.age_teen.pack()

        self.age_adult = tk.Radiobutton(self.root, text="Adult", variable=self.age_var, value="adult")
        self.age_label.grid(column=1,row=5, sticky=tk.NW, padx=5, pady=5)
        # self.age_adult.pack()
               
        self.next_button = tk.Button(self.root, text="Next", command=self.next_image)
        self.next_button.grid(column=2,row=3, sticky=tk.NW, padx=5, pady=5)
        # self.next_button.pack()
        
        self.save_button = tk.Button(self.root, text="Save", command=self.save_label)
        self.save_button.grid(column=2,row=4, sticky=tk.NW, padx=5, pady=5)
        # self.save_button.pack()
        
        self.root.mainloop()

    def selected_item(self):
        for i in self.cat_box.curselection():
            print(self.cat_box.get(i))
            self.categories.append(self.cat_box.get(i))
    
    
    def load_image(self):
        image_path = os.path.join(self.image_folder, self.image_list[self.current_index])
        image = Image.open(image_path)
        image = image.resize((600, 600))
        self.photo_image = ImageTk.PhotoImage(image)
        self.canvas.itemconfigure(self.image_label, image=self.photo_image)
        self.canvas.grid(column=0, row=0, columnspan=2,rowspan=6, sticky=tk.W, padx=5, pady=5)
        
    def next_image(self):
        self.current_index += 1
        if self.current_index >= len(self.image_list):
            self.current_index = 0
        self.load_image()
        
    def save_label(self):
        
        # After choose button save, label return a string
        # import pdb; pdb.set_trace()
        # label = self.label_var.get()
        # label_categories = self.category_vars.get()
        # label_gender = self.gender_var.get()
        # label_age = self.age_var.get()
    
        image_path = os.path.join(self.image_folder, self.image_list[self.current_index])
        base_name = image_path.split('/')
        
        # Compare base_name index in csv file
        base_name = base_name.split('.')[0]
        
        # Write code for save label to csv file
           
            
if __name__ == '__main__':
    print("Start: ")
    folder_context = '/home/cuongacpe/workspace/FER-app/data/context'
    csv_path = 'asian.csv'
    print("Init labeler application: ")
    ImageLabeler(folder_context, csv_path)