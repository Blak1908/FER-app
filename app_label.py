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
        self.data_csv_path = data_csv_path
        self.data_frame = pd.read_csv(data_csv_path)
        self.root = tk.Tk()
        self.root.title("Image Labeler")
        self.canvas = tk.Canvas(self.root, width=1080, height=600)
        
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
        self.btn_add_cats.grid(column=2, row=0, sticky=tk.NW, padx=5, pady=180)
        
        # Placing the button and listbox
        # self.btn_add_cats.pack(side='bottom')
        self.cat_box.grid(column=2, row=0, sticky=tk.NW, padx=5)
        # self.cat_box.pack()

        # Add gender radio buttons
        self.gender_var = tk.StringVar(value="male")
        self.gender_label = tk.Label(self.root, text="Gender:")
        self.gender_label.grid(column=2,row=0, sticky=tk.NW, padx=5, pady=220)
        self.male_radio = tk.Radiobutton(self.root, text="Male", variable=self.gender_var, value="Male")
        self.male_radio.grid(column=2,row=0, sticky=tk.NW, padx=5, pady=245)
        self.male_radio = tk.Radiobutton(self.root, text="Female", variable=self.gender_var, value="Female")
        self.male_radio.grid(column=2,row=0, sticky=tk.NW, padx=5, pady=270)

        # Add radio buttons for age
        self.age_var = tk.StringVar(value="kid")
        self.age_label = tk.Label(self.root, text="Age:")
        self.age_label.grid(column=2,row=0, sticky=tk.NW, padx=5, pady=310)

        self.age_kid = tk.Radiobutton(self.root, text="Kid", variable=self.age_var, value="Kid")
        self.age_kid.grid(column=2,row=0, sticky=tk.NW, padx=5, pady=335)

        self.age_teen = tk.Radiobutton(self.root, text="Teenager", variable=self.age_var, value="Teenager")
        self.age_teen.grid(column=2,row=0, sticky=tk.NW, padx=5, pady=360)

        self.age_adult = tk.Radiobutton(self.root, text="Adult", variable=self.age_var, value="Adult")
        self.age_adult.grid(column=2,row=0, sticky=tk.NW, padx=5, pady=385)
               
        self.next_button = tk.Button(self.root, text="Next", command=self.next_image)
        self.next_button.grid(column=2,row=0, sticky=tk.NW, padx=5, pady=410)
        
        self.save_button = tk.Button(self.root, text="Save", command=self.save_label)
        self.save_button.grid(column=2,row=0, sticky=tk.NW, padx=5, pady=435)
        
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
        self.canvas.grid(column=0, row=0, sticky=(tk.N, tk.S, tk.E, tk.W), columnspan=3, rowspan=2)
        
    def next_image(self):
        self.current_index += 1
        if self.current_index >= len(self.image_list):
            self.current_index = 0
        self.load_image()
           
    def save_label(self):
        # Get categories
        label_categories = '['
        for idx,  cat in enumerate(self.categories):
            if idx == 0:
                label_categories = label_categories + f'{cat}'
            else:
                label_categories = label_categories + f', {cat}'
        label_categories = label_categories + ']'
        
        gender = self.gender_var.get()
        age = self.age_var.get()

    
        image_path = os.path.join(self.image_folder, self.image_list[self.current_index])
        base_name = image_path.split('/')[-1]
        
        # Find the row 
        idx = self.data_frame.index[self.data_frame['Filename'] == base_name][0]

        # Add the categories, gender, and age
        self.data_frame.at[idx, 'Categorical_Labels'] = label_categories
        self.data_frame.at[idx, 'Gender'] = gender
        self.data_frame.at[idx, 'Age'] = age

        # Save to CSV file
        self.data_frame.to_csv(self.data_csv_path, index=False)
        
        # reset categories list
        self.clean_data()
        print("Label Sucess!")  
        
    def clean_data(self):
        self.categories = []
        
if __name__ == '__main__':
    print("Start: ")
    folder_context = '/home/cuongacpe/workspace/FER-app/data/context'
    csv_path = 'asian.csv'
    print("Init labeler application: ")
    ImageLabeler(folder_context, csv_path)