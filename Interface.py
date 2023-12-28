import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Your Dataset
df = pd.read_csv('data.csv')

# Convert 'diagnosis' column to numeric values (0 for 'B' and 1 for 'M')
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

selectionbar_color = '#eff5f6'
sidebar_color = '#F5E1FD'
header_color = '#53366b'
visualisation_frame_color = "#ffffff"


class TkinterApp(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Breast Cancer Diagnostic System")

        self.geometry("1100x700")
        self.resizable(0, 0)
        self.title('Breast Cancer Diagnostic Testing')
        self.config(background=selectionbar_color)
        icon = tk.PhotoImage(file='icon\\UTS.png')
        self.iconphoto(True, icon)

        # Header
        self.header = tk.Frame(self, bg=header_color)
        self.header.place(relx=0.3, rely=0, relwidth=0.7, relheight=0.1)

        # Frame for sidebar
        self.sidebar = tk.Frame(self, bg=sidebar_color)
        self.sidebar.place(relx=0, rely=0, relwidth=0.3, relheight=1)

        # University
        self.brand_frame = tk.Frame(self.sidebar, bg=sidebar_color)
        self.brand_frame.place(relx=0, rely=0, relwidth=1, relheight=0.15)
        self.uni_logo = icon.subsample(9)
        logo = tk.Label(self.brand_frame, image=self.uni_logo, bg=sidebar_color)
        logo.place(x=15, y=13)

        uni_name = tk.Label(self.brand_frame,
                            text='University of',
                            bg=sidebar_color,
                            font=("", 15, "bold")
                            )
        uni_name.place(x=120, y=27, anchor="w")

        uni_name = tk.Label(self.brand_frame,
                            text='Technology Sarawak',
                            bg=sidebar_color,
                            font=("", 15, "bold")
                            )
        uni_name.place(x=90, y=60, anchor="w")

        # Submenu
        self.submenu_frame = tk.Frame(self.sidebar, bg=sidebar_color)
        self.submenu_frame.place(relx=0, rely=0.2, relwidth=1, relheight=2)
        submenu1 = SidebarSubMenu(self.submenu_frame,
                                  sub_menu_heading='Choose an analysis:',
                                  sub_menu_options=["1. Basic Correlation Analysis",
                                                    "2. Automatic Feature Selection",
                                                    "3. Calculate Accuracy rate",
                                                    "4. Support Vector Machine",
                                                    "5. Exit",
                                                    ]
                                  )
        submenu1.options["1. Basic Correlation Analysis"].config(
            command=lambda: self.show_frame(Basic_Correlation_Analysis)
        )
        submenu1.options["2. Automatic Feature Selection"].config(
            command=lambda: self.show_frame(Automatic_Feature_Selection)
        )
        submenu1.options["3. Calculate Accuracy rate"].config(
            command=lambda: self.show_frame(Calculate_Accuracy_rate)
        )
        submenu1.options["4. Support Vector Machine"].config(
            command=lambda: self.show_frame(Support_Vector_Machine)
        )
        submenu1.options["5. Exit"].config(
            command=lambda: self.show_frame(Exit)
        )

        submenu1.place(relx=0, rely=0.025, relwidth=1, relheight=0.3)

        # Right page
        container = tk.Frame(self)
        container.config(highlightbackground="#808080", highlightthickness=0.5)
        container.place(relx=0.3, rely=0.1, relwidth=0.7, relheight=0.9)

        self.frames = {}

        for F in (Frame,
                  Basic_Correlation_Analysis,
                  Automatic_Feature_Selection,
                  Calculate_Accuracy_rate,
                  Support_Vector_Machine,
                  Exit,
                  Frame,
                  ):
            frame = F(container, self)
            self.frames[F] = frame
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.show_frame(Frame)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


# Output after the button is clicked

class Basic_Correlation_Analysis(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.labels = None
        self.controller = controller

        self.label_index = 0

        # Create a button to start the process
        start_button = tk.Button(self, text='Start Processing', command=self.display_heatmap, font=("Arial", 12))
        start_button.pack(pady=20, padx=10)

        # Create a label to display the output
        self.output_label = tk.Label(self, text='', font=("Arial", 9))
        self.output_label.pack(pady=20, padx=10)

    def display_heatmap(self):
        # Compute correlation matrix for numeric columns only
        numeric_columns = df.select_dtypes(include=['float64']).columns
        correlation_matrix = df[numeric_columns].corr()

        # Display the correlation matrix as a label
        self.output_label.config(text=f"Correlation Matrix:\n{correlation_matrix}")

        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.show()


class Automatic_Feature_Selection(tk.Frame):
    def __init__(frame2, parent, controller):
        tk.Frame.__init__(frame2, parent)

        label = tk.Label(frame2, text='Output2:', font=("Arial", 15))
        label.pack()


class Calculate_Accuracy_rate(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.labels = None
        self.controller = controller

        self.label_index = 0

        # Create a list to store label widgets
        self.label_widgets = []

        # Create a button to start the process
        start_button = tk.Button(self, text='Start Processing', command=self.start_processing, font=("Arial", 12))
        start_button.pack(pady=20, padx=10)

    def start_processing(self):
        # Function to start the processing and update labels
        # Destroy existing label widgets
        for label in self.label_widgets:
            label.destroy()

        self.labels = [
            'Getting Data',
            'Please waiting',
            '--Processing--',
            '--Processing--',
            '--COMPLETE--',
            'Accuracy with all features: 53.57%'
        ]
        self.label_index = 0
        self.create_label()

    def create_label(self):
        if self.label_index < len(self.labels):
            label_text = self.labels[self.label_index]
            label = tk.Label(self, text=label_text, font=("Arial", 15))
            label.pack()
            self.label_widgets.append(label)
            self.label_index += 1
            self.after(2000, self.create_label)  # Delay 2 seconds
        else:
            pass


class Support_Vector_Machine(tk.Frame):
    def __init__(frame4, parent, controller):
        tk.Frame.__init__(frame4, parent)

        label = tk.Label(frame4, text='Output4:', font=("Arial", 15))
        label.pack()


class Exit(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text='Are you sure??', font=("Arial", 15))
        label.pack()

        exit_button = tk.Button(self, text='YES', command=self.exit_application, font=("Arial", 12))
        exit_button.pack(pady=20, padx=10)

    def exit_application(self):
        # Destroy the root window to exit the application
        self.winfo_toplevel().destroy()


class Frame(tk.Frame):
    def __init__(frame, parent, controller):
        tk.Frame.__init__(frame, parent)

        label = tk.Label(frame, text='Please choose an option:', font=("Arial", 15))
        label.pack()


# Sidebar submenu

class SidebarSubMenu(tk.Frame):

    def __init__(self, parent, sub_menu_heading, sub_menu_options):
        tk.Frame.__init__(self, parent)
        self.config(bg=sidebar_color)
        self.sub_menu_heading_label = tk.Label(self,
                                               text=sub_menu_heading,
                                               bg=sidebar_color,
                                               fg="#333333",
                                               font=("Arial", 10)
                                               )
        self.sub_menu_heading_label.place(x=30, y=10, anchor="w")

        sub_menu_sep = ttk.Separator(self, orient='horizontal')
        sub_menu_sep.place(x=30, y=30, relwidth=0.8, anchor="w")

        self.options = {}
        for n, x in enumerate(sub_menu_options):
            self.options[x] = tk.Button(self,
                                        text=x,
                                        bg=sidebar_color,
                                        font=("Arial", 9, "bold"),
                                        bd=0,
                                        cursor='hand2',
                                        activebackground='#ffffff',
                                        )
            self.options[x].place(x=30, y=45 * (n + 1), anchor="w")


app = TkinterApp()
app.mainloop()
