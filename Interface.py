import tkinter as tk
from tkinter import ttk

selectionbar_color = '#eff5f6'
sidebar_color = '#F5E1FD'
header_color = '#53366b'
visualisation_frame_color = "#ffffff"


class TkinterApp(tk.Tk):

    def __init__(window):
        tk.Tk.__init__(window)
        window.title("Breast Cancer Diagnostic System")

        window.geometry("1100x700")
        window.resizable(0, 0)
        window.title('Breast Cancer Diagnostic Testing')
        window.config(background=selectionbar_color)
        icon = tk.PhotoImage(file='icon\\UTS.png')
        window.iconphoto(True, icon)

        # Header
        window.header = tk.Frame(window, bg=header_color)
        window.header.place(relx=0.3, rely=0, relwidth=0.7, relheight=0.1)

        # Frame for sidebar
        window.sidebar = tk.Frame(window, bg=sidebar_color)
        window.sidebar.place(relx=0, rely=0, relwidth=0.3, relheight=1)

        # University
        window.brand_frame = tk.Frame(window.sidebar, bg=sidebar_color)
        window.brand_frame.place(relx=0, rely=0, relwidth=1, relheight=0.15)
        window.uni_logo = icon.subsample(9)
        logo = tk.Label(window.brand_frame, image=window.uni_logo, bg=sidebar_color)
        logo.place(x=15, y=13)

        uni_name = tk.Label(window.brand_frame,
                            text='University of',
                            bg=sidebar_color,
                            font=("", 15, "bold")
                            )
        uni_name.place(x=120, y=27, anchor="w")

        uni_name = tk.Label(window.brand_frame,
                            text='Technology Sarawak',
                            bg=sidebar_color,
                            font=("", 15, "bold")
                            )
        uni_name.place(x=90, y=60, anchor="w")

        # Submenu
        window.submenu_frame = tk.Frame(window.sidebar, bg=sidebar_color)
        window.submenu_frame.place(relx=0, rely=0.2, relwidth=1, relheight=2)
        submenu1 = SidebarSubMenu(window.submenu_frame,
                                  sub_menu_heading='Choose an analysis:',
                                  sub_menu_options=["1. Basic Correlation Analysis",
                                                    "2. Automatic Feature Selection",
                                                    "3. Calculate Accuracy rate",
                                                    "4. Support Vector Machine",
                                                    "5. Exit",
                                                    ]
                                  )
        submenu1.options["1. Basic Correlation Analysis"].config(
            command=lambda: window.show_frame(Basic_Correlation_Analysis)
        )
        submenu1.options["2. Automatic Feature Selection"].config(
            command=lambda: window.show_frame(Automatic_Feature_Selection)
        )
        submenu1.options["3. Calculate Accuracy rate"].config(
            command=lambda: window.show_frame(Calculate_Accuracy_rate)
        )
        submenu1.options["4. Support Vector Machine"].config(
            command=lambda: window.show_frame(Support_Vector_Machine)
        )
        submenu1.options["5. Exit"].config(
            command=lambda: window.show_frame(Exit)
        )

        submenu1.place(relx=0, rely=0.025, relwidth=1, relheight=0.3)

        # Right page
        container = tk.Frame(window)
        container.config(highlightbackground="#808080", highlightthickness=0.5)
        container.place(relx=0.3, rely=0.1, relwidth=0.7, relheight=0.9)

        window.frames = {}

        for F in (Basic_Correlation_Analysis,
                  Automatic_Feature_Selection,
                  Calculate_Accuracy_rate,
                  Support_Vector_Machine,
                  Exit,
                  Frame,
                  ):
            frame = F(container, window)
            window.frames[F] = frame
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        window.show_frame(Frame)

    def show_frame(results, cont):
        frame = results.frames[cont]
        frame.tkraise()


# Output after the button is clicked

class Basic_Correlation_Analysis(tk.Frame):
    def __init__(frame1, parent, controller):
        tk.Frame.__init__(frame1, parent)

        label = tk.Label(frame1, text='Output1:', font=("Arial", 15))
        label.pack()


class Automatic_Feature_Selection(tk.Frame):
    def __init__(frame2, parent, controller):
        tk.Frame.__init__(frame2, parent)

        label = tk.Label(frame2, text='Output2:', font=("Arial", 15))
        label.pack()


class Calculate_Accuracy_rate(tk.Frame):
    def __init__(frame3, parent, controller):
        tk.Frame.__init__(frame3, parent)

        label = tk.Label(frame3, text='Accuracy with all features: 53.57%', font=("Arial", 15))
        label.pack()


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
