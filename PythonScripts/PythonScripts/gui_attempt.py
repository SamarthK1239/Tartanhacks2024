from tkinter import *
from tkintermapview import TkinterMapView


def raise_frame(frame):
    frame.tkraise()

root = Tk()
root.geometry("850x600")

home = Frame(root)
result = Frame(root)

for frame in (home, result):
    frame.grid(row=0, column=0, sticky='news')

# Making Home frame
raise_frame(home)

label = Label(home, text="Welcome", font='Arial 12 bold')
label.grid(row=0, column=0, padx=10, pady=10)



map_view = TkinterMapView(home, width=600, height=400)
map_view.grid(row=1, column=0, padx=10, pady=10)
map_view.set_address('new york')  # Paris, France
map_view.set_zoom(5)



def sel():
    global delay_entry
    delay_entry = Entry(home)
    delay_entry.grid(row = 3, column = 1)


def submit():
    path_1 = map_view.set_path([])
    choice = int(var.get())
    if choice == 2:
        R2.deselect()
        entered = delay_entry.get()
        delay_entry.destroy()
        lst = entered.split(", ")
        #refactor(int(lst[0]), int(lst[1]))
    elif choice == 1:
        R1.deselect()
        #move_normal()

var = IntVar()
R1 = Radiobutton(home, text="Option 1", variable=var, value=1, command=sel)
R1.grid(row = 2, column = 0)
R2 = Radiobutton(home, text="Option 2", variable=var, value=2, command=sel)
R2.grid(row = 3, column = 0)
R3 = Radiobutton(home, text="Option 3", variable=var, value=3, command=sel)
R3.grid(row = 4, column = 0)
label = Label(home)
label.grid(row = 6, column = 0)

Button(home, text="Submit", command=lambda:submit()).grid(row=5, column=0, padx=10, pady=10)

def submit_home():
    pass

# Making result frame

def setup_result(json):
    

    Button(result, text='Back', command=lambda:back_result()).grid(row=row+1, column=0, padx=10, pady=10)

def back_result():
    raise_frame(home)
    for widget in result.winfo_children():
        widget.destroy()
    
root.mainloop()