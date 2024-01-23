import sys
import tkinter
from tkinter import *
import pyttsx3

# Create a new Tkinter Form
master1 = tkinter.Tk()
e = Entry(master1, width=30)

# Function to convert text to
# speech
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    rate = engine.getProperty('rate')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 145)
    engine.say(command)
    engine.runAndWait()


# Function to save new Student Number
def saveID(name):
    # Gets the new value in the tkinter text box
    with open("New_Data.txt", "w") as text_file:
        text_file.write(name)

    master1.quit()
    master1.destroy()
    import registration
    sys.exit()


def proceed():
    entered_ni = str(e.get())
    if len(entered_ni) > 0:
        saveID(entered_ni)


# Tkinter dialogue settings
master1.title('Enter Your Student Number Below')
master1.geometry("300x300")

# Tkinter textbox settings
# Create a text box to enter Student number

e.pack()
e.focus_set()

# Tkinter button settings
b = Button(master1, text="Click To Proceed", width=20, command=proceed)
b.pack()

print("WELCOME PLEASE ENTER YOUR STUDENT NUMBER")
SpeakText("WELCOME PLEASE ENTER YOUR STUDENT NUMBER")

# Start the program loop
master1.mainloop()

