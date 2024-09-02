# This will be the GUI using tkinter. Should allow start/ stop of recording with it being saved to file and 

import sounddevice as sd 
import soundfile as sf 
from tkinter import *

# create functions for gui buttons

def start_rec(): 
	myrecording = sd.rec(samplerate=48000, channels=2) 
	return myrecording

def stop_rec(myrecording):
	sd.stop()
	sd.wait() 
	return sf.write('audioFile.mp3', myrecording)

master = Tk() 
master.title("Meeting Guru")
master.geometry("500x500")

b1 = Button(master, text="Start Recording", command=start_rec) 
b1.place(x=25, y=100)

b2 = Button(master, text="Stop Recording", command=stop_rec) 
b2.place(x=175, y=100)

mainloop()
