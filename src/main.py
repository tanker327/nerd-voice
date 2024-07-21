import tkinter as tk
from tkinter import Text
import sounddevice as sd
import numpy as np
import io
import soundfile as sf
from pydub import AudioSegment
import tempfile
import os
from ai import GPT4
import pyperclip
from utils.logger import LOG

# from ai.whisper_wrapper import transcript

APP_WIDTH = 193
APP_HEIGHT = 410


class VoiceApp:
    def __init__(self, master):
        self.master = master
        master.title("Nerd Voice")
        self.master.geometry(self._get_center_geometry_str())
        self.master.resizable(False, False)
        self.master.attributes("-topmost", True)

        top_frame = tk.Frame(master, width=APP_WIDTH, height=200)
        top_frame.grid(row=0, column=0, padx=6, pady=5)

        top_right_frame = tk.Frame(top_frame, width=APP_WIDTH - 20, height=200)
        top_right_frame.grid(row=0, column=1)

        bottom_frame = tk.Frame(master, width=APP_WIDTH, height=200)
        bottom_frame.grid(row=1, column=0, padx=6, pady=0)

        self.record_button = tk.Button(
            top_frame, height=8, width=10, text="Record", command=self.toggle_record
        )
        self.record_button.grid(row=0, column=0)
        # 当窗口被激活时，设置焦点到 btn_record 上
        self.master.bind("<FocusIn>", lambda event: self.record_button.focus_set())
        # 绑定空格键到 toggle_recording 函数
        # self.record_button.bind("<space>", self.toggle_record)

        self.status_canvas = tk.Canvas(top_right_frame, width=50, height=90)
        self.status_canvas.grid(row=0, column=0)
        self.status_rect = self.status_canvas.create_rectangle(
            0, 0, 50, 90, fill="green", width=0
        )

        self.play_button = tk.Button(
            top_right_frame, height=2, width=2, text="Play", command=self.play
        )
        self.play_button.grid(row=1, column=0)

        self.text = Text(bottom_frame, height=19, width=25)
        self.text.grid(row=0, column=0)

        self.is_recording = False
        self.fs = 44100  # Sample rate
        self.audio_data = []
        self.stream = None
        self.buffer = None

    def callback(self, indata, frames, time, status):
        self.audio_data.append(indata.copy())

    def toggle_record(self, _=None):
        if not self.is_recording:
            LOG.debug("Start recording....")

            self.is_recording = True
            self.status_canvas.itemconfig(self.status_rect, fill="red")
            self.record_button.config(text="Stop")
            self.audio_data = []

            self.stream = sd.InputStream(
                callback=self.callback,
                channels=1,  # Changed from 2 to 1
                samplerate=self.fs
            )
            self.stream.start()
        else:
            LOG.debug("Stop recording. ")
            self.is_recording = False
            self.record_button.config(text="Transcribing...")
            self.status_canvas.itemconfig(self.status_rect, fill="yellow")
            self.record_button.config(state="disabled")
            self.master.update()  # update UI

            self.stream.stop()
            self.stream.close()

            self.buffer = io.BytesIO()
            if len(self.audio_data) <= 0:
                raise Exception("No audio data recorded.")
            audio_array = np.concatenate(self.audio_data, axis=0)
            np.save(self.buffer, audio_array)
            self.buffer.seek(0)
            self.handle_record()
            self.record_button.config(text="Record")
            self.record_button.config(state="normal")

    def play(self):
        if self.buffer and not self.is_recording:
            self.buffer.seek(0)
            audio_play_data = np.load(self.buffer)
            sd.play(audio_play_data, self.fs)
            sd.wait()

    def handle_record(self):
        if self.buffer:
            temp_dir = tempfile.gettempdir()
            temp_wav = os.path.join(temp_dir, "temp.wav")
            temp_mp3 = os.path.join(temp_dir, "temp.mp3")

            self.buffer.seek(0)
            audio_data = np.load(self.buffer)
            sf.write(temp_wav, audio_data, self.fs)

            sound = AudioSegment.from_wav(temp_wav)
            sound.export(temp_mp3, format="mp3")

            LOG.debug("mp3 file created.")
            
            ##====== insert mp3 file path here ======##
            ## temp_mp3 = "/Users/ericwu/Downloads/study.mp3"
            ##========================================##

            result = GPT4.transcribe(f"{temp_mp3}")

            # result = transcript(f"{temp_mp3}")

            self.text.delete(1.0, tk.END)
            self.text.insert(tk.END, result)

            LOG.debug("Text ready")

            pyperclip.copy(result)  # copy content to clipboard

            self.status_canvas.itemconfig(self.status_rect, fill="green")

    def _get_center_geometry_str(self):
        return f"{APP_WIDTH}x{APP_HEIGHT}+{int((self.master.winfo_screenwidth() - APP_WIDTH) / 2)}+{int((self.master.winfo_screenheight() - APP_HEIGHT) / 2)}"


root = tk.Tk()
app = VoiceApp(root)
root.mainloop()
