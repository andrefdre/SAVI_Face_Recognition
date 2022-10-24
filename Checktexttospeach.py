#!/usr/bin/env python3


# pip3 install python-vlc
# sudo apt-get install vlc

from gtts import gTTS
from playsound import playsound

text = "Testa di cazzo."
tts = gTTS(text, lang = 'it')
tts.save("./hi.mp3")
playsound('./hi.mp3')