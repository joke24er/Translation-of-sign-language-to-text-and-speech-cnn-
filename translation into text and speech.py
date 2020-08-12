import googletrans
from googletrans import Translator
from gtts import gTTS

import os
def trans(language,word):
    print(googletrans.LANGUAGES)
    translator=Translator()
    translated = translator.translate(word, src='eo', dest=language)
    print(translated.text)


    myobj = gTTS(text=translated.text, lang=language, slow=True)

    # Saving the converted audio in a mp3 file named
    # welcome
    myobj.save("welcome.mp3")

    # Playing the converted file
    os.system("start welcome.mp3")