import io
import os
import time
import argparse
import threading
from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
import speech_recognition as sr
import whisper
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline





def transcribe_audio(audio_data, audio_model, temp_file):
    wav_data = io.BytesIO(audio_data.get_wav_data())
    with open(temp_file, 'w+b') as f:
        f.write(wav_data.read())
    result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
    text = result['text'].strip()
    return text


def translate_text(text, translation_pipeline):
    translated_text = translation_pipeline(text, max_length=450)[0]['translation_text']
    return translated_text

data_queue = Queue()

def main():
    global data_queue

    # 初始化翻译模型
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cuda:0", type=str, help='Device to use (e.g. "cpu", "cuda:0")')
    args = parser.parse_args()
    device = args.device
    model_name = 'DDDSSS/translation_en-zh'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    translation_pipeline = pipeline("translation_en_to_zh", model=model, tokenizer=tokenizer,
                                     torch_dtype="float", device=device)

    print("Model loaded")

    # 初始化语音模型
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=4,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # The last time a recording was retreived from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout




    temp_file = NamedTemporaryFile(delete=False).name
    transcription = ['']


    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")

    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                phrase_time = now

                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)

                text = transcribe_audio(audio_data, audio_model, temp_file)

                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription:
                    print(line)
                    translated_text = translate_text(line, translation_pipeline)
                    print("翻译：", translated_text)
                print('', end='', flush=True)

                sleep(0.25)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\n\nTranscription:")
    for line in transcription:
        print(line)

    os.remove(temp_file)


if __name__ == "__main__":
    main()
