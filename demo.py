import io
import os
import random
import threading
import tkinter as tk
from hashlib import md5
from queue import Queue
from tkinter import font

import requests
import speech_recognition as sr
import whisper
from dotenv import load_dotenv

# 下载/加载whisper语音模型
model = 'base.en'
audio_model = whisper.load_model(model, device='cuda:0', download_root='./models', in_memory=True)
print("whisper Model loaded.\n")

# 加载.env文件中的变量
load_dotenv()

# 初始化音频设备
recognizer = sr.Recognizer()
source = sr.Microphone(sample_rate=16000)
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = True

# 初始化所有变量
data_queue = Queue()
audio_queue = Queue()
last_sample = bytes()
phrase_time = None
temp_file = './temp/temp.wav'
transcription = ['']
text_date = Queue()
App_id = os.getenv('appid')
App_key = os.getenv('appkey')

with source:
    recognizer.adjust_for_ambient_noise(source)


def record_callback(_, audio: sr.AudioData) -> None:
    data = audio.get_raw_data()
    audio_queue.put(data)
    if not audio_queue.empty():
        data = audio_queue.get()
        _audio(data)


recognizer.listen_in_background(source, record_callback, phrase_time_limit=5)  #phrase_time_limit=5 延时5秒


# 音频处理 保存到临时文件
def _audio(audio_data):
    try:
        wav = sr.AudioData(audio_data, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
        wav_data = io.BytesIO(wav.get_wav_data())

        with open(temp_file, 'w+b') as f:
            f.write(wav_data.read())
        data_queue.put(temp_file)
    except Exception as e:
        print(f"An error occurred: {e}")


# 获取处理后的音频 转换成文字
def whisper_to_E_text():
    if not data_queue.empty():
        audio_file = data_queue.get()
        result = audio_model.transcribe(audio_file, temperature=0.4)
        text = result['text'].strip()
        # print(f"text: {text}")
        return text
    else:
        return None


# 获取英文文字 翻译成中文
def baidu_tran(E_text):
    if E_text is not None:
        appid = App_id
        appkey = App_key
        from_lang = 'en'
        to_lang = 'zh'
        endpoint = 'http://api.fanyi.baidu.com'
        path = '/api/trans/vip/translate'
        url = endpoint + path
        query = E_text

        def make_md5(s, encoding='utf-8'):
            return md5(s.encode(encoding)).hexdigest()

        salt = random.randint(32768, 65536)
        sign = make_md5(appid + query + str(salt) + appkey)
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}
        r = requests.post(url, params=payload, headers=headers)
        result = r.json()

        if 'trans_result' in result and len(result['trans_result']) > 0:
            C_text = result['trans_result'][0]['dst'].strip()
            # print('<<:', C_text)
            return C_text
        else:
            print("Translation result not found in the API response.")
            return None

    else:
        return None


# 将中 英文保存到变量 待显示到窗口
def update_text():
    while True:
        E_text = whisper_to_E_text()
        if E_text is not None:
            C_text = baidu_tran(E_text)
            if C_text:
                # 将需要更新的文本放入队列
                text_date.put((E_text, C_text))
            # time.sleep(1)


# 创建窗口 并获取中 英文文本显示到窗口
def create_window():
    window = tk.Tk()
    window.geometry("1200x80")  # 初始窗口大小，可以随意拉大或缩小
    window.overrideredirect(True)  # 隐藏标题栏
    window.attributes('-alpha', 0.5)  # 设置窗口透明度为50%
    window.attributes('-topmost', 1)  # 设置窗口置顶

    # 创建一个Frame容器，用于放置两个Label
    frame = tk.Frame(window)
    frame.pack(fill=tk.BOTH, expand=True)

    E_text, C_text = text_date.get()

    # 创建两个Label，一个用于显示英文，一个用于显示中文
    english_text = tk.Label(frame, text=E_text, bg="black", fg="white")
    chinese_text = tk.Label(frame, text=C_text, bg="black", fg="white")

    # 设置字体大小
    english_font = font.Font(size=13)
    chinese_font = font.Font(size=16)
    english_text.config(font=english_font, anchor='nw', justify='left')
    chinese_text.config(font=chinese_font, anchor='nw', justify='left')

    # 将Label放置到Frame容器中
    english_text.pack(fill=tk.BOTH, expand=True)
    chinese_text.pack(fill=tk.BOTH, expand=True)

    def update_wraplength(event):
        english_text.config(wraplength=window.winfo_width())
        chinese_text.config(wraplength=window.winfo_width())

    window.bind("<Configure>", update_wraplength)

    # 添加鼠标事件处理函数
    def start_move(event):
        window.x = event.x
        window.y = event.y

    def stop_move(event):
        window.x = None
        window.y = None

    def do_move(event):
        dx = event.x - window.x
        dy = event.y - window.y
        x = window.winfo_x() + dx
        y = window.winfo_y() + dy
        window.geometry(f"+{x}+{y}")

    def start_resize(event):
        window.startX = event.x
        window.startY = event.y

    def stop_resize(event):
        window.startX = None
        window.startY = None

    def do_resize(event):
        dx = event.x - window.startX
        dy = event.y - window.startY
        width = window.winfo_width() + dx
        height = window.winfo_height() + dy
        window.geometry(f"{width}x{height}")

    # 绑定鼠标事件到窗口上
    window.bind("<ButtonPress-1>", start_move)
    window.bind("<ButtonRelease-1>", stop_move)
    window.bind("<B1-Motion>", do_move)

    # 创建一个用于调整大小的控件，并绑定鼠标事件到该控件上
    sizer = tk.Label(window, bg="red")
    sizer.place(relx=1.0, rely=1.0, anchor="se")
    sizer.bind("<ButtonPress-1>", start_resize)
    sizer.bind("<ButtonRelease-1>", stop_resize)
    sizer.bind("<B1-Motion>", do_resize)

    def get_text():
        if not text_date.empty():
            local_E_text, local_C_text = text_date.get()
            english_text.config(text=local_E_text)
            chinese_text.config(text=local_C_text)

        # 使用after方法定期调用更新函数
        window.after(100, get_text)

    get_text()

    return window


if __name__ == "__main__":
    load_dotenv()

    # 创建并启动update_text线程
    thread = threading.Thread(target=update_text)
    thread.daemon = True  # 这允许线程在主程序退出时退出
    thread.start()

    # 创建并运行tkinter窗口
    window = create_window()
    window.mainloop()
