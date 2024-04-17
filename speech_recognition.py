import datetime
import time
import pyaudio
import numpy as np
import pyautogui
import pyperclip
import torch
from faster_whisper import WhisperModel

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class CcServer:
    def __init__(self, threshold=0.01, duration=1, sample_rate=44100):
        self.wisperModel = WhisperModel(model_size_or_path="./whisper_large-v3\models--Systran--faster-whisper-large-v3\snapshots\edaa852ec7e145841d8ffdb056a99866b5f0a478", device=device, compute_type="float16")
        # 设置声音强度的阈值
        self.threshold = threshold
        # 设置录音时长为1秒
        self.duration = duration
        # 设置采样率为44100 Hz
        self.sample_rate = sample_rate
        self.pAudio = pyaudio.PyAudio()
        self.output_stream = self.pAudio.open(format=pyaudio.paInt16,
                                              channels=1,
                                              rate=16000,
                                              output=True)
        self.chunk = 2048
        format = pyaudio.paInt16
        channels = 1
        rate = 16000
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=format,
                                  channels=channels,
                                  rate=rate,
                                  input=True,
                                  frames_per_buffer=self.chunk)

    def start(self):
        self.main()

    def listen(self):
        """
        录音方法
        """
        mindb = 3500
        delay_time = 0.7
        frames = []
        silence_duration = 50  # 无声音超过10秒结束录音
        start_silence_time = time.time()  # 检测到无声音的时间戳
        flag = False  # 开始录音的标志
        stat = True  # 是否继续录音
        stat2 = False  # 声音是否变小的标志
        start_time = time.time()  # 开始录音的时间戳
        temp_time = 0  # 上次声音变小的时间戳
        try:
            print("开始检测声音...")
            while stat:
                data = self.stream.read(self.chunk)
                frames.append(data)
                audio_data = np.frombuffer(data, dtype=np.int16)
                temp = np.max(audio_data)
                # 检测到足够的声音强度，开始录音
                if time.time()-start_silence_time >= silence_duration:
                    stat = False
                    self.no_voice = True
                if temp > mindb and not flag:
                    flag = True
                    print(temp)
                    print("检测到声音，开始录音")
                    temp_time = time.time()
                # 如果已经开始录音
                if flag:
                    # 检测到声音变小
                    if temp < mindb and not stat2:
                        print(temp)
                        stat2 = True
                        temp_time = time.time()
                        print("声音变小，记录时间点")
                    # 检测到声音变大
                    if temp > mindb:
                        stat2 = False
                        temp_time = time.time()
                    # 如果声音小的时间超过了设定的延迟时间，结束录音
                    if time.time() - temp_time > delay_time and stat2:
                        stat = False
                        print("长时间声音小，结束录音")
        except Exception as e:
            print(f"录音过程中发生错误: {e}")
        finally:
            print("录音结束")
            start_time = datetime.datetime.now()
            print("语音识别模型开始处理", start_time)
            data = bytes(bytearray().join(frames))
            resp = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            segments, info = self.wisperModel.transcribe(audio=resp, beam_size=5,vad_filter=True,initial_prompt="简体字")
            texts = [segment.text for segment in segments]
            full_text = "".join(texts)
            end_time = datetime.datetime.now()
            print("语音处理完成", end_time)
            run_time = end_time - start_time
            print(f"语音处理时间: {run_time}")
            print("语音结果：", full_text)
            return full_text

    def main(self):
        # input("请按回车启动")
        end = False
        while not end:
            text = self.listen()
            pyperclip.copy(text)
            pyautogui.hotkey('ctrl', 'v')
            # a = input("输入q结束，按回车继续。")
            # if a == "q":
            #     end = True
if __name__ == '__main__':
    cc = CcServer()
    cc.start()