import time
import pyautogui as pa

i=0
while(i<720):
    pos = pa.position()
    print(pos)
    pa.click(x=576, y=1139)

    # BSおす
    # pa.press("backspace")
    # A入力
    #1秒まつ
    time.sleep(10.0)
    i += 1

# 576.1139