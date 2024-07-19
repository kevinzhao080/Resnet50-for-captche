import os
import shutil
import string
import random
from PIL import Image, ImageDraw, ImageFont

colors1 = ['red', 'blue', 'green', 'purple', 'orange','black']
colors2 = ['black']
font1 = "arial.ttf"
font2 = r'font_room\Arial Black.ttf'
font3 = r'font_room\DFPShaoNvW5-GB.ttf'
font4 = r'font_room\huawennishu.TTF'
font5 = r'font_room\PingFang SC Regular.ttf'
fontlist1 = [font1,font2,font3,font4,font5]
fontlist2 = [font1,font3,font5]
fontlist3 = [font2,font4]
fontlist4 = [font1]
fontlist5 = [font2]
fontlist6 = [font3]
fontlist7 = [font4]
fontlist8 = [font5]
# 验证码生成函数
def generate_captcha_image(captcha_text):
    if random.random() <0.3:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    else :
        color = 'white'
    image = Image.new('RGB', (320, 120),color)
    draw = ImageDraw.Draw(image)

    # 绘制验证码文本
    randomnum = random.random()
    if  randomnum< 0.7:
        color = random.choice(colors1)
    else:
        color = random.choice(colors2)

    for i in range(len(captcha_text)):
        randomnum2 = random.random()
        if randomnum2<=0.2:
            font= random.choice(fontlist1)
        elif 0.2<randomnum2<=0.3:
            font = random.choice(fontlist2)
        elif 0.3<randomnum2<=0.4:
            font = random.choice(fontlist3)
        elif 0.4<randomnum2<=0.5:
            font = random.choice(fontlist4)
        elif 0.5<randomnum2<=0.6:
            font = random.choice(fontlist5)
        elif 0.6<randomnum2<=0.7:
            font = random.choice(fontlist6)
        elif 0.7<randomnum2<=0.8:
            font = random.choice(fontlist7)
        else:
            font = random.choice(fontlist8)
        font = ImageFont.truetype(font, random.randint(50,60))
        draw.text((i*random.randint(40, 50) + 120, random.randint(29, 39)), captcha_text[i], font=font, fill=color)

    # 添加干扰线
    if random.random() < 0.3:
        for _ in range(random.randint(0,7)):
            x1 = random.randint(0, 320)
            y1 = random.randint(0, 120)
            x2 = random.randint(0, 320)
            y2 = random.randint(0, 120)
            draw.line(((x1, y1), (x2, y2)), fill=random.choice(colors1), width=random.randint(1, 3))

    # 扭曲图像
    if random.random() < 0.3:
        image = image.transform((320, 120), Image.AFFINE, (1, random.uniform(-0.2,0.2), 0, -0.1, 1, 0))

    # 添加背景噪声
    if random.random() < 0.3:
        for _ in range(random.randint(0, 3000)):
            x = random.randint(0, 320)
            y = random.randint(0, 120)
            draw.point((x, y), fill='black')

    # # 计算裁剪区域的坐标
    # if random.random()<0.2:
    #     left = 320 / random.uniform(2.9,5)
    #     top = 120 / random.uniform(2.9,5)
    #     right = 320/3*random.uniform(1.9,3.2)
    #     bottom = 120 / 3*random.uniform(1.9,3.3)
    # else:
    #     left = 320 / 3
    #     top = 120 / 3
    #     right = 320/3*2
    #     bottom = 120 / 3*2
    
    # # 裁剪图片
    # image = image.crop((left, top, right, bottom))

    # 随机选择目标大小
    if random.random() <0.5:
        num = random.uniform(0.8, 3.0)
        w,h = image.size
        target_width = w*num
        target_height = h*num
    else:
        target_width = 160
        target_height = 60
    # 调整图像大小
    image = image.resize((int(target_width), int(target_height)))
    return image

#清空文件夹
def clear_folder(folder_path):  
    for filename in os.listdir(folder_path):  
        file_path = os.path.join(folder_path, filename)  
        try:  
            if os.path.isfile(file_path) or os.path.islink(file_path):  
                os.unlink(file_path)  
            elif os.path.isdir(file_path):  
                shutil.rmtree(file_path)  
        except Exception as e:  
            print(f'Failed to delete {file_path}. Reason: {e}') 

if __name__ == '__main__':
    path = '../test'
    if not os.path.exists(path):
        os.makedirs(path)
    clear_folder(path) #清空文件夹
    length = 4
    num = 1000
    NUMBER = ['0','1','2','3','4','5','6','7','8','9']
    UPPER = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']
    # CHARSET = NUMBER+UPPER
    characters = NUMBER+UPPER
    for i in range(num):
        captcha_text = ''.join(random.choices(characters, k=length))
        image = generate_captcha_image(captcha_text)
        image.save(f"./{path}/{i}_{captcha_text}.png")
        print(f"Generated {i+1}: {captcha_text}")
