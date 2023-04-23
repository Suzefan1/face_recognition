import sensor,image,lcd  # import 相关库
import KPU as kpu
import time
from Maix import FPIOA,GPIO
from fpioa_manager import fm
task_fd = kpu.load(0x300000) # 从flash 0x300000 加载人脸检测模型
task_ld = kpu.load(0x400000) # 从flash 0x400000 加载人脸五点关键点检测模型
task_fe = kpu.load(0x500000) # 从flash 0x500000 加载人脸196维特征值模型

#从SD卡中加载模型
#task_fd = kpu.load("/sd/FD_face.smodel") # 加载人脸检测模型
#task_ld = kpu.load("/sd/KP_face.smodel") # 加载人脸五点关键点检测模型
#task_fe = kpu.load("/sd/FE_face.smodel") # 加载人脸196维特征值模型

clock = time.clock()  # 初始化系统时钟，计算帧率
key_pin=16 # 设置按键引脚 FPIO16
fpioa = FPIOA()
fpioa.set_function(key_pin,FPIOA.GPIO7)
key_gpio=GPIO(GPIO.GPIO7,GPIO.IN)
last_key_state=1
key_pressed=0 # 初始化按键引脚 分配GPIO7 到 FPIO16
def check_key(): # 按键检测函数，用于在循环中检测按键是否按下，下降沿有效
    global last_key_state
    global key_pressed
    val=key_gpio.value()
    if last_key_state == 1 and val == 0:
        key_pressed=1
    else:
        key_pressed=0
    last_key_state = val
#**************指示灯操作******************#
fm.register(13, fm.fpioa.GPIO0)
led_r=GPIO(GPIO.GPIO0, GPIO.OUT)
led_r.value(1)#低电平有效
fm.register(14, fm.fpioa.GPIO1)
led_b=GPIO(GPIO.GPIO1, GPIO.OUT)
led_b.value(1)#低电平有效
fm.register(0, fm.fpioa.GPIO3)
buzzer=GPIO(GPIO.GPIO3, GPIO.OUT)
buzzer.value(0)#高电平有效
lcd.init() # 初始化lcd
sensor.reset() #初始化sensor 摄像头
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_hmirror(1) #设置摄像头镜像
sensor.set_vflip(0)   #设置摄像头翻转
sensor.run(1) #使能摄像头
anchor = (1.889, 2.5245, 2.9465, 3.94056, 3.99987, 5.3658, 5.155437, 6.92275, 6.718375, 9.01025) #anchor for face detect 用于人脸检测的Anchor
dst_point = [(44,59),(84,59),(64,82),(47,105),(81,105)] #standard face key point position 标准正脸的5关键点坐标 分别为 左眼 右眼 鼻子 左嘴角 右嘴角
a = kpu.init_yolo2(task_fd, 0.5, 0.3, 5, anchor) #初始化人脸检测模型
img_lcd=image.Image() # 设置显示buf
img_face=image.Image(size=(128,128)) #设置 128 * 128 人脸图片buf
a=img_face.pix_to_ai() # 将图片转为kpu接受的格式
record_ftr=[] #空列表 用于存储当前196维特征
record_ftrs=[b'\xac\xc9\xe7@\xb0XX\x0b\xbeDu\xdf\x03\xdf+\xd5\xa6/\nu\xd2\xd1\xed\xdf\xc8\xb8\xc6\xe6!3\x15\x07\x80\x11\x05&\x0eq@\xc0\x80\r\xce\xf3\xd0%\x11L\x02\xff%\xd6\xf2\x15g\xd6.\xd0\xda_<\xed\xe3\x07\x06\xda\x16B\x1a\xed\x13\xfd\xae\xed\x15\xf7\xf6\xf2\x16\t/"&\xf9\xe7\x1d\xf3\xd5K_\x93\x03\xbe\xe7D\x80\xd7!\x85\xf6\xe5\xfa\xb4\xeb\x1e\xdfF\x88.:\xae&\xed%D\xad\x05\xda\xac.\xe5\xc1\r\x00\xcc\x7f>\x0b:\xc4\xc4\xc1\x07\x90a\x80C\xb8\r\xdf\x02\xad\x1e\xf9\x16\x13!\xaa\x80\xe7\x19\x026\xfbee\xdb\xfdS\xc0w\x01\xf6\xfe\x11\xdd\xf24X\xf62\xb8\xc8\xd3\xf5\t\x03\x0b\xe3\xe5NC\x1f/\xdf\\\xd1\x1b}\xa8\xd6\xc2\xbe\xa0\xe9\xd9',
             b'\x07\xf9."\xe7\x1a\xe38\xd6\xdd\x1a\xed\x11\x07\x1d\n\x80\xed\xe7\x80\xeb@\x19\x05\xd7\xf1\x01\x15\x01\xcaC0\x07\xd0/"/o>\xd3\xdd\xeb\xdf\xd5\xc8\x19\xfb2\x05Pi%\x99\x00\xba\xf1\x03\x11\xc8\x02\x1d\xff\xff\x12\x12\x0e\x03-\x1f\x15\xe1#\xf6\xdb\xfe\x02H\x02\x0e\xfa:\xfe\xe9\xeb\xf3\x1f\x13\xe3%\xfe\xc53\xc6D8o\xccK\xa1\x11\xca#\xed\xc8\xd9\xcd)\xd0\xfb\xd6\xb9\xe2\xe3lP\x97\x0b\xc9\xc20\xf1?\xc4\xe1\x02\x0f&/\x05\xa9\xfd#4\x85L\x06?\xdd\xa1\xfe\xf7\xdb\xed\x85\xc6\x11\x17\x93\x84\xd6\x80q\x0b\xf7\'&\x0b<B\xb8V\x06\xb1\x13\x0b\x1d\xc8\xf6\xcc\xf7W\xf5\xe5\x0b\x17\xeb\r\x1b\xd9*>\x1e!_\x0b\x0f\xbeG\xf3\xef\xef\xb6\xd2\xe3\xceH',
             b';\xe3\xff\x1a\n8\xfe\x19\x01\xd2\xf7\x16\x1d.\x19\xd9\xd3\xe6\x12&\xce\x19\xfa\x11\xfa\x05\x07\x01\xd2\xc2\x0b\xfd#\xf6\xe1\xfb@;+\xd7\x00\xe6\xff\x06\xd2\x1d\xf3\xff\xf6\t\x1b<\xcd\x11\xda\x11\xf5\xf9\xea\x1b;\xf9\x05\xd2\r\x15\x0704\xf9\xe3\x05!G\xd1\xfd\x1a\x0b\xf3\x13\xff\xf6\xfb\xf7\xe3\x15\r\xe3\x0e%\xeb\r\x1b"\x1d6\xeb3\xb1\xe7\xdb#\xfe\xe7>\xeb%*\x17\xc9\xbd\xf6\xf1\r#\xfd\xd3\xd3\x07\x12\x11\'\xbd\xf5\xdd\x0f0\x1b6\xde\x06/\t\xaa&*\xf7\xc5\xa5\xd1\x13\xf9%\x15\xf5\xd9\x02\xb2\xef\xf7G#\x05\xc6\x03\xe1.\xf5/\xc8)\x15\xa8"\t!\xba\x11\xd5\xe9!\xd2\xd0\x06\x03\x1f\x12\x1e\x03\xeb\x17\x0f*S":\xe3&0\xeb\x06\xff\xfe\xe1\xc6\xfb',
             b'\x13\x1a\xcc\x06\x9d\x19\xb4\xff\x06\xea\x07\xfb.\x1b\x1b\xd3\xd9\x02\x02\xf7\x00\x16\x1e\xf2#\xd5\xf3\xee&\xda\x0b\x00\xc2\x06\xfa\xe5\xfe\x1f[\xff\n\x1b.\x1b%3@\x0f\x17\xd7B\x17\xb8\xf1\xf1%\x02\xe5\xd0\x1a0\xf2\xfd\xe9\xae%\xda\x1e\xea\xd2\xeb\x12\xc6\r\x19X\xe7\xc1\xff"\x1b\xfe\xd6&\xb4%\xdeB!04\xf93\x03\xd2C\x15\xe1\xd5\x1f\x03\'4\x00\x0f\xf5\xd5\xe7\x0e\xeb&\xe9\xee\xea*\xe2\x1d\xdb\xae+\xe3\x07\x06\xbc\x00\x07\xf6.\x1e\x19P\x1aT\xc4\xc6\xe5\x19\xcd\xce%\xe7\xf2\xd5\x1b&\x05\x07\xa4\xf1\xe28\x11\x07\xf6\xd0-\xdb\x0f\xff\xbc;\x15\xc9*\x07\xee\xde\xfd\xee\x0b\x0f\xf7\xb4&\xdb\xd5\xc5"\xc2\xc82\x1e\xf9:\xf1\xef\x010\x19\xf7\xf6\xfb))\xce\x05',
             b'\xa0\xd7\xe7S\xc6\x138\x03\x17\xed*\xdd\xdb\x11\x00\xe7\x80\xef\xd3q\xac\xf6\xee\x02\x1e\xd2\xd5\x01\xce\t\xe5\x99\xf5D\xf9e7\x80\x19\xda\xb8/\xd9\x1a\xe9\x12\xe2)\x19\x01+!\xc8%\xbaH\xfb\xb2\xce7{\xb9D\xee\xaa\x11&h/\xc5\x0b\x13\xc8?\x02B*\xc2\x11\x1a@\xee\x0b\r\xd3\xe6T!K\xef\xd7\xcd;\r\x0fN\xed\xef\xcd\xdf\xff!\x17\x1d\xe3\x19\xfe\xf9\xc2"\xc2)\xc60"\x9c\xc2\x80\xb8m\xdf3\x15\xc9\x1f\xfdL\xf3O\xed\xcc\xc1\xd1\xa4\x07x?\xd1\xd1\x16\xbc\xe5\xcc\xf3\xd1+\xdd\xc5\xeb\xdbZ\x7f-\xde7\x12\xcc\xfe\x12\x8c+\xea\xfd\xd7"\x110O)\x072\xb8\xc2\x06\xf7\xe2\xdb\x1b\x0f\xdd\x06d!K\xf5\x1b\xd27\\\x80+\x1e\x1e\xe3\xf7\xca',
             ] #空列表 用于存储按键记录下人脸特征， 可以将特征以txt等文件形式保存到sd卡后，读取到此列表，即可实现人脸断电存储。
names = ['zhenglian', 'youlian', 'zuolian', 'wuyanjing', 'Mr.5', 'Mr.6', 'Mr.7', 'Mr.8', 'Mr.9' , 'Mr.10'] # 人名标签，与上面列表特征值一一对应。
while(1): # 主循环
    check_key() #按键检测
    img = sensor.snapshot() #从摄像头获取一张图片
    clock.tick() #记录时刻，用于计算帧率
    code = kpu.run_yolo2(task_fd, img) # 运行人脸检测模型，获取人脸坐标位置
    led_b.value(1)#低电平有效
    led_r.value(1)#低电平有效
    buzzer.value(0)#高电平有效
    if code: # 如果检测到人脸
        for i in code: # 迭代坐标框
            # Cut face and resize to 128x128
            a = img.draw_rectangle(i.rect()) # 在屏幕显示人脸方框
            face_cut=img.cut(i.x(),i.y(),i.w(),i.h()) # 裁剪人脸部分图片到 face_cut
            face_cut_128=face_cut.resize(128,128) # 将裁出的人脸图片 缩放到128 * 128像素
            a=face_cut_128.pix_to_ai() # 将猜出图片转换为kpu接受的格式
            #a = img.draw_image(face_cut_128, (0,0))
            # Landmark for face 5 points
            fmap = kpu.forward(task_ld, face_cut_128) # 运行人脸5点关键点检测模型
            plist=fmap[:] # 获取关键点预测结果
            le=(i.x()+int(plist[0]*i.w() - 10), i.y()+int(plist[1]*i.h())) # 计算左眼位置， 这里在w方向-10 用来补偿模型转换带来的精度损失
            re=(i.x()+int(plist[2]*i.w()), i.y()+int(plist[3]*i.h())) # 计算右眼位置
            nose=(i.x()+int(plist[4]*i.w()), i.y()+int(plist[5]*i.h())) #计算鼻子位置
            lm=(i.x()+int(plist[6]*i.w()), i.y()+int(plist[7]*i.h())) #计算左嘴角位置
            rm=(i.x()+int(plist[8]*i.w()), i.y()+int(plist[9]*i.h())) #右嘴角位置
            a = img.draw_circle(le[0], le[1], 4)
            a = img.draw_circle(re[0], re[1], 4)
            a = img.draw_circle(nose[0], nose[1], 4)
            a = img.draw_circle(lm[0], lm[1], 4)
            a = img.draw_circle(rm[0], rm[1], 4) # 在相应位置处画小圆圈
            # align face to standard position
            src_point = [le, re, nose, lm, rm] # 图片中 5 坐标的位置
            T=image.get_affine_transform(src_point, dst_point) # 根据获得的5点坐标与标准正脸坐标获取仿射变换矩阵
            a=image.warp_affine_ai(img, img_face, T) #对原始图片人脸图片进行仿射变换，变换为正脸图像
            a=img_face.ai_to_pix() # 将正脸图像转为kpu格式
            #a = img.draw_image(img_face, (128,0))
            del(face_cut_128) # 释放裁剪人脸部分图片
            # calculate face feature vector
            fmap = kpu.forward(task_fe, img_face) # 计算正脸图片的196维特征值
            feature=kpu.face_encode(fmap[:]) #获取计算结果
            reg_flag = False
            scores = [] # 存储特征比对分数
            for j in range(len(record_ftrs)): #迭代已存特征值
                score = kpu.face_compare(record_ftrs[j], feature) #计算当前人脸特征值与已存特征值的分数
                scores.append(score) #添加分数总表
            max_score = 0
            index = 0
            for k in range(len(scores)): #迭代所有比对分数，找到最大分数和索引值
                if max_score < scores[k]:
                    max_score = scores[k]
                    index = k
            if max_score > 74: # 如果最大分数大于75， 可以被认定为同一个人
                a = img.draw_string(i.x(),i.y(), ("%s :%2.1f" % (names[index], max_score)), color=(0,255,0),scale=2) # 显示人名 与 分数
                led_b.value(0)#低电平有效
                led_r.value(1)#低电平有效
                buzzer.value(0)#高电平有效
            else:
                a = img.draw_string(i.x(),i.y(), ("X :%2.1f" % (max_score)), color=(255,0,0),scale=2) #显示未知 与 分数
                led_r.value(0)#低电平有效
                led_b.value(1)#低电平有效
                buzzer.value(1)#高电平有效
            if key_pressed == 1: #如果检测到按键
                key_pressed = 0 #重置按键状态
                record_ftr = feature
                print("人脸特征值：",feature)
                record_ftrs.append(record_ftr) #将当前特征添加到已知特征列表
            break
    else: # 如果未检测到人脸
        led_b.value(1)#低电平有效
        led_r.value(1)#低电平有效
    fps =clock.fps() #计算帧率
    print("%2.1f fps"%fps) #打印帧率
    a = lcd.display(img) #刷屏显示
    #kpu.memtest()

#a = kpu.deinit(task_fe)
#a = kpu.deinit(task_ld)
#a = kpu.deinit(task_fd)
close
