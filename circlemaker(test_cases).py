import argparse
import random
import numpy as np
import cv2 
import time
import colorsys

from PIL import Image, ImageDraw

CANVAS_SIZE = 400




def draw_image(d, hue, output_path):
    image = Image.new('RGB', (CANVAS_SIZE, CANVAS_SIZE), 'white')
    draw = ImageDraw.Draw(image)
    center = CANVAS_SIZE / 2
    draw.ellipse(
        (center - d / 2, center - d / 2, center + d / 2, center + d / 2),
        fill=f'hsv({hue}, 100%, 100%)'
    )
    draw.rectangle(
        (0, 0, 399, 399),
        outline=f'hsv({random.randint(0, 360)}, 100%, 100%)'
    )
    image.save(output_path, format='png')


def float_in_range(vmin, vmax):
    def _float_in_range(number):
        try:
            f = float(number)
        except ValueError:
            raise argparse.ArgumentTypeError(
                'Argument must be a float type number')

        if not (vmin <= f <= vmax):
            raise argparse.ArgumentTypeError(
                f'Argument must be within {vmin} <= arg <= {vmax}')

        return f

    return _float_in_range

def get_circle_dia():
    img = cv2.imread('test.png')
    output = img.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    circles = cv2.HoughCircles(image=gray,
                            method=cv2.HOUGH_GRADIENT, 
                            dp=1.1, 
                            param1=100,
                            param2=30, 
                            minDist=30, 
                            minRadius=1, 
                            maxRadius=399)
    time.sleep(3)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            r_mm = round(r/53.3, 2)
            cv2.circle(output, (x,y), r, (0, 0, 0), 1)
            cv2.rectangle(output, (x-2, y-2), (x+2, y+2), (0,255,0), -1)
            cv2.putText(output, str(r*2), 
                        (x-15, y-5), 
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                        0.7, (0, 0, 0), 1)
            print ('The Obtained Radius of circle is around: ' , str(r))
        # cv2.imshow("output", np.hstack([output]))
        # cv2.waitKey(0)
        return (r*2)

def get_circle_hue():
    img = cv2.imread("test.png")
    hh, ww = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    min_dist = int(ww/10)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=min_dist, param1=150, param2=20, minRadius=0, maxRadius=0)
    img_circle = img.copy()
    mask = np.zeros_like(gray)
    for circle in circles[0]:
        (x,y,r) = circle
        x = int(x)
        y = int(y)
        r = int(r)
        cv2.circle(img_circle, (x, y), r, (0, 0, 255), 2)
        cv2.circle(mask, (x, y), r, 255, -1)
    ave_color = cv2.mean(img, mask=mask)[:3]
    print('The value of Red color is: ',ave_color[2])
    print('The value of Green color is: ',ave_color[1])
    print('The value of Blue color is: ',ave_color[0])
    return calculate_hue_from_rgb(ave_color[2],ave_color[1],ave_color[0])

def calculate_hue_from_rgb (red,green,blue):
    min_value = min(min(red,green),blue)
    max_value = max(max(red,green),blue)
    if(min_value == max_value):
        return 0
    hue =0
    if(max_value==red):
        hue = (green-blue) / (max_value-min_value)
    elif (max_value==green):
        hue = 2 + (blue-red) / (max_value-min_value)
    else:
        hue = 4 + (red-green) / (max_value-min_value)
    hue = hue * 60
    if (hue < 0 ): 
        hue = hue+360
    return round (hue)

def verifying_diameter_testcase(obtained_diameter,d):
    print('')
    print('The obtained Diameter of circle is :',obtained_diameter)
    if(obtained_diameter == d):
        print('The value of obtained diameter is same as provided diameter')   
    elif(obtained_diameter-1 ==d or obtained_diameter+1 ==d):
         print('The value of obtained diameter is almost equlavent to the provided diameter with difference of: ',obtained_diameter-d)   
    elif(obtained_diameter-2 == d or obtained_diameter+2 ==d):
         print('The value of obtained diameter is almost equlavent to the provided diameter with difference of: ',obtained_diameter-d)
    elif(obtained_diameter-3 == d or obtained_diameter+3 ==d):
         print('The value of obtained diameter is almost equlavent to the provided diameter with difference of: ',obtained_diameter-d)
    else:
        print('The obtained diameter is different from given diameter')
        print('Testcase 1 Failed!')
        return
    print('Testcase 1 Passed!')

def verifying_hue_testcase(obtained_hue,hue):
    print('')
    print('The obtained Hue (color) of circle is :',obtained_hue)
    if(obtained_hue == hue):
        print('The value of obtained Hue (color) is same as provided Hue (color)')   
    elif(obtained_hue-1 ==hue or obtained_hue+1 ==hue):
         print('The value of obtained Hue (color) is almost equlavent to the provided Hue (color) with difference of: ',obtained_hue-hue)   
    else:
        print('The obtained Hue (color) is different from given Hue (color)')
        print('Testcase 1 Failed!')
        return
    print('Testcase 1 Passed!')          
   


if __name__ == '__main__':
    obtained_diameter = 0
    obtained_hue = 0
    parser = argparse.ArgumentParser(description='Draw a circle')
    parser.add_argument(
        '-d', type=float_in_range(0, 399), help='diameter of a circle')
    parser.add_argument(
        '-hue', type=float_in_range(0, 360),  help='hue of the HSV color of a circle')
    parser.add_argument(
        '-path', type=str, help='output path of generated image')

    args = parser.parse_args()

    draw_image(args.d, args.hue, args.path)

    obtained_diameter =  get_circle_dia()
    verifying_diameter_testcase(obtained_diameter,args.d)
    
    obtained_hue = get_circle_hue()
    verifying_hue_testcase(obtained_hue,args.hue)
