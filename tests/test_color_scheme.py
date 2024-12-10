import sys, os, cv2
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from image_processing.color_scheme import ColorSchemeCreator


color_zone_img = cv2.imread("C:\Victor\DrawByNumbers\TestOutput\MickeySuccess_1.png")
color_zone_img = cv2.imread("C:\Victor\DrawByNumbers\TestOutput\\NadineSuccess_1.png")
color_zone_img = cv2.cvtColor(color_zone_img, cv2.COLOR_BGR2RGB)