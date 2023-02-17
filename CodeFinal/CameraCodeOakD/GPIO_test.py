''' This code is just for the testing of reading/writing to Pi GPIO/Serial '''

# libraries for GPIO pin writing (either will work, will be up to preference)
import RPi.GPIO as GPIO
# other libs
import time


# address GPIOS via their GPIO number
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT)
while True:
    GPIO.output(11, GPIO.LOW)
    print(GPIO.input(11))
    # print("On")
    # time.sleep(0.5)
    # GPIO.output(11, GPIO.LOW)
    
    # #print("Off")
    # time.sleep(0.5)









# person
# bicycle
# car
# motorcycle
# airplane
# bus
# train
# truck
# boat
# traffic light
# fire hydrant
# street sign
# stop sign
# parking meter
# bench
# bird
# cat
# dog
# horse
# sheep
# cow
# elephant
# bear
# zebra
# giraffe
# hat
# backpack
# umbrella
# shoe
# eye glasses
# handbag
# tie
# suitcase
# frisbee
# skis
# snowboard
# sports
# kite
# baseball bat
# baseball glove
# skateboard	skateboard	skateboard	sports
# surfboard	surfboard	surfboard	sports
# tennis racket	tennis racket	tennis racket	sports
# bottle	bottle	bottle	kitchen
# plate	-	-	kitchen
# wine glass	wine glass	wine glass	kitchen
# cup	cup	cup	kitchen
# fork	fork	fork	kitchen
# knife	knife	knife	kitchen
# spoon	spoon	spoon	kitchen
# bowl	bowl	bowl	kitchen
# banana	banana	banana	food
# apple	apple	apple	food
# sandwich	sandwich	sandwich	food
# orange	orange	orange	food
# broccoli	broccoli	broccoli	food
# carrot	carrot	carrot	food
# hot dog	hot dog	hot dog	food
# pizza	pizza	pizza
# donut	donut	donut
# cake	cake	cake
# chair	chair	chair
# couch	couch	couch
# potted plant	potted	potted plant	furniture
# bed	bed	bed
# mirror	-	-
# dining table	dining	dining table	furniture
# window	-	-
# desk	-	-
# toilet	toilet	toilet
# door	-	-
# tv	tv	tv
# laptop	laptop	laptop
# mouse	mouse	mouse
# remote	remote	remote
# keyboard	keyboard	keyboard
# cell phone	cell	cell phone	electronic
# microwave	microwave	microwave
# oven
# toaster
# sink
# refrigerator
# blender
# book
# clock
# vase
# scissors
# teddy bear
# hair drier
# toothbrush
# hair brush