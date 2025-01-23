# Animaldetectionandprevention



India, Assam where I am from people are highly dependent on farming 

Around 69% of people are dependant on farming in Assam and elephants re a huge problem So,  in my 9th grade i decided to build a device that is centrally controlled with many individual cameras that can be placed around a farm to detect and deter elephants or other animals that may pose a threat.

for this prject i used my laptop as the main processing unit and than a esp32 wroom da module to  controll the other devices.

Things used:-

1. Esp 32 WROOM DA-MODULE - I used this has it is low power but can still outperform an arduino, and also incase any one wanted to make a standalone option without a laptop it would be easier to switch.

2. Led Lights - Led lights were used as a proof of concept that showed bigger flood lights or other strobe lights can be controoled with the esp32 using a external power source fot the light.

3. Buzzer - Buzzers were used in order to proof that more powerfull systems can be used. Similarly to the lights the better alarm system would need external power has the esp 32 is only at best is able to provide 5v.


SOFTWARE

I am using a object_detection_COCO it is reprository in github it is a light weight object detection library. 

It uses mobilenet_iter_73000.caffemodel. 

It has been chosen because it is light weight and esy to work with.
