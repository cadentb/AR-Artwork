# AR-Artwork

This project was heavily jump started using Logan Engstrom's fast-style-transfer repository.

ARArtworkApp is the application side of the project.
fast-style-transfer is the style transfer algorithm side of the project.

Within each of these directories, you'll find their own READMEs.

Google Colab (for running and testing the algorithm) - https://colab.research.google.com/drive/1ATiIfH8aMpWvgFvr-XGT6tt9NUbK5mrk

## Description

The augmented reality system that our team is working on will take existing AI Art technology such as DALL-E and apply it to a real-time AR technology. While plenty of AI art models take in text prompts or images to convert into a final piece, the project we are working on will take in live footage and stylize the footage in real time, meaning that much of the stylization process will have to be revised to be more efficient compared to existing models. This will lead to new possibilities in AR/AI technology, allowing for new forms of education and entertainment.


In general, this app can serve as the base for other artists and developers to build off from. Combining the recent technologies of AR/AI with art will create numerous possibilities for developers working on AR applications and games. As AR technology becomes more popular among the general population, features such as facial, object, and text recognition could become very important to serve as accessibility tools.


Our app will first collect the image input from the device, secondly our system will recognize the objects, people and fonts in the image, and finally the system will convert the image into an artistic style image according to the given style. As mentioned above, the combination of AR and games brings not only a simple superposition of technology, but also an upgrade of gameplay and user experience. AR is very suitable for concepts such as dreams, portals, and travel in games. Through ingenious art methods, the virtual and reality are naturally integrated to present different environments. At the same time, AR can also provide more exotic forms of visual expression


# Anaconda Outline




## Image Transfer




### Make the Environment




1. Open the Anaconda Prompt.
2. Within the Anaconda Prompt, type the following commands:
```
$ conda create -n tf-gpu tensorflow-gpu=2.1.0  
$ conda activate tf-gpu 
$ conda install jupyterlab
```
### Clone the Repository




Clone the GitHub repository by running the following command:




```
$ git clone https://github.com/lengstrom/fast-style-transfer.git
```
Make sure you’re in a directory that you’re comfortable having a cloned github in




### Download pre built models
1. Download the entire file from the following link: https://drive.google.com/drive/folders/0B9jhaT37ydSyRk9UX0wwX3BpMzQ?resourcekey=0-Z9LcNHC-BTB4feKwm4loXw.
2. Put the downloaded files into the fast-style-transfer directory.




### Run the command
To apply style transfer to an image, run the following command:
```
$ python evaluate.py --checkpoint models/la_muse.ckpt --in-path path/to/image/img.JPG --out-path ./path/to/examples/results.JPG
```


For example:
```
$ python evaluate.py --checkpoint models/la_muse.ckpt --in-path cat.JPG --out-path ./examples/results.JPG
```


## Video Transfer




### Setup




1. Download a video and place it in your working directory.
2. Run the following commands to circumvent broken pipe errors:




```
$ pip install imageio-ffmpeg
$ pip install --upgrade imageio-ffmpeg==0.2.0
$ pip show imageio-ffmpeg
```
Make sure that it shows something like this:
```
Name: imageio-ffmpeg
Version: 0.2.0
Summary: FFMPEG wrapper for Python
Home-page: https://github.com/imageio/imageio-ffmpeg
Author: imageio contributors
Author-email: almar.klein@gmail.com
License: (new) BSD
Location: c:\users\lg681\anaconda3\envs\tf-gpu\lib\site-packages
Requires:
Required-by: moviepy
```


### Running the scripts


```
python transform_video.py --in-path cat2.mp4 --checkpoint models/la_muse.ckpt --out-path ./examples/video_la.MP4
```


## Android App:


Download android studio
Clone repository containing android app code
Bring cloned repository into android studio
Create a virtual device (pixel 6 API 30)
Startup virtual device
Run app


## Design Prototypes


https://www.figma.com/proto/oPbBfMbDCuWo1FNSUT1g3H/AR-Art?node-id=3-260&scaling=scale-down&page-id=0%3A1&starting-point-node-id=6%3A66 






# Functionality Outline


## Group Introduction and Partner List


Lucas Garcia – AI Team​
garcluca@oregonstate.edu​


Caden Burke – AI Team​
burkecad@oregonstate.edu​


Brandon Christensen – UI/UX Design Team​
Chribran@oregonstate.edu​


Sree Gajula – UI/UX Design Team​
gajulas@oregonstate.edu​


Chenyu Song – AI Team​
songchen@oregonstate.edu


## Overview and Context


This project aims to enable real-time style transfer on videos and images, allowing the creation of art anywhere.


## System Walkthrough and Diagrams


### System Outline


1. Takes input of a video or image.
2. Real-time video input processing (in the workload of the Spring term).
3. Uses pre-made models to apply styles to the input.
4. Utilizes two different models for image and video processing.
5. Background processing files include image processing and AI library files.
6. Each processing file performs its specific task.
7. Utilizes GPU to accelerate the processing speed.


Related Links:
Colab: https://colab.research.google.com/drive/1ATiIfH8aMpWvgFvr-XGT6tt9NUbK5mrk#scrollTo=2LnjxKWRtVim
Github Code:
https://github.com/SS-CC-YY/fast-style-transfer (all the transferred images and video can be found in ./examples)
Application Code:
https://github.com/bdchris18/ArArtworkApp2


## Constraints


Constraint 1: TIME:
At present, our team has planned the weekly online meeting time, which is 9:20 on Tuesday morning and TA meeting, and 4:00 pm on Tuesday with project partner Lucas. And we have decided to hold offline meetings and work in Kelly's study room or the library's classroom every weekend. In addition, on the project schedule, after our online meeting with the project partner, our main task in the near future is to determine everyone's team role. The next tasks are the construction of the neural network model and the acquisition and processing of the data set. The next task is the training of the neural network model for image recognition. The last task of this semester is to make an AI model for artistic image style transformation. Finally, since our group members have courses this term, we need to spend time on the courses every week, but it does not affect the progress of the project at first glance. In addition, project partner Lucas will have a slightly serious time limit due to work and courses.


Constraint 2: RESOURCES:
The resources we will need won’t impact us much financially. To learn the concepts, books may need to be bought. The majority of our resources however will be free. Specifically, the resources available to us include open-source code libraries such as the Python library TensorFlow for machine learning capabilities. We will all have access to phones and computers for developing the app.


Constraint 3: SCOPE:
The scope of this project will increase in complexity as we approach the real-time augmentation section of the project. The scope mainly consists of using TensorFlow and machine learning tools to achieve style transfer. To construct the neural network model, we will need to process a data set. Once that is accomplished, we will need to apply that to video and implement real-time processing. Some constraints that we may face include dividing the work while understanding what everyone else is doing to bring those parts together.


## Future Work


- Model Optimization. Since we are still using vgg-16 network to train the model. Although the result is good, it requires a lot of computing resources. Therefore, considering the use of latest MobileNet can greatly reduce the use of computing units and enable mobile devices to run the model independently​
- Real-time image style-transfer. The current model cannot achieve real-time image style transfer, so consider combing the diffusion model or some other techniques to achieve real-time transfer
- The mobile application to port the functionality of the project to mobile devices
- The machine learning environment setup in GCP virtual machine.
- GCP virtual machine api setup and link to the application
- Concern: At present, my vm has applied for an Nvidia Tesla K80 graphics card and a 200G hard disk. This configuration will cost about 250 US dollars a month, while the free bonus of GCP is only 300. Therefore, I can't keep the cloud platform running until the end of June which means I can only turn it on for a short time during testing or demonstration.
