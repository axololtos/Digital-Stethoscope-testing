# Digital-Stethoscope-AI

## Introduction
We all know COVID19 is extremely contagious by now, and shortage in healthcare provider is already happening in first world, this would only further escalate in 3rd world nations who do not have much resources. By being part of UN Sustainable Development Goal 3, we want to innovate in this area and able to be able to help everyone in the world. We are team MixPose, a live streaming yoga class company, and we want to build something that can help with COVID19.

### Telemedicine
Telemedicine have been maturing for the past decade, and many have expected after COVID19, this will be pushed towards mainstream. This also allows doctors working remotely to treat people in developing nations, that's in dire need of healthcare resources right now. Every doctor wears a stethoscope, they can use that to diagnose symptoms for the patients.

The digital stethoscopes in the market is generally not affordable at individual level, especially in Developing nations and struggling communities, they are a great tool for the doctor, but in telemedicine, we need the tool to be able to diagnose patients remotely. In case of no doctors available for Developing Nations, we need AI that's affordable to do that diagnosis.

### Affordable Digital Stethoscopes with AI
So we want to build an open sourced digital stethoscopes that's under Rs 500.00 and can easily be given to everyone in the world, so that we can empower doctors via telemedicine to treat more people.

![app working](https://github.com/user-attachments/assets/6887da10-bd37-495d-8d75-129e22e93173)

In this project, we will also train an AI to use sound classification so that patients can do a self checkup anytime, and when the AI detects possible symptoms, the audio data is recorded for the patients to send it to a doctor for a further checkup.

## Step 1: Building the Stethoscope and Recording Sound
To make this, we only need a normal Stethoscope and a tiny microphone, the Stethoscope can be purchased Rs. 250 retail, the Microphone for another Rs. 300 retail. Although this adds up to Rs. 980 retail, we do not need most of the parts. Also, if we were to order these whole sale, we can easily cut the entire bill of material down to about Rs. 500.
Now that we have the materials, we can cut the scope itself and attach directly to the small microphone. This would allow sound goes directly through microphone itself coming from the plate. Full building instruction can be seen below. This step is enough for those simply want to make an affordable stethoscope and record your own heart rate or breath.

## Step 2: Recording from Stethoscope
Now that we have built a digital stethoscope from off the shelf material, we can use that to record our heart as well as our respiratory system. The recording can be done via our computer or our phone, making it much more accessible around the world
![screen_shot_2020-04-29_at_10_03_22_pm_q94SDcnchK](https://github.com/user-attachments/assets/9d0b7dab-b23d-44b3-82eb-cb9df0f7fe03)

## Step 3: Training Tensorflow Sound Classification AI
Users now can record their own internal sound from Stethoscope, we will tensorflow sound classification to determine whether it's COVID related such as Pneumonia, or other disease which you can wait for.
![gc](https://github.com/user-attachments/assets/77bab32b-3eb0-467b-93b6-4a6849ab7f28)


As for data, we can retrieve data from different sources. For this project we will be able to classify 5 different types of sound. To make the project simple, we will focus on Trachea, except for heart rate, which we will be using stethoscope on the heart.

- Healthy
- Chronic Obstructive Pulmonary Disease
- Pneumonia (often associated symptom with COVID19)
- Upper Respiratory Tract Infection

We have to change train.py to following format

We can load the files into training_data folder and, change the training file to following

```bash
$ python3 train.py
```
# Android App for Audio Recognition

This repository provides instructions to build an Android app for audio recognition using TensorFlow. Below are the steps to convert a frozen graph to a TFLite model and train it.

## Step 4: Build an Android App

To run the app on Android, the easiest approach is to use a TFLite file, as `.pb` files are no longer supported in TensorFlow 2.0 or above. Use the following code to convert a frozen graph to TFLite:

```python
import tensorflow as tf

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    "./froze_graph.pb",
    input_arrays=['decoded_sample_data', 'decoded_sample_data:1'],
    output_arrays=['labels_softmax']
)
converter.allow_custom_ops = True
tflite_model = converter.convert()
open("output.tflite", "wb").write(tflite_model)
```
Training on the Cloud
You can train the model on the cloud. For this, you can easily spin up a GPU server using TensorFlow 2.2 on Google Cloud Platform.

Audio Recognition
This project follows TensorFlow's audio recognition tutorial. For a complete understanding, refer to:

https://www.tensorflow.org/tutorials/sequences/audio_recognition
