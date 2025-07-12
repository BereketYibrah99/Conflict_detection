# Conflict Detection

## Introduction

- Conflict detection is a computer vision project that utilizes key points extracted from a key detection model to identify fights, conflicts, and violent behavior.  
- By analyzing video feeds from surveillance cameras, the system can recognize patterns indicative of conflict.  
- This technology assists security personnel and camera operators in real-time monitoring, enhancing their ability to respond quickly to potential threats and maintain safety in various environments, such as public spaces.  
- Security camera operators often struggle with multiple screens, leading to fatigue and difficulty in focusing. This can affect their ability to detect real-time violence, allowing criminals to escape and putting victims at risk.  
- This project will notify security camera operators when such kind of problem happens, by making them focus on the screen where the problem occurs, making them respond quickly.  

---

## In What Way Will It Guide Security Camera Operators

---

## Objective

- To assist security camera operators to make there work easy and efficient.  
- To reduce the negative consequence that happens after the conflict or violence.  
- To save the victim as quick as possible.  
- To reduce the escaping chance of the criminals.  
- To reduce violence rate in the country.  
- To secure peoples safety.  

---

## Scope

- The scope of our project involves tracking and detecting the pattern of conflict, violence, fighting and other normal activities.  
- To predict whether that activity is violent or not to safeguard the safety of the civilians by notifying security camera operators in real-time.  

---

## Limitations

- Our project won't be able to detect violence that include remote weapons like guns.  
- Some actions that are related to fighting are sometimes confuses the model.  
- It don't have the ability to work on its own it needs a person which monitor its activity.  
- It doesn't differentiate between people who are realy fighting or just messing with each other.  

---

## The Algorithm We Used Behind Our Project

- We selected the **YOLOv8n-pose** model for pose estimation because it effectively detects pose landmarks and works well on CPU, allowing us to measure distances between key points for analyzing the patterns of fighting.  
- Our algorithm primarily focuses on measuring the distances between pairs of key points for each person’s body. If anybody key points are not detected, the algorithm estimates their locations based on the visible data.  
- Each time a person appears on the screen, the algorithm calculates the distances between the specified key point pairs and stores these distances in a list. This list is then used to create a pickle file for training our model.  
- Once the model is trained, it utilizes the distances between the specified pairs of key points. These distances are provided by the module we developed for our algorithm. This key-point distances which are returned by our module used by our model we trained to predict whether fighting is occurring or not.  

---

## Methodology

1. Reading image file using the file path or reading frame using OpenCV.  
2. If there is a person in the image or frame detecting key-points of the person using YOLOv8n-pose model.  
3. If there are multiple person the algorithm will calculate the distances between key-point pairs of each person that we provide to it. If there are hidden key-points it will automaticly handel value to it using symetric key-points or nearest key-point as a value for it.  
4. After calculating the key-point pairs distances for each person it will store the value into a list.  
5. Finally the module will return the list of key-point distances and using for loop it will provide the value of each key-point pair distances as an input for our model so that it will predict if there is a fight or not.  
