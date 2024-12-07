# cnn-for-chf-detection
CNNs for Congestive Heart Failure Detection. Explanaible AI technique called GradCAM is used.

This repo is built to replicate the process mentioned in the research paper "A convolutional 
neural network approach to detect congestive heart failure" by Mihaela Porumb, Ernesto Iadanza, 
Sebastiano Massaro, and Leandro Pecchia.


Link: https://www.sciencedirect.com/science/article/abs/pii/S1746809419301776

![img.png](images/img.png)

Steps for reproducibility:
1. Execute the steps in the "/data/README.md". This will mean the data is in the correct format.
2. Next, execute 'data_processor.py' to get the processed_data. 
3. Next, execute 'Heartbeat_Classification.ipynb' to see the results.