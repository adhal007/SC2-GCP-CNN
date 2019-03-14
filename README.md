# SC2-GCP-CNN

## Local Implementation of CNN neural network for StarCraft II using Keras
2 files to run the CNN neural network for SCII game data on command prompt.
1. ```model.py``` : This file defines the model architecture for the neural network. It currently uses 3 hidden layers of CNN with 32, 64, 128 architecture followed by Dense MLP with softmax implementation. It is imported into the second file
2. ``` task.py ```: This file builds the training data with the following arguments:<br/>
    a. ```--model-dir``` : ```(str)```type   - Give model directory where raw input data is located   <br/>    
    b. ```--model-name```: ```(str)``` type  - Give a model name for saving<br/>  
    c. ```--batch-size```: ```(int)```type   - Specify the batch size for each epoch to the run the training and evaluate the results<br/>  
    d. ```--test-split```: ```(float)```type - a value between (0, 1) for training testing data splitting <br/>  
    e. ```--seed```: ```(int)``` type        - set seed for the run <br/>  
    f. ```--increment```: ```(int)``` type   - additional argument for iterating through n files at a time when training and testing.
 
