# Text_segmentor
_________________________________________________________________
Embedding used >> 
Word embedding(Glove Vectors) 
	+ 
char embedding(BiLSTM Network to generate char embedding vector)
_________________________________________________________________
Model >>
BiLSTM + crf
(Bidirectional LSTM WITH CONDITIONAL RANDOM FIELDS NETWORK
_________________________________________________________________

# steps to follow:
1. Put datafile in the data folder
2. Data should be pure sentences in txt file.
3. Put glove embedding file of required dimension in glove folder(download it from stanford.edu)
4. Fill all slots(Hyperparameters & names) in config_params.json
5. use python train.py to start training
6. predict.py for future inferences

(Best model will be saved in the model folder after training)

_________________________________________________________________
Benchmark         | without punct | with punct    
zeroshot(appx.)   |     83.76     |     87.8   
Finetune(appx.)   |     95.02     |     97.1
_________________________________________________________________
