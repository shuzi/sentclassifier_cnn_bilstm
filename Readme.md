
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 th train.lua -type cuda -trainFile ../NLC_data/train -validFile ../NLC_data/valid -testFile ../NLC_data/test -embeddingFile ../NLC_data/embedding -numLabels 311 -epoch 100 -batchSize 1 -batchSizeTest 5 -learningRate 0.02 -numFilters 1000 -hiddenDim 1000 -wordHiddenDim 100 -LSTMmode 5 -model 3 -LSTMhiddenSize 500 -lastReLU


If you publish anything using this code, please cite our paper: Applying Deep Learning to Answer Selection: A Study and An Open Task Minwei Feng, Bing Xiang, Michael R. Glass, Lidan Wang, Bowen Zhou ASRU 2015
