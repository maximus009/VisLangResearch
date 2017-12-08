import os
base_path = os.getcwd()
if not os.path.basename(base_path) == 'moviescope':
    base_path = os.path.join(base_path, '..')

import sys
sys.path.append(base_path)

## recurrent networks; 
# 1. Identify all data points with similar frames
# 2. create batches and save in pickle
# 3. load and train model for each batch
# 4. compare and test model


## model can be various architectures like LSTM, Bi-LSTM, etc
