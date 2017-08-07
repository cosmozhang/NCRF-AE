# Neural CRF Autoencoder
This repository is the source code for the paper:

**Semi-supervised Structured Prediction with Neural CRF Autoencoder**
In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP) , 2017
*Xiao Zhang, Yong Jiang, Hao Peng, Kewei Tu and Dan Goldwasser*


* First run the *process_data.py* file to generate a pickle file.
    `process_data.py [bin_file] [training_file] [validation_file] [testing_file] [output_file]`
    1. The "bin_file" should has the same format as the generated file by https://code.google.com/archive/p/word2vec/.
    2. The "training_file", "validation_file" and "testing_file" should be in the format of CONLL-U, same as the one in the "data_format_example".
    3. The "output_file" is the destination file.
* Then run the *runsemi_EM.py* file to start.
    Run `runsemi_EM.py -h` to check the usage of the program.

For data, please refer to the references in our paper and download from the original sources of the datasets.

The code is under BSD-3 license.