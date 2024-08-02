#!/bin/bash

# Download model weights and check their MD5 checksums
wget https://github.com/ncbi-nlp/DeepSeeNet/releases/download/0.1/adv_amd_model.h5
echo '0adbf448491ead63ac384da671c4f7ee  adv_amd_model.h5' | md5sum -c -

wget https://github.com/ncbi-nlp/DeepSeeNet/releases/download/0.2/cga_model.h5
echo 'fe4a441d5286154633de46c25099b520  cga_model.h5' | md5sum -c -

wget https://github.com/ncbi-nlp/DeepSeeNet/releases/download/0.1/drusen_model.h5
echo '997a8229f972482e127e8a32d1967549  drusen_model.h5' | md5sum -c -

wget https://github.com/ncbi-nlp/DeepSeeNet/releases/download/0.2/ga_model.h5
echo '59350371c73eaaff397d477b702c456a  ga_model.h5' | md5sum -c -

wget https://github.com/ncbi-nlp/DeepSeeNet/releases/download/0.1/pigment_model.h5
echo 'e38f60fa9c0fc6cd7a5022b07b722927  pigment_model.h5' | md5sum -c -
