#!/usr/bin/env bash
set -e

# Replace with your dataset dir
export DATA_ROOT="/path/to/dataset"
cd $DATA_ROOT


###*  Download Konk Lab Massive Memo dataset *###
echo "Download Konk Lab dataset"
mkdir -p KonkLab
pushd KonkLab
wget --progress=bar \
   http://olivalab.mit.edu/MM/archives/ObjectCategories.zip \
   -O ObjectCategories.zip
unzip ObjectCategories.zip
rm ObjectCategories.zip
popd


###* Download ImageNet val *###
echo "Downloading ImageNet val"
mkdir -p ImageNet/ILSVRC2012
pushd ImageNet/ILSVRC2012
wget --progress=bar \
   https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar \
   -O ILSVRC2012_img_val.tar

wget --progress=bar \
   https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz \
   -O ILSVRC2012_devkit_t12.tar.gz
popd


###* Download broden1_224 *###
if [ ! -f Broden/broden1_224/index.csv ]
then

echo "Downloading broden1_224"
mkdir -p Broden
pushd Broden
wget --progress=bar \
   http://netdissect.csail.mit.edu/data/broden1_224.zip \
   -O broden1_224.zip
unzip broden1_224.zip
rm broden1_224.zip
popd

fi

