echo "Starting setup..."
echo " "
echo " "
echo " "

# install packages for project
pip install -r requirements.txt

# make cv2 work as expected
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

#install miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash

# install xformers to run model
~/miniconda3/bin/conda install xformers -c xformers/label/dev

# install packages for fairface
pip install -r requirements_fairface.txt
pip install cmake
pip install dlib

# copy models into fairface folder (and rename them) 
# cp fairface_models/res34_fair_align_multi_4_20190809.pt fairface/fair_face_models/fairface_alldata_4race_20191111.pt
# cp fairface_models/res34_fair_align_multi_7_20190809.pt fairface/fair_face_models/fairface_alldata_7race_20191111.pt

echo " "
echo " "
echo " "
echo "Finished setup."