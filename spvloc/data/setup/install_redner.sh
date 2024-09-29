git clone --recursive https://github.com/BachiLi/redner.git
cd redner 
python3 setup.py install
python3 -m pip wheel -w dist --verbose .
cd ..
rm -rf redner

