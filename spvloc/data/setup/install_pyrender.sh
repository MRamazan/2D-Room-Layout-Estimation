git clone --recursive https://github.com/mmatl/pyrender.git
git clone --recursive https://github.com/eyllanesc/pyrender.git pyrender_fix
cd pyrender_fix
git checkout alpha-channel
cd ..
cp -f pyrender_fix/pyrender/offscreen.py pyrender/pyrender
cp -f pyrender_fix/pyrender/renderer.py pyrender/pyrender
cd pyrender
python3 setup.py install
python3 -m pip wheel -w dist --verbose .
cd ..
rm -rf pyrender
rm -rf pyrender_fix

