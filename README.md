2D Layout estimation <br>
### Install spvloc dependencies 
```bash
#Ubuntu
source venv/bin/activate
git clone https://github.com/MRamazan/2D-Room-Layout-Estimation
cd 2D-Room-Layout-Estimation
pip install -r requirements.txt
```

```bash
cd spvloc
# Build and install redner
./data/setup/install_redner.sh
# Install patched version of pyrender
./data/setup/install_pyrender.sh
# if permission denied
# chmod +x ./data/setup/install_redner.sh ./data/setup/install_pyrender.sh
# rerun the scripts
```

```bash
python -m spvloc.tools.download_pretrained_models
```

```bash
cd ../
```

## Testing

### Segmented Image
```bash
python layout.py --image_path bedroom.jpg

```

### Layout Image
```bash
python core.py --image_path bedroom.jpg

```

### Results
![My Logo](results/res1.jpg) ![My Logo](results/res2.jpg) ![My Logo](results/res3.jpg)



