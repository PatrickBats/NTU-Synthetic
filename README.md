# 🧠 Discovering Hidden Visual Concepts Beyond Linguistic Input in Infant Learning

🔗 [**Project Page**](https://kexueyi.github.io/webpage-discover-hidden-visual-concepts/)  
📄 [**Paper (CVPR 2025)**](https://arxiv.org/abs/2501.05205)

---

Can a computational model that imitates infant learning develop **broader visual concepts** beyond the vocabulary it has heard — **similar to how infants naturally learn**?

We explore this question using the [**CVCL** model](https://www.science.org/doi/10.1126/science.adi1374) (a multimodal infant-inspired model) proposed by Vong et al., published in *Science*.

---

## 🔍 Key Findings

- 🧒 The infant model develops a **broader understanding of visual concepts** than those found in its linguistic training input.
- 🧠 Certain **neurons** in the model can **classify visual concepts** **without additional training**.
- 🔄 There are clear **differences in representations** between this infant model and broadly-trained models such as **CLIP** and **ImageNet-based models**.


## Setup
### Dataset
You may simply run `dataset_setup.sh`
- Konkle Lab Dataset: [download here](http://olivalab.mit.edu/MM/archives/ObjectCategories.zip)  
- Broden Dataset: [download here](http://netdissect.csail.mit.edu/data/broden1_227.zip) (for [net-dissect](https://netdissect.csail.mit.edu) in [experiment 3](#experiment-3-layer-wise-analysis))
- ImageNet Validation set: [download here](http://image-net.org/synset?wnid=n01440764) (for [CKA](https://arxiv.org/pdf/1905.00414) in [experiment 3](#experiment-3-layer-wise-analysis))


### Environment
We recommend using `conda` to create a new environment. We use `pytho=3.8` and `torch==2.0.1`.
```bash
conda create --name prob_cvcl python=3.8
conda activate prob_cvcl
pip install -r requirements.txt
python -m spacy download en_core_web_sm # download spacy model for CVCL
pip install -e .
```

## Run Experiments
We provide bash scripts under `scripts/` folder to reproduce our results. 

### Experiment 1: Neuron Labeling 
We utlize [CILP-dissect](https://github.com/Trustworthy-ML-Lab/CLIP-dissect) to label neurons for CVCL.
```bash
.\scripts\neuron_labeling.sh
```

### Experiment 2: *NeuronClassifier* 
Note that below process will generate trials for *n-way* classification in `json` format under `data/trials/` folder. 

#### Baseline:
Run vanilla *n-way* classification:
```bash
.\scripts\baseline_n_way.sh 
```

#### Neuron-based Classification:
This step requires the neuron labeling ([experiment 1](#experiment-1-neuron-labeling)) to be completed first.

To reproduce our study using neurons for classification inside representations:
```bash
.\scripts\neuron_n_way.sh
```

### Experiment 3: Layer-wise Analysis
Comparing infant CVCL model with ResNext and CLIP models using CKA:
```bash
python -m src.compute_cka --batch_size 512 --model_x resnext
python -m src.compute_cka --batch_size 512 --model_x clip-res
```
Further, analysis ResNeXt and CVCL using NetDissect:
```bash
cd net-dissect
python main.py  # settings: net-dissect/settings.py
```

Please be aware that [net-dissect](https://github.com/CSAILVision/NetDissect-Lite) is very memory-intensive. Ensure you have sufficient storage space before running. 


## Reproduce All Results 
**All figures and table** in this paper can be reproduced using the jupyter notebooks under `notebooks/` folder. 

While the results may not be exactly same as trials are influenced by device variation (even with fixed random seeds), overall trends should be consistent.

## Notes

### Customization
- For custom datasets: Modify `DATASET_ROOTS` in `src/utils/dataset_loader.py`
- For custom models: Adapt the `load_model` and `show_available_models` functions in `src/utils/model_loader.py`

### Running Notes
- The classification task follows *n-way* setting, with each experiment generating trial `json` files stored in the `data/trials/` folder.
- Overall classficiation results are append to `csv` files under `experiments/trials/results/` folder. Be careful if you run the same experiment multiple times, as it will results duplicate rows in the `csv` files, influencing the final results.
- Refer to [CLIP-dissect](https://github.com/Trustworthy-ML-Lab/CLIP-dissect). Activation files are saved under `experiments/` for each run. If an experiment fails, partial activations remain but won't update in subsequent runs due to name detection. To rerun, please delete the corresponding activation files manually.

### Device

The code is for run on a single GPU. For multiple GPUs, please modify accordingly. However, as we mainly perform inference (no training at all) in this study, single *RTX3090Ti* should be sufficient for memory.

## Sources
Many thanks to the researchers and contributors behind these resources that made our work possible. We are grateful for their inspiration and for open-sourcing the code and data that our study builds upon:

- Age of Acquisition (AoA): https://norare.clld.org/contributions/Kuperman-2012-AoA 
- CKA: https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment
- CLIP-Dissect: https://github.com/Trustworthy-ML-Lab/CLIP-dissect
- Common English words 30k: https://github.com/arstgit/high-frequency-vocabulary/blob/master/30k.txt
- CVCL's Implementation: https://github.com/wkvong/multimodal-baby
- DINO Infant Model: https://github.com/eminorhan/silicon-menagerie/blob/master/vision_transformer_dino_mugs.py
- Net-Dissect: https://github.com/CSAILVision/NetDissect-Lite

## Citation 
🥳 We appreciate your interest in our work! If you find our study helpful, please cite the following:
```bibtex
@inproceedings{ke2025discovering,
  author = {Xueyi Ke and Satoshi Tsutsui and Yayun Zhang and Bihan Wen},
  title = {Discovering Hidden Visual Concepts Beyond Linguistic Input in Infant Learning},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2025},
  note = {To appear}
}
```
