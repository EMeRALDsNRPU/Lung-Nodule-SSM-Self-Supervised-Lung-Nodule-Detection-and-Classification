# Lung Nodule-SSM: Self-Supervised Lung Nodule Detection and Classification

This repository contains the source code for the paper *"Lung Nodule-SSM: Self-Supervised Lung Nodule Detection and Classification in Thoracic CT Images"* by Muniba Noreen and Furqan Shaukat, published in 2025. The project implements a fully self-supervised learning approach using DINOv2 for lung nodule detection and classification on the LUNA16 dataset, achieving an accuracy of 98.37% with a Random Forest classifier. Bounding boxes are inferred using feature similarity from DINOv2 representations.

## Requirements

- Python 3.8+
- NVIDIA GPU (tested on RTX 3090)
- Dependencies listed in `requirements.txt`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{noreen2025lung,
  title={Lung Nodule-SSM: Self-Supervised Lung Nodule Detection and Classification in Thoracic CT Images},
  author={Noreen, Muniba and Shaukat, Furqan},
  journal={...},
  year={2025}
}
```

## License

This project is licensed under the MIT License.
