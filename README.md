# SpamClassifier_LSTM

> **Abstract:**
This project aims to develop a Chinese spam email detection system based on Long Short-Term Memory (LSTM) networks. By analyzing features such as grammatical errors, persuasive language, and unknown sender addresses, the system efficiently identifies spam emails. It utilizes natural language processing techniques for text preprocessing and leverages deep learning algorithms to enhance classification performance. Experimental results demonstrate that the classifier excels in terms of accuracy, precision, and recall, effectively filtering spam in complex Chinese contexts.

## Preparation

### Install

We test the code on PyTorch 2.2.0 + TorchText 0.17.0 with MPS(MacBook Pro M1max).

In a Linux environment, we test our code on PyTorch 2.2.0 + TorchText 0.17.0 with CUDA 11.8(NVIDIA RTX4090).

1. Create a new conda environment
```sh
conda create -n py39 python=3.9
conda activate py39
```
2. Install dependencies
```sh
# for Linux
pip install torch==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118
# for MacOS
pip install torch==2.2.0

pip install -r requirements.txt
```

### Download

You can download the pretrained models and datasets on [BaiduPan](https://pan.baidu.com/s/1SG4jZSySIRYgjrqhQ2y6CQ?pwd=yvp1).

Extract the compressed package, select the required files, and place them into folders named 'ham' and 'spam' respectively.

The final file path should be the same as the following:

```
┬─ data
│   ├─ ham
│   │   ├─ 1
│   │   ├─ 9
│   │   └─ ... (other texts)
│   └─ spam
│       ├─ 2
│       └─ ... (other texts)
├─ LSTM.py
├─ inference.py
├─ preprocess.py
└─ ...
```

## Training and Evaluation

If you are using an NVIDIA graphics card and accelerating inference with CUDA, please replace line 12 in `LSTM.py`, line 8 in `inference.py` and line 10 in `eval.py` with the following code.

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Train

You can modify the training settings for each experiment in `LSTM.py`. Then run the following script to train the model:

```sh
python LSTM.py
```

### Test

Run the following script to test the trained model:

```sh
python eval.py
```

### Inference

You can either input your own data or use randomly generated samples for inference.
Run the script below to perform model inference:

```sh
python inference.py
```

Run the script below to launch the graphical operating system:

```sh
python GUI.py
```

## Results

The performance of our LSTM-based spam detection model was evaluated using the **TREC 2006 (trec06c) dataset**, with the following results:

| **Metric**   | **Score** |
|--------------|-----------|
| Accuracy     | 0.9904    |
| Precision    | 0.9889    |
| Recall       | 0.9957    |
| F1 Score     | 0.9923    |

These results demonstrate the model's high effectiveness in identifying spam emails, with a strong balance between precision and recall, indicating its reliability in both detecting spam and minimizing false positives.

## Train your own data

If you wish to train using your own dataset, replace the texts in the ‘ham’ and ‘spam’ folders with your data, and then run thw following script:

```sh
python preprocess.py
```

After running the script to generate `processed_emails.csv`, proceed by executing `LSTM.py` to complete the model training. 

Subsequently, you can perform model evaluation and inference.

## Notes

1. Course Project at Shanghai University: This project was developed as a part of the Natural Language Processing (NLP) course at Shanghai University. The primary goal was to apply cutting-edge machine learning techniques to real-world problems, specifically focusing on spam detection in Chinese language datasets.
2. Open for Collaboration and Improvement: We believe that knowledge grows through collaboration. We welcome contributions, suggestions, and feedback from anyone interested in improving this project. Whether it’s enhancing the LSTM model’s accuracy, integrating new NLP techniques, or optimizing feature extraction, we are open to all ideas.
3. Learning and Sharing: As part of an academic initiative, we encourage students, researchers, and professionals alike to engage with this project. Feel free to use this repository as a learning resource or a foundation for your own projects. If you have any questions or insights, we would be delighted to exchange ideas and grow together.

Thank you for your interest, and we look forward to meaningful discussions and innovations!


