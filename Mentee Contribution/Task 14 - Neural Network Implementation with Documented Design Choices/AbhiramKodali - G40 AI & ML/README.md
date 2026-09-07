\# Task 14 - Neural Network Implementation with Documented Design Choices



\## 1. Objective



Build a binary image classification CNN for quality control to classify casting images into:



\- `ok\_front` - acceptable casting

\- `def\_front` - defective casting



The implementation documents the important design and training decisions and evaluates the trained model on both test images and five unseen images.



\---



\## 2. Dataset Description



A small synthetic image dataset was created specifically for this implementation so that the complete pipeline can be reproduced locally.



The dataset contains two balanced classes:



| Dataset Split | ok\_front | def\_front | Total |

|---|---:|---:|---:|

| Training | 80 | 80 | 160 |

| Validation | 20 | 20 | 40 |

| Test | 20 | 20 | 40 |

| Unseen | 5 | 0/5\* | 5 |



\\\*The five unseen images are used only for prediction demonstration and are not used during training or evaluation.



Images are resized to \*\*224 x 224\*\* pixels and normalized before being passed to the CNN.



The defective examples contain visually distinct defect patterns such as cracks, spots, and scratches, while acceptable examples represent normal casting surfaces.



\---



\## 3. Project Structure



```text

AbhiramKodali - G40 AI \& ML/

|

├── data/

│   ├── train/

│   │   ├── ok\_front/

│   │   └── def\_front/

│   ├── val/

│   │   ├── ok\_front/

│   │   └── def\_front/

│   ├── test/

│   │   ├── ok\_front/

│   │   └── def\_front/

│   └── unseen/

|

├── models/

│   └── casting\_defect\_model.keras

|

├── plots/

│   ├── accuracy\_plot.png

│   ├── loss\_plot.png

│   └── confusion\_matrix.png

|

└── src/

&#x20;   ├── generate\_dataset.py

&#x20;   ├── data\_loader.py

&#x20;   ├── model.py

&#x20;   ├── train.py

&#x20;   └── evaluate.py

4\. CNN Architecture



The model uses three convolutional feature-extraction blocks followed by global average pooling and fully connected classification layers.



Architecture:



Input: 224 x 224 x 3

&#x20;       |

Conv2D - 32 filters, 3x3, ReLU

&#x20;       |

MaxPooling2D

&#x20;       |

Conv2D - 64 filters, 3x3, ReLU

&#x20;       |

MaxPooling2D

&#x20;       |

Conv2D - 128 filters, 3x3, ReLU

&#x20;       |

MaxPooling2D

&#x20;       |

GlobalAveragePooling2D

&#x20;       |

Dropout (0.40)

&#x20;       |

Dense (128), ReLU

&#x20;       |

Dropout (0.40)

&#x20;       |

Dense (1), Sigmoid



The final model contains 109,889 trainable parameters.



Global Average Pooling was used instead of flattening the final feature maps. This substantially reduces the number of parameters while retaining spatially aggregated feature information.



5\. Design Decision Table

Decision	Selected Design	Reason

Image Size	224 x 224	Standard size suitable for CNN image processing

Task Type	Binary Classification	Two classes: acceptable and defective

Model	CNN	Suitable for learning visual image features

Convolution Filters	32, 64, 128	Gradually increases feature capacity

Kernel Size	3 x 3	Captures local visual patterns efficiently

Activation	ReLU	Provides non-linear feature learning

Pooling	MaxPooling2D	Reduces spatial dimensions and computation

Global Feature Reduction	GlobalAveragePooling2D	Reduces parameters compared with Flatten

Output Activation	Sigmoid	Produces a binary classification probability

Optimizer	Adam	Efficient gradient-based optimizer

Learning Rate	0.001	Selected as the training learning rate

Loss Function	Binary Cross-Entropy	Appropriate for binary classification

Batch Size	32	Provides a practical training batch size

Maximum Epochs	25	Limits training duration

Dropout	0.40	Helps reduce overfitting

Augmentation	Flip, rotation, zoom, contrast	Improves variation in training images

Evaluation Metrics	Accuracy, Precision, Recall	Measures classification performance from multiple perspectives

6\. Training Configuration



The model was trained for a maximum of 25 epochs.



Early stopping was used with a patience value of 5 epochs, while the best validation model was saved using model checkpointing.



The training pipeline records:



Training accuracy

Validation accuracy

Training loss

Validation loss

Precision

Recall



The resulting accuracy and loss graphs are stored in the plots/ directory.



7\. Test Evaluation



The final trained model was evaluated using the independent test set containing 40 images.



Results

Metric	Result

Accuracy	1.0000

Precision	1.0000

Recall	1.0000

Test Loss	0.0192



The confusion matrix was:



\[\[20, 0],

&#x20;\[ 0, 20]]



This means all 20 images from each test class were classified correctly, with no false positives or false negatives in this test set.



The confusion matrix visualization is saved as:



plots/confusion\_matrix.png

8\. Unseen Image Predictions



Five images that were not used for model training or test evaluation were passed through the trained model.



Image	Predicted Class

unseen\_1.png	ok\_front

unseen\_2.png	def\_front

unseen\_3.png	ok\_front

unseen\_4.png	def\_front

unseen\_5.png	ok\_front



The model produced confident probability values for these examples.



These images are used only to demonstrate prediction behavior on previously unseen inputs.



9\. Reproducibility



To generate the dataset:



python src/generate\_dataset.py



To train the CNN:



python src/train.py



To evaluate the trained model:



python src/evaluate.py



The trained model is saved as:



models/casting\_defect\_model.keras

10\. Conclusion



A binary CNN-based quality-control classifier was implemented using a reproducible image-processing and training pipeline.



The model achieved 100% accuracy, precision, and recall on the current 40-image test set, with a confusion matrix showing 20 correct predictions for each class.



The architecture uses three convolutional layers with increasing filter capacity, MaxPooling, Global Average Pooling, dropout, and a sigmoid output layer.



The evaluation also demonstrates predictions on five unseen images.



Because the dataset used for this implementation is synthetic and relatively small, the reported performance should not be interpreted as evidence of real-world industrial performance. A production system would require a larger, independently collected dataset containing representative casting variation and real defect conditions.



11\. Files Delivering the Implementation

src/generate\_dataset.py - reproducible dataset generation

src/data\_loader.py - image loading and preprocessing

src/model.py - CNN architecture

src/train.py - model training and graph generation

src/evaluate.py - test evaluation, confusion matrix, and unseen predictions

models/casting\_defect\_model.keras - trained CNN

plots/accuracy\_plot.png - accuracy graph

plots/loss\_plot.png - loss graph

plots/confusion\_matrix.png - confusion matrix

