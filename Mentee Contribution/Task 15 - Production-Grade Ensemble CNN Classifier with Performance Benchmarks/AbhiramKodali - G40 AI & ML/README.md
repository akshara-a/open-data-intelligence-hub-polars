\# Task 15 - Production-Grade Ensemble CNN Classifier with Performance Benchmarks



\## 1. Objective



The objective of this task is to build a production-oriented image classification system using multiple CNN models and combine their predictions using ensemble methods.



The project evaluates:



\- Multiple CNN architectures

\- Training-time image augmentation

\- Individual model accuracy

\- Precision, recall, and F1-score

\- Confusion matrices

\- Majority Voting

\- Soft Voting

\- Robustness to controlled image noise

\- Model disagreement

\- Inference latency

\- Throughput

\- Parameter count

\- Model size

\- Observed process memory increase



The final recommendation is based on the measured trade-offs between predictive performance and production cost.



\---



\## 2. Dataset



This project uses a deterministic synthetic casting-defect image dataset.



The dataset contains two classes:



\- `ok\_front` - acceptable casting images

\- `def\_front` - defective casting images



\### Dataset split



| Split | OK Images | Defect Images | Total |

|---|---:|---:|---:|

| Training | 120 | 120 | 240 |

| Validation | 30 | 30 | 60 |

| Test | 30 | 30 | 60 |

| Unseen | - | - | 5 |



The dataset is generated locally using `src/generate\_dataset.py`.



\### Synthetic dataset disclosure



The dataset is synthetic and was created specifically for this implementation because a production manufacturing dataset was not bundled with the task implementation.



Therefore, the reported metrics demonstrate the complete machine-learning workflow and benchmark methodology, but they should not be interpreted as evidence of real-world manufacturing performance.



\---



\## 3. Project Structure



```text

AbhiramKodali - G40 AI \& ML/

│

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

│

├── models/

│   ├── cnn\_small.keras

│   ├── cnn\_standard.keras

│   └── cnn\_deep.keras

│

├── plots/

│   ├── cnn\_small\_accuracy.png

│   ├── cnn\_small\_loss.png

│   ├── cnn\_standard\_accuracy.png

│   ├── cnn\_standard\_loss.png

│   ├── cnn\_deep\_accuracy.png

│   ├── cnn\_deep\_loss.png

│   ├── cnn\_small\_confusion\_matrix.png

│   ├── cnn\_standard\_confusion\_matrix.png

│   ├── cnn\_deep\_confusion\_matrix.png

│   ├── ensemble\_majority\_confusion\_matrix.png

│   └── ensemble\_soft\_confusion\_matrix.png

│

├── src/

│   ├── generate\_dataset.py

│   ├── data\_loader.py

│   ├── models.py

│   ├── train\_models.py

│   ├── ensemble.py

│   └── production\_benchmark.py

│

├── ensemble\_results.json

├── model\_benchmarks.json

├── production\_benchmarks.json

└── README.md

4\. Environment



The implementation was tested with:



Python 3.13.9

TensorFlow 2.21.0

scikit-learn 1.8.0

pandas 2.3.3

matplotlib 3.10.8

psutil 7.2.1

NumPy



TensorFlow was executed on a Windows CPU environment.



The TensorFlow GPU warning shown during execution is informational. The models trained and evaluated successfully using CPU execution.



5\. Data Preparation



Run the dataset generator:



python src/generate\_dataset.py



This creates the training, validation, test, and unseen-image directories.



Images are resized to:



224 x 224 x 3



Pixel values are normalized to the range:



0 to 1



The training data uses shuffling and batching with a batch size of 32.



6\. Data Augmentation



Training-time augmentation is included in every CNN.



The augmentation pipeline uses:



Horizontal flipping

Random rotation

Random zoom

Random contrast adjustment



The purpose is to expose the models to small variations in image appearance and reduce dependence on a single exact image presentation.



Augmentation is applied during training and is not intended to alter the deterministic test evaluation.



7\. CNN Architectures



Three CNN architectures were implemented to compare different model capacities.



CNN Small

Input 224x224x3

&#x20;   ↓

Conv2D 16 filters

&#x20;   ↓

MaxPooling

&#x20;   ↓

Conv2D 32 filters

&#x20;   ↓

MaxPooling

&#x20;   ↓

Global Average Pooling

&#x20;   ↓

Dropout

&#x20;   ↓

Dense 64

&#x20;   ↓

Dropout

&#x20;   ↓

Sigmoid Output



Parameters:



7,265



CNN Standard

Input 224x224x3

&#x20;   ↓

Conv2D 32 filters

&#x20;   ↓

MaxPooling

&#x20;   ↓

Conv2D 64 filters

&#x20;   ↓

MaxPooling

&#x20;   ↓

Conv2D 128 filters

&#x20;   ↓

MaxPooling

&#x20;   ↓

Global Average Pooling

&#x20;   ↓

Dropout

&#x20;   ↓

Dense 128

&#x20;   ↓

Dropout

&#x20;   ↓

Sigmoid Output



Parameters:



109,889



CNN Deep



The Deep model uses additional convolutional blocks:



Input 224x224x3

&#x20;   ↓

Conv2D 32

&#x20;   ↓

Conv2D 32

&#x20;   ↓

MaxPooling

&#x20;   ↓

Conv2D 64

&#x20;   ↓

Conv2D 64

&#x20;   ↓

MaxPooling

&#x20;   ↓

Conv2D 128

&#x20;   ↓

Conv2D 128

&#x20;   ↓

MaxPooling

&#x20;   ↓

Global Average Pooling

&#x20;   ↓

Dropout

&#x20;   ↓

Dense 128

&#x20;   ↓

Dropout

&#x20;   ↓

Sigmoid Output



Parameters:



303,649



8\. Training Configuration



All three models use the same basic optimization configuration so that the architecture comparison remains meaningful.



Setting	Value

Image size	224 × 224

Classes	2

Optimizer	Adam

Learning rate	0.001

Loss	Binary Cross-Entropy

Batch size	32

Maximum epochs	25

Output activation	Sigmoid

Metrics	Accuracy, Precision, Recall

Augmentation	Flip, rotation, zoom, contrast

9\. Individual Model Evaluation



The models were evaluated on the 60-image test set.



Predictive and production benchmarks

Model	Clean Accuracy	Robust Accuracy	Accuracy Drop	Parameters	Model Size	Latency (ms/image)	Throughput (images/sec)	Memory Increase

CNN Small	53.33%	88.33%	-35.00%	7,265	0.1355 MB	4.9518	201.95	0.0938 MB

CNN Standard	100%	100%	0%	109,889	1.3175 MB	11.1634	89.58	0.1250 MB

CNN Deep	100%	100%	0%	303,649	3.5497 MB	28.9616	34.53	0.0898 MB

Interpretation



CNN Small is the lightest and fastest model, but its clean test accuracy is only 53.33%.



CNN Standard achieves 100% clean and robust accuracy while remaining substantially smaller and faster than CNN Deep.



CNN Deep also achieves 100% clean and robust accuracy, but requires more parameters, more storage, and higher inference latency.



The negative accuracy drop for CNN Small means its accuracy was higher on this particular noisy robustness sample than on the clean test sample. This should not be interpreted as evidence that Gaussian noise generally improves model performance.



10\. Ensemble Methods



Two ensemble strategies were implemented.



10.1 Majority Voting



Each CNN produces a binary prediction.



The final prediction is determined by the majority of the three model predictions.



For three models:



2 or more positive votes → positive

otherwise → negative

10.2 Soft Voting



Each CNN produces a probability.



The probabilities from the three models are averaged:



average probability = mean(model probabilities)



A threshold of 0.5 is then used to generate the final binary prediction.



11\. Ensemble Results

Majority Voting

Metric	Clean Test	Robust Test

Accuracy	100%	100%

Precision	100%	—

Recall	100%	—

F1 Score	100%	—

Accuracy Drop	—	0%



Confusion matrix:



\[\[30, 0],

&#x20;\[ 0,30]]

Soft Voting

Metric	Clean Test	Robust Test

Accuracy	100%	100%

Precision	100%	—

Recall	100%	—

F1 Score	100%	—

Accuracy Drop	—	0%



Confusion matrix:



\[\[30, 0],

&#x20;\[ 0,30]]



Both ensemble methods achieved perfect classification on the clean and robustness test sets used in this experiment.



12\. Model Disagreement



The individual CNN predictions were also compared.



The models disagreed on:



28 out of 60 test images



Disagreement rate:



46.67%



This demonstrates that the models do not make identical predictions for every sample, which is relevant when evaluating whether an ensemble can provide complementary decision behavior.



Despite these disagreements, both Majority Voting and Soft Voting produced 100% accuracy on the evaluated clean test set.



13\. Robustness Evaluation



A controlled robustness test was performed by adding Gaussian noise to the test images.



The noise was generated using a fixed random seed to make the experiment reproducible.



The noisy images were clipped to the valid normalized pixel range.



This test measures sensitivity to one specific synthetic perturbation. It is not a substitute for testing on real manufacturing variation, lighting changes, camera differences, or real-world defects.



Robustness results

Model / Ensemble	Clean Accuracy	Robust Accuracy	Accuracy Drop

CNN Small	53.33%	88.33%	-35.00%

CNN Standard	100%	100%	0%

CNN Deep	100%	100%	0%

Majority Voting	100%	100%	0%

Soft Voting	100%	100%	0%

14\. Production Performance



The benchmark was executed on the local CPU environment.



CNN Standard



CNN Standard provides the strongest balance in this experiment:



100% clean accuracy

100% robustness-test accuracy

109,889 parameters

1.3175 MB model size

11.1634 ms/image latency

89.5788 images/sec throughput

CNN Deep



CNN Deep also achieves perfect clean and robustness-test accuracy, but has:



303,649 parameters

3.5497 MB model size

28.9616 ms/image latency

34.5285 images/sec throughput



Therefore, Deep has a significantly higher computational cost without improving the measured accuracy on this dataset.



Ensemble cost



Running all three models sequentially gives an estimated combined latency of:



45.0768 ms/image



Estimated sequential throughput:



22.1844 images/sec



This estimate is based on the sum of the current individual-model latency benchmarks.



The ensemble therefore provides multiple-model decision aggregation at the cost of running all three CNNs.



15\. Production Recommendation

Recommended default model: CNN Standard



Based on the measured results, CNN Standard is the recommended default production candidate for this experiment.



The reason is not simply its accuracy.



CNN Standard achieved:



100% clean accuracy

100% robustness-test accuracy

1.3175 MB model size

11.1634 ms/image latency

89.5788 images/sec throughput

substantially fewer parameters than CNN Deep



CNN Deep provides the same measured predictive performance but is considerably larger and slower.



CNN Small is substantially faster and smaller, but its 53.33% clean accuracy makes it unsuitable as the primary classifier based on this test set.



When to use the ensemble



Majority Voting and Soft Voting both achieved 100% clean and robust accuracy in this experiment.



However, the ensemble requires all three models to be executed, producing an estimated sequential latency of 45.0768 ms/image.



Therefore:



Use CNN Standard when balanced production efficiency is the priority.

Consider CNN Deep when additional model capacity is justified by future real-world validation.

Consider the ensemble when the additional inference cost is acceptable and multiple-model agreement is valuable.



The final choice should be revalidated on a representative real-world casting dataset before deployment.



16\. Limitations

The dataset is synthetic rather than a production manufacturing dataset.

The test set contains only 60 images.

Robustness testing uses controlled Gaussian noise only.

The benchmarks were collected on a local CPU environment.

Memory increase is an observed process-memory change during benchmarking and should not be treated as a complete hardware memory profile.

Ensemble latency is a sequential estimate based on the measured individual-model latencies.

Real production deployment should include additional validation for lighting, camera variation, image quality, defect diversity, and distribution shift.

17\. Reproducibility



From the Task 15 directory:



Generate the dataset

python src/generate\_dataset.py

Train the three CNNs

python src/train\_models.py

Run the individual production benchmarks

python src/production\_benchmark.py

Run ensemble evaluation

python src/ensemble.py



The generated JSON benchmark files and PNG visualizations contain the measured results from these experiments.



18\. Conclusion



This task demonstrates a complete production-oriented CNN ensemble workflow for binary casting-defect classification.



Three CNN architectures were trained and benchmarked, followed by Majority Voting and Soft Voting ensemble evaluation.



CNN Standard and CNN Deep both achieved 100% clean and robustness-test accuracy, while CNN Standard required substantially fewer parameters and lower inference latency.



Both ensemble approaches also achieved 100% accuracy on the evaluated clean and robustness test sets. However, the ensemble has a higher inference cost because all three models must be executed.



For this experiment, CNN Standard provides the best balance of predictive performance, model size, latency, and throughput.



The results should be considered a reproducible technical demonstration rather than a claim of production manufacturing accuracy because the dataset is synthetic.

