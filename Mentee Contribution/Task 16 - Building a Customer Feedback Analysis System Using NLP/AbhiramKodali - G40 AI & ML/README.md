\# Task 16 - Customer Feedback Analysis System Using NLP



\*\*Contributor:\*\* AbhiramKodali - G40 AI \& ML



\## 1. Objective



Build a customer feedback analysis system using basic Natural Language Processing (NLP) techniques.



The system analyzes customer feedback for:



\- Sentiment: positive, neutral, or negative

\- Primary feedback category

\- Important words and phrases

\- Overall feedback patterns

\- Incorrect predictions and possible reasons for errors



The implementation focuses on NLP and machine learning rather than databases, APIs, or deployment.



\## 2. Approach



The implementation follows this workflow:



```text

Customer Feedback

&#x20;      |

&#x20;      v

Text Cleaning

&#x20;      |

&#x20;      v

Train/Test Split

&#x20;      |

&#x20;      v

TF-IDF Feature Extraction

&#x20;      |

&#x20;      +----------------------+

&#x20;      |                      |

&#x20;      v                      v

Sentiment Classifier    Category Classifier

&#x20;      |                      |

&#x20;      v                      v

Evaluation             Evaluation

&#x20;      |                      |

&#x20;      +----------+-----------+

&#x20;                 |

&#x20;                 v

&#x20;         Error Analysis

&#x20;                 |

&#x20;                 v

&#x20;      Important Terms \& Summary



3\. Dataset



A deterministic synthetic customer-feedback dataset was generated for reproducibility.



Dataset size

Total records: 378

Positive: 126

Neutral: 126

Negative: 126

Feedback categories

Category	Records

Payment	54

Login	54

Performance	54

Support	54

UI	54

Bug	54

Feature Request	54



The dataset contains the following columns:



feedback

sentiment

category



The dataset generator uses a fixed random seed (42) so that the same dataset can be reproduced.



4\. Text Preprocessing



The preprocessing step performs lightweight cleaning:



Converts text to lowercase

Removes HTML tags

Removes URLs and email addresses

Removes unnecessary punctuation

Normalizes whitespace



Important negation words are preserved.



For example, words such as:



not

no

never

cannot



are not removed because they can change the meaning of a sentiment statement.



The implementation intentionally avoids aggressive cleaning because removing useful context can reduce sentiment-classification quality.



5\. TF-IDF Feature Extraction



TF-IDF is used to convert customer feedback into numerical features.



Configuration:



Unigrams and bigrams: ngram\_range=(1, 2)

Sublinear TF scaling enabled

Maximum features: 5,000

Minimum document frequency: 1



The actual training data produced 709 TF-IDF features.



Using both unigrams and bigrams allows the model to capture useful phrases such as:



payment failing

very slow

not arriving

easy navigate

6\. Machine Learning Models



Two separate Logistic Regression classifiers are trained.



Sentiment classifier



Predicts:



Positive

Neutral

Negative

Category classifier



Predicts:



Payment

Login

Performance

Support

UI

Bug

Feature Request



Both models use:



Logistic Regression

max\_iter = 1000

random\_state = 42

7\. Train/Test Split



The dataset is divided into:



Training records: 283

Test records: 95



A fixed random seed of 42 is used for reproducibility.



The sentiment labels are stratified during the split so that the three sentiment classes remain represented in the test set.



8\. Evaluation Results

Sentiment classification

Metric	Result

Accuracy	98.95%

Macro Precision	98.99%

Macro Recall	98.92%

Macro F1	98.94%

Weighted F1	98.95%



There was 1 incorrect sentiment prediction out of 95 test records.



Category classification

Metric	Result

Accuracy	100%

Macro Precision	100%

Macro Recall	100%

Macro F1	100%

Weighted F1	100%



There were 0 category classification errors on the test set.



9\. Confusion Matrices



The following plots are generated automatically:



plots/

├── sentiment\_confusion\_matrix.png

├── category\_confusion\_matrix.png

└── sentiment\_distribution.png



The confusion matrices provide a class-by-class view of model performance.



10\. Important Words and Phrases



The Logistic Regression coefficients are used together with the TF-IDF vocabulary to identify terms strongly associated with each class.



The extracted results are stored in:



reports/important\_terms.csv



The report contains:



Model

Class

Rank

Term

Weight



Both individual words and two-word phrases are included because the TF-IDF representation uses unigrams and bigrams.



11\. Error Analysis



The sentiment classifier produced one incorrect prediction:



Feedback	Actual	Predicted

In my experience, Payment keeps failing during checkout.	Negative	Neutral

Analysis



The feedback clearly describes a failed payment, so the correct sentiment is negative.



However, the TF-IDF + Logistic Regression model classified it as neutral. A likely explanation is that the example contains strong payment-related vocabulary, while the model did not assign enough sentiment weight to the phrase describing the failure.



This demonstrates a limitation of basic TF-IDF classification: it relies heavily on the vocabulary and statistical associations learned from the training examples and does not deeply understand the meaning of a sentence.



The incorrect predictions are saved to:



reports/sentiment\_errors.csv



The error-analysis summary is saved to:



reports/error\_analysis\_summary.json

12\. Overall Feedback Summary



The generated dataset is balanced by sentiment and category.



The analysis identifies:



Most common sentiment: Negative

Most common category: Bug



The overall summary is stored in:



reports/feedback\_summary.json

13\. Project Structure

AbhiramKodali - G40 AI \& ML/

│

├── data/

│   └── feedback.csv

│

├── plots/

│   ├── sentiment\_distribution.png

│   ├── sentiment\_confusion\_matrix.png

│   └── category\_confusion\_matrix.png

│

├── reports/

│   ├── sentiment\_classification\_report.csv

│   ├── category\_classification\_report.csv

│   ├── sentiment\_errors.csv

│   ├── category\_errors.csv

│   ├── important\_terms.csv

│   ├── error\_analysis\_summary.json

│   └── feedback\_summary.json

│

├── src/

│   ├── generate\_dataset.py

│   └── analyze\_feedback.py

│

└── README.md

14\. Reproducibility



From the repository root, run:



python "Mentee Contribution/Task 16 - Building a Customer Feedback Analysis System Using NLP/AbhiramKodali - G40 AI \& ML/src/generate\_dataset.py"



Then run:



python "Mentee Contribution/Task 16 - Building a Customer Feedback Analysis System Using NLP/AbhiramKodali - G40 AI \& ML/src/analyze\_feedback.py"



The random seed is fixed at 42.



15\. Limitations



This implementation is a basic NLP system and has several limitations:



The dataset is synthetic rather than collected from real customers.

TF-IDF does not understand language context as deeply as modern language models.

The category classifier predicts one primary category per feedback record.

Mixed sentiment can be difficult to classify correctly.

Sarcasm and subtle language are not explicitly modeled.

The dataset is relatively small.

The perfect category score should not be interpreted as evidence that the model would achieve 100% accuracy on real-world customer feedback.



A future implementation could use a larger real-world dataset, stronger multi-label classification, word embeddings, or transformer-based NLP models.



16\. Conclusion



This project demonstrates a complete basic NLP workflow for customer feedback analysis.



The system successfully performs:



Text preprocessing

TF-IDF feature extraction

Sentiment classification

Feedback category classification

Quantitative model evaluation

Confusion-matrix visualization

Important-term extraction

Error analysis

Overall feedback summarization



The sentiment classifier achieved 98.95% test accuracy, while the category classifier achieved 100% test accuracy on the synthetic test set.



The error analysis also demonstrates why evaluation should not rely only on accuracy: inspecting the single incorrect sentiment prediction reveals a limitation in how TF-IDF represents contextual meaning.

