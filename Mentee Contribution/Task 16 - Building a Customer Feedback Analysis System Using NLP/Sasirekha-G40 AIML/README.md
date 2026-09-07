# Customer Feedback Analysis System Using NLP

This version is corrected for the actual uploaded dataset:
`Customer Complaints Sentiment and Priority Dataset.csv`.

The original starter project expected columns named `feedback`, `sentiment`, and `categories`. The real CSV instead contains:
- `Consumer_complaint` -> input text
- `Product` -> complaint category
- `Sentiment` -> 0/1, mapped to negative/positive
- `Priority` -> 0/1, mapped to low/high

## Windows commands

```powershell
cd C:\Users\ELCOT\Desktop\customer-feedback-nlp
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
python predict.py "The application is very slow and I cannot complete my payment."
streamlit run app.py
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The model uses TF-IDF + Logistic Regression. It predicts sentiment, product/complaint category, and priority, and also provides keywords and similar complaints.
