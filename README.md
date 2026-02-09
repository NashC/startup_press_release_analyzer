# Startup Press Release Analyzer

NLP analysis of 6,500+ startup press releases to discover latent topics using unsupervised learning. Built as a capstone project for the Galvanize Data Science Immersive program.

## Approach

1. Scraped press releases from PR Newswire's API using a Crunchbase company list as seed data, stored in MongoDB
2. Cleaned and featurized text (lemmatization, sentiment analysis via TextBlob, city/state/region extraction)
3. Transformed corpus with TF-IDF, then factored the term-document matrix with Non-negative Matrix Factorization (NMF) to extract latent topics

## Model Validation

NMF doesn't have a standard accuracy metric or a straightforward way to do cross-validation — you can't remove 20% of rows from the TF-IDF matrix without breaking the factorization. Following published approaches for NMF grid search, I instead zeroed out a random 20% of matrix entries per fold, reconstructed the approximate matrix from W and H, and measured MSE against the original. Five-fold cross-validation showed diminishing returns beyond 3 components (~0.1-0.5% MSE improvement per additional component), but 3 topics were too coarse for human interpretation. After reviewing the top-15 words at several settings, 8 topics gave the best separation.

External validation: the 8 discovered topics correlated well with PR Newswire's own industry classifications, which the model never saw.

## Results

Eight latent topics with clear thematic separation:

| Topic | Theme | Top words |
|-------|-------|-----------|
| 1 | Insurance | insurance, quote, car, auto, coverage, driver, online |
| 2 | Enterprise SaaS | data, cloud, service, solution, software, enterprise, platform |
| 3 | Healthcare | health, patient, care, medical, hospital, clinical, cancer |
| 4 | Digital Media | content, video, social, game, digital, marketing, medium |
| 5 | Clean Energy | solar, energy, power, renewable, module, utility, battery |
| 6 | Hardware/Printing | epson, printer, label, print, 3d, projector, ink |
| 7 | Mobile Apps | app, mobile, user, device, android, apps, iphone |
| 8 | SEC Filings | statement, forward, risk, securities, future, differ, exchange |

## Code

- `data_acquisition.py` — PR Newswire API scraping with rate limit handling, batch processing, HTML→text extraction via BeautifulSoup, MongoDB storage
- `data_munging.py` — Text cleaning, feature engineering (industry dummies, geographic extraction, lemmatization, sentiment)
- `model.py` — TF-IDF vectorization, NMF factorization, topic extraction. Run with `python model.py`

## Tech Stack

Python, scikit-learn (TF-IDF, NMF), MongoDB, pandas, TextBlob, BeautifulSoup, Flask (demo app)
