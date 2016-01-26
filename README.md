# Startup Press Release Analyzer  

##Project Overview
**Project Summary:**  
Performed NLP text analysis on large corpus of Startup press releases. Used TF-IDF and NMF to find latent topics. Explored how latent topics vary by industry and location.


**Process:**  
List of startup companies from [Crunchbase.com](www.crunchbase.com).  
Downloaded corpus of Press Releases using [PR Newswire](http://www.prnewswire.com/) API and store in MongoDB.  
Data Cleaning and Feature Engineering  
NMF Models: TF-IDF, NMF => Get Latent Topics among Press Releases  
Results: Found several startup type buzzwords as topics (i.e. Software, Biotech, Mobile, Apps, etc).

##Detailed Process
###NLP Algorithms Used
**[TF-IDF: Term Frequency - Inverse Document Frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)**  
After cleaning the data I fed every press release text through a TF-IDF model. This turns every document into a numerical vector representing how important each word is to the document. The word strength increases by the number of times the word appears in the text (term frequency), but is offset by the frequency of the word (inverse document frequency) as rarely used terms can often be more important than common ones.

**[NMF: Non-Negative Matrix Factorization](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization)**  
I took the output matrix of the TF-IDF model and fed it into an NMF model. The NMF algorithm factors the input matrix into two separate matrices, W & H. The H matrix represents the entire vocabulary of your corpus, while the W matrix represents the strength of each vocab word across a number of latent features. I then got the top words for each latent feature, to figure out the most represented topics in my corpus.  

###Model Validation
Unlike a regression model, there is no accuracy or other metric to use to validate your findings. You also can't use algorithms like [Scikit Learn's Grid Search](http://scikit-learn.org/stable/modules/grid_search.html) to find the optimal inputs for each of the TF-IDF and NMF model parameters. 

The one parameter that I was especially interested in optimizing was the number of latent features for my NMF model. This determines the number of columns for the W matrix when the model performs the factorization. The most difficult part is doing a proper train test split on the TF-IDF output matrix for your cross validation. For a five fold cross validation, you would typically hold out 20% of the observations. If you do that for on the input to the NMF model, it can't factor the matrix properly that will give you any useful results.

Therefore, after reading a couple scientific papers on grid search's for NMF models, I found that instead of removing 20% of the rows, you could set a random 20% of the entries to zero. Then find the Mean Squared Error (MSE) between the approximate original matrix formed by multiplying the W and H sub-matrices, and the actual input matrix. You then do this five times for an equivalent five fold cross validation, take the average of the MSE's for those five runs, and that's your overall MSE for that parameter setting.

I built several functions to perform this cross validation. I found after having greater than three components, the MSE only improved by an average of 0.1 - 0.5% for each additional component. However, if I only choose three components, that would mean I would only have three major topics for my documents. This is where I had to make a decision between mathematical optimization and human interpretability. After reviewing the top 15 words for each topic on several different settings, I decided having eight latent topics was the best choice for a human reader.

I also got an additional form of soft model validation. When downloading the press releases from PR Newswire, each document comes with attached industry classifications. After choosing my eight latent topics, I found they correlated very well with the industry classifications provided from that outside source.

###Results
Here are the top 15 words from each of my eight latent topics. As you can see, they all have pretty clear separation of topics and contain several "startup" type words and themes.  
**Topic #1:**  
insurance quote car auto coverage driver online comparing website agency client plan policy multiple carrier  
**Topic #2:**  
data cloud service solution customer business software enterprise management network security technology platform application provider  
**Topic #3:**  
health patient care healthcare medical hospital clinical cancer physician elsevier treatment research disease drug program  
**Topic #4:**  
content video student social game digital marketing medium online brand learning company school new platform  
**Topic #5:**  
solar energy power project renewable module utility sunpower electricity skypower battery smart technology electric home  
**Topic #6:**  
epson printer label print printing pos color america projector ink 3d trademark ricoh seiko registered  
**Topic #7:**  
app mobile user device new android apps store iphone windows available feature apple vehicle phone  
**Topic #8:**  
statement looking forward risk company result uncertainty securities release future factor actual materially differ exchange

##Detailed Code Walk Through
Below will be a brief description of each file of my code. I'll go over the big chunks of what's going on in each file, so you could follow along if you're interested in reading the code in more depth.

###data_acquisition.py
This file handles taking the list of crunchbase companies, querying the PR Newswire API and storing the results in the local MongoDB database.  
- Import list of Crunchbase organizations from CSV file into Pandas DataFrame  
- Import only organziations that classify themselves as a 'company' or 'investor'  
- Create MongoDB connection with proper database and collection name  
- Prep the request parameters for the PR Newswire API as specified in their documentation
- After trial and error, I found the API could only handle about 50 company names for each request, so I built a function to separate the company names into sets of 50 and prep each API request with this in mind.  
- Each press release came as an HTML doc. I used [BeautifulSoup](http://www.crummy.com/software/BeautifulSoup/) to extract the text from the HTML.  
- I assembled each API response into a dictionary and then inserted it into MongoDB.  
- I also had an API call limit of 5000 requests per day, so I created status print statements to note the last batch of company names successfully downloaded, so I could start at the next batch for tomorrow.  
- At the end I had over 6500 unique press releases.

###data_munging.py  
This file takes the press releases from MongoDB, cleans the data and adds some new features, making it ready for TF-IDF model input.
- Import MongoDB collection into a Pandas DataFrame for easier manipulation in Python.  
- Take all the industry and subject classifications and make dummy columns for later use.
- The city source name of each release is in the text of each document at the beginning. I make a function to extract these city names from the text and store them in another column.  
- There were too many cities to do any useful analysis, so I built some dictionaries to add the state, country and regions based on each city name.  
- I used [TextBlob](https://textblob.readthedocs.org/en/dev/) to lemmatize and perform sentiment analysis on each document.
- The DataFrame was now ready to be put into the TF-IDF model.

###model.py  
Puts each press release through TF-IDF and NMF to produce latent topics.
- Import press releases from MongoDB into a Pandas DataFrame.  
- Call main function from data_munging.py to clean and prep the data.
- Input press release text into [Scikit Learn's TF-IDF Vectorizer Model](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) to output the sparse matrix.  
- Take the TF-IDF output and put it into [Sci-kit Learn's NMF model](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html) to get the W and H factorization matrices based on the number of latent topics I specified.  
- Lastly, I wrote a program to print the top words from each latent topic so a human could read them and make their own analysis.

##How to run the program
In order to run the entire program, you would have to have access to the Crunchbase list of companies csv file and the MongoDB database containing the press releases; both of which are too large for this Github repo. However, if you did have both those files, you would simply open up a terminal and call 'python model.py'. The entire program would run and return to you the top 15 words from the top eight latent topics from my NMF model.