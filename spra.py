#spra.py

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.grid_search import GridSearchCV
from nltk import tokenize
from nltk.corpus import stopwords
from pymongo import MongoClient
import math
import re
from scipy import sparse
from textblob import TextBlob
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score


def mongo_to_df(db, collection):
	'''
		Import data from MongoDB into Pandas DataFrame
		
		INPUT:
		- db: database name
		- collection: collection name

		OUTPUT:
		- df: Pandas DataFrame containing all collection entries
	'''
	#start pymongo client
	connection = MongoClient('127.0.0.1', 27017)
	#set database
	db = connection[db]
	#set collection
	input_data = db[collection]
	#put all entries into Pandas dataframe
	df = pd.DataFrame(list(input_data.find()))
	return df

def make_dummies(df, col, prefix):
	'''
		Create dummies columns based on column name
		
		INPUT:
		- df: Pandas DataFrame
		- col: column name for which dummies will be made
		- prefix: prefix for names of new dummies columns

		OUTPUT:
		- df: Pandas DataFrame with new dummy columns appended
	'''
	#set column variable
	s = df[col]
	#make dummies columns
	dummies = pd.get_dummies(s.apply(pd.Series), prefix=prefix, prefix_sep='_').sum(level=0, axis=1)
	#concat dummies columns to right side of dataframe
	result = pd.concat([df, dummies], axis=1)
	return result

def get_cities(text):
	'''
		Extract city name from text string of press release
		
		INPUT:
		- text (string): press release text

		OUTPUT:
		- city name (string): lowercased city name
	'''
	#city name appears at the start of every press release followed by 'PRNewswire', so split on that string and take first entry
	city_date = text.split('PRNewswire')[0].split()
	#temp list to append results of below for loop
	temp = []
	for i, word in enumerate(city_date[:-1]):
		#all city names are upper case, this identifies them
		if word.isupper():
			#some city names are two words, this checks for that and puts into one entry
			if city_date[i+1].isupper():
				temp.append(' '.join((city_date[i], city_date[i+1])))
			#checks if previous entry was uppercase and then doesn't recount this entry.
			elif city_date[i-1].isupper():
				continue
			#if only one uppercased word, then take this entry
			else:
				temp.append(word)
	#there were a two entries that did not have any results from the above city search code, decided to test length of city name list to prevent errors
	if len(temp) >= 1:
		#sometimes above code gets more than one string, so we only want the first entry
		city = temp[0]
		#remove commas that always follow uppercased city names
		city = re.sub(r',', '', city, flags=re.M)
		#lowercase city names
		city = city.lower()
		return city
	else:
		#for those releases without a city name, return 'none'
		return 'none'

def state_from_city(city_name):
	'''
		Takes in city name and returns USA state or international country.
		
		INPUT:
		- city_name

		OUTPUT:
		- state name if in dictionary or 'none' if not
	'''
	#dictionary with USA states or international cities as keys, and associated city names as values. Dictionary created from highest value_counts() of city names in the DataFrame
	city_states = {
		'CA': ['los angeles', 'san francisco', 'san jose', 'san diego', 'irvine', 'sunnyvale', 'santa clara', 'mountain view', 'san mateo', 'palo alto', 'long beach', 'santa monica', 'newport beach', 'milpitas', 'fremont', 'campbell', 'redwood city', 'pleasanton', 'santa barbara', 'fountain valley', 'beverly hills', 'aliso viejo', 'foster city', 'anaheim', 'menlo park', 'south san', 'el segundo', 'pasadena'],
		'NY': ['new york', 'armonk n.y', 'hauppauge n.y'],
		'IL': ['chicago', 'naperville'],
		'MA': ['boston', 'burlington', 'waltham'],
		'UK': ['england', 'cambridge', 'oxford'],
		'TX': ['dallas', 'austin', 'houston', 'fort worth', 'irving', 'plano', 'arlington'],
		'GA': ['atlanta'],
		'NV': ['las vegas'],
		'Tel Aviv': ['tel aviv', 'tel-aviv'],
		'CO': ['denver', 'boulder', 'colorado springs', 'broomfield', 'fort collins'],
		'DC': ['washington'],
		'FL': ['miami', 'orlando', 'fort lauderdale', 'tampa', 'jacksonville', 'boca raton'],
		'OR': ['portland', 'beaverton'],
		'PA': ['philadelphia', 'pittsburgh', 'malvern', 'west chester'],
		'UT': ['salt lake'],
		'MO': ['st louis', 'kansas city'],
		'VA': ['reston', 'mclean', 'alexandria'],
		'MD': ['baltimore', 'bethesda'],
		'France': ['paris'],
		'MN': ['minneapolis'],
		'MI': ['ann arbor', 'troy'],
		'WA': ['seattle', 'bellevue', 'redmond'],
		'AZ': ['phoenix', 'scottsdale', 'chandler'],
		'OH': ['cleveland', 'cincinnati', 'columbus'],
		'UAE': ['dubai uae'],
		'NC': ['raleigh n.c', 'durham n.c', 'charlotte n.c'],
		'NJ': ['englewood cliffs', 'paramus', 'dover n.j', 'newark', 'newark n.j'],
		'India': ['bangalore', 'new delhi'],
		'Spain': ['barcelona'],
		'Germany': ['berlin'],
		'IN': ['indianapolis'],
		'DE': ['wilmington'],
		'NH': ['portsmouth n.h'],
		'Sweden': ['gothenburg'],
		'BC': ['vancouver'],
		'Belgium': ['brussels'],
		'WI': ['milwaukee'],
		'Finland': ['helsinki'],
		'AL': ['huntsville', 'birmingham'],
		'Ireland': ['dublin'],
		'Austria': ['vienna'],
		'TN': ['nashville'],
		'CT': ['stamford', 'new haven'],
		'SC': ['greenville s.c'],
		'NE': ['lincoln'],
		'Russia': ['moscow'],
		'Singapore': ['singapore'],
		'Switzerland': ['lausanne', 'zurich'],
		'Italy': ['milan'],
		'ON': ['toronto']
	}
	#iterate through dictionary, return state name if city_name is in the associated value list
	for k,v in city_states.iteritems():
		if city_name in v:
			return k
	return 'none'

def country_from_state(state_name):
	'''
		Takes in state name and returns country name
		
		INPUT:
		- state_name

		OUTPUT:
		- country name if in dictionary or 'none' if not
	'''
	#dictionary with country names as keys and state_name lists as values
	state_countries = {
		'USA': ['WA','WI','WV','FL','WY','NH','NJ','NM','NA','NC','ND','NE','NY','RI','NV','GU','CO','CA','GA','CT','OK','OH','KS','SC','KY','OR','SD','DE','DC','HI','PR','TX','LA','TN','PA','VA','VI','AK','AL','AS','AR','VT','IL','IN','IA','AZ','ID','ME','MD','MA','UT','MO','MN','MI','MT','MP','MS'],
		'Canada': ['BC', 'ON'],
		'UK': ['UK', 'Ireland'],
		'Belgium': ['Belgium'],
		'UAE': ['UAE'],
		'Spain': ['Spain'],
		'Switzerland': ['Switzerland'],
		'Israel': ['Tel Aviv'],
		'India': ['India'],
		'Italy': ['Italy'],
		'France': ['France'],
		'Singapore': ['Singapore'],
		'Germany': ['Germany'],
		'Finland': ['Finland'],
		'Sweden': ['Sweden'],
		'Russia': ['Russia'],
		'Austria': ['Austria']
	}
	#iterate through dictionary, return country name if state_name is in the associated value list
	for k,v in state_countries.iteritems():
		if state_name in v:
			return k
	return 'none'

def region_from_state(state_name):
	'''
		Takes in state name and returns region name
		
		INPUT:
		- state_name

		OUTPUT:
		- region name if in dictionary or 'none' if not
	'''	
	#dictionary with region names as keys and state_name lists as values
	regions = {
		'EMEA': ['Tel Aviv', 'France', 'UAE', 'Spain', 'Germany', 'Sweden', 'UK', 'Finland', 'Ireland', 'Austria', 'Belgium', 'Russia'],
		'West Coast (ex-CA)': ['NV', 'CO', 'OR', 'UT', 'WA', 'AZ', 'BC'],
		'Northeast': ['ON', 'NY', 'MA', 'DC', 'PA', 'VA', 'MD', 'NJ', 'DE', 'NH', 'CT'],
		'Midwest/South': ['NE', 'MO', 'MN', 'MI', 'WI', 'IL', 'IN', 'OH','TX', 'GA', 'TN', 'SC', 'NC', 'FL', 'AL']
	}
	#iterate through dictionary, return region name if state_name is in the associated value list
	for k,v in regions.iteritems():
		if state_name in v:
			return k
	return 'none'

def norcal_socal(city_name):
	'''
		Takes in california city name and returns 'Silicon Valley' or 'Southern CA'
		
		INPUT:
		- city_name

		OUTPUT:
		- CA region name
	'''
	#dictionary with CA region name as keys and associated list of cities as values
	california = {
		'Silicon Valley': ['san francisco', 'san jose', 'sunnyvale', 'santa clara', 'mountain view', 'san mateo', 'palo alto', 'milpitas', 'fremont', 'campbell', 'redwood city', 'pleasanton', 'menlo park', 'south san'],
		'Southern CA': ['los angeles', 'san diego', 'irvine', 'long beach', 'newport beach', 'santa monica', 'santa barbara', 'fountain valley', 'beverly hills', 'aliso viejo', 'foster city', 'anaheim', 'el segundo', 'pasadena']
	}
	#iterate through dictionary, return CA region name if city_name is in the associated value list, otherwise pass
	for k,v in california.iteritems():
		if city_name in v:
			return k
	pass

def lemmatize(doc_text):
	'''
		Lemmatize text from press release
		
		INPUT:
		- doc_text: press release text

		OUTPUT:
		- same press release text, but all words replaced by associated lemmas
	'''
	#create TextBlob object using doc_text as input
	blob = TextBlob(doc_text)
	#temp list for appending results of below for loop
	temp = []
	#Loop through all words in the documents and append their lemmas to the temp list
	for word in blob.words:
		temp.append(word.lemmatize())
	#rejoin all words from temp list with space
	result = ' '.join(temp)
	return result

def prep_release_text(df):
	'''
		Perform operations on press release text to prep for model input
		
		INPUT:
		- df: Pandas DataFrame containing release text

		OUTPUT:
		- same DataFrame, but with operations performed on release text
	'''
	#regex to remove all URL's
	df['release_text'] = df['release_text'].apply(lambda x: re.sub(r'\(?(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)\)?', '', x, flags=re.M))
	#Lemmatize all words using about lemmatize() function
	df['release_text'] = df['release_text'].apply(lambda x: lemmatize(x))
	return df

def print_top_words(model, feature_names, n_top_words):
	'''
		Print n most common words for each latent topic in a corpus
		
		INPUT:
		- model: sklearn fitted model object
		- feature_names: Vocabulary of corpus (features)
		- n_top_words: Number of top words to print

		OUTPUT:
		- Prints n top words for each latent topic
	'''
	for topic_idx, topic in enumerate(model.components_):
		print("Topic #%d:" % topic_idx)
		print(" ".join([feature_names[i]
						for i in topic.argsort()[:-n_top_words - 1:-1]]))
	pass

def make_stop_words(new_words_list):
	'''
		Take in list of user-created stop words and join with Tfidf 'english' stop words.
		
		INPUT:
		- new_words_list: user-created stop words

		OUTPUT:
		- New master list of stop words including user and model inputs
	'''
	#create temporary TfidfVectorizer object
	tfidf_temp = TfidfVectorizer(stop_words='english')
	#get Tfidf 'english' stop words from model
	stop_words = tfidf_temp.get_stop_words()
	#combine two lists of stop words
	result = list(stop_words) + new_words_list
	return result

def get_topics(n_components=10, n_top_words=15, print_output=True, max_features=None):
	custom_stop_words = make_stop_words(new_stop_words)
	tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=custom_stop_words, max_features=max_features)
	tfidf = tfidf_vectorizer.fit_transform(release_texts)
	tfidf = normalize(tfidf, axis=1, norm='l1')
	nmf = NMF(n_components=n_components, random_state=1)
	nmf.fit(tfidf)
	W = nmf.transform(tfidf)
	if print_output:
		print("\nTopics in NMF model:")
		tfidf_feature_names = tfidf_vectorizer.get_feature_names()
		print_top_words(nmf, tfidf_feature_names, n_top_words)
	return tfidf, nmf, W

def tfidf_traintestsplit(tfidf_sparse, test_size=0.2):
	A = tfidf_sparse
	total_entries = A.shape[0] * A.shape[1]
	train_size = 1. - test_size
	ones = np.ones(math.ceil(total_entries * train_size))
	zeros = np.zeros(math.floor(total_entries * test_size))
	r_temp = np.append(ones, zeros)
	np.random.shuffle(r_temp)
	R = np.reshape(r_temp, (A.shape))
	R_flip = np.logical_not(R)
	R = sparse.csr_matrix(R)
	R_flip = sparse.csr_matrix(R_flip)
	A_train = sparse.spmatrix.multiply(A,R)
	A_test = A
	return A_train, A_test

def grid_search_nmf_ncomponents(tfidf, folds, low, high, export_array):
	tfidf_dense = tfidf.toarray()
	mse_min = 99
	mse_min_ncomponents = -1
	for i in xrange(low, high + 1):
		print 'Fitting n_components = %d ...' %i
		mse_arr = []
		for j in xrange(1, folds + 1):
			print 'Testing fold # %d' %j
			test_size = 1./folds
			A_train, A_test = tfidf_traintestsplit(tfidf, test_size=test_size)
			nmf_temp = NMF(n_components=i, random_state=1)
			nmf_temp.fit(A_train)
			W = nmf_temp.transform(A_train)
			H = nmf_temp.components_
			tfidf_pred = np.dot(W, H)
			mse_fold = mean_squared_error(A_test.toarray(), tfidf_pred)
			mse_arr.append(mse_fold)
		mse_temp = np.mean(mse_arr)
		export_array.append((i, mse_temp))
		if mse_temp < mse_min:
			mse_min = mse_temp
			mse_min_ncomponents = i
		print 'MSE of n_components = %d: %.10f' %(i, mse_temp)
		print '-------------------------------'
	pass

def sum_dummie_counts(df):
	for col in df.columns:
		try:
			print col, sum(df[col])
		except:
			pass

def textblob_sentiment(text):
	blob = TextBlob(text)
	polarity = blob.sentiment.polarity
	subjectivity = blob.sentiment.subjectivity
	return (polarity, subjectivity)

def feature_engineering(df):
	df.drop_duplicates(subset=['article_id'], inplace=True)
	df['city'] = df['release_text'].apply(lambda x: get_cities(x))
	df['state'] = df['city'].apply(lambda x: state_from_city(x))
	df['region'] = df['state'].apply(lambda x: region_from_state(x))
	df['region'] = df.apply(lambda x: norcal_socal(x['city']) if x['state'] == 'CA' else x['region'], axis=1)
	df['country'] = df['state'].apply(lambda x: country_from_state(x))
	df['CA_region'] = df['city'].apply(lambda x: norcal_socal(x))
	df['polarity'] = df['release_text'].apply(lambda x: textblob_sentiment(x)[0])
	df['subjectivity'] = df['release_text'].apply(lambda x: textblob_sentiment(x)[1])
	return df

def make_subject_dict():
	subject_list = [
	[u'ACC', u'Accounting News, Issues'],
	[u'TNM', u'Acquisitions, Mergers, Takeovers'],
	[u'BCY', u'Bankruptcy'],
	[u'RTG', u'Bond/Stock Ratings'],
	[u'CON', u'Contracts'],
	[u'DIV', u'Dividends'],
	[u'ERN', u'Earnings'],
	[u'ERP', u'Earnings Projects or Forecasts'],
	[u'ECO', u'Economic News, Trends and Analysis'],
	[u'FNC', u'Financing Agreements'],
	[u'JVN', u'Joint Ventures'],
	[u'LIC', u'Licensing/Marketing Agreements'],
	[u'PDT', u'New Products/Services'],
	[u'OFR', u'Offerings'],
	[u'DSC', u'Oil/Gas Discoveries'],
	[u'OTC', u'OTC/SmallCap IRW'],
	[u'PER', u'Personnel Announcements'],
	[u'RLE', u'Real Estate Transactions'],
	[u'RCN', u'Restructuring/Recapitalizations'],
	[u'SLS', u'Sales Reports'],
	[u'SRP', u"Shareholders' Rights Plans"],
	[u'LEG', u'Federal and State Legislation'],
	[u'EXE', u'Federal Executive Branch, Agency News'],
	[u'CPN', u'Political Campaigns'],
	[u'LBR', u'Labor/Union news'],
	[u'BLK', u'African-American News'],
	[u'HSP', u'Hispanic-Oriented News'],
	[u'LAW', u'Legal Issues'],
	[u'AWD', u'Awards'],
	[u'NPT', u'Not for Profit'],
	[u'TDS', u'Tradeshow News'],
	[u'CCA', u'Conference Call Announcements'],
	[u'CHI', u'Children-Related News'],
	[u'WOM', u'Women-related News'],
	[u'VEN', u'Venture Capital'],
	[u'BFA', u'Broadcast Feed Announcement'],
	[u'ASI', u'Asian-Related News'],
	[u'EGV', u'European Government'],
	[u'MAV', u'Media Advisory/Invitation'],
	[u'SVY', u'Surveys, Polls & Research'],
	[u'INO', u'Investments Opinions'],
	[u'ZHA', u'Xinhua'],
	[u'FOR', u'Foreign policy/International affairs'],
	[u'POL', u'Domestic Policy'],
	[u'TRD', u'Trade Policy'],
	[u'REL', u'Religion'],
	[u'STS', u'Stock Split'],
	[u'PET', u'Animals/Pets'],
	[u'TRI', u'Clinical Trials/Medical Discoveries'],
	[u'RCY', u'Conservation/Recycling'],
	[u'CSR', u'Corporate Social Responsibility'],
	[u'FDA', u'FDA Approval'],
	[u'DIS', u'Handicapped/Disabled'],
	[u'LGB', u'Lesbian/Gay/Bisexual'],
	[u'NTA', u'Native American'],
	[u'PLW', u'Patent Law'],
	[u'RCL', u'Product Recalls'],
	[u'PSF', u'Public Safety'],
	[u'SCZ', u'Senior Citizens'],
	[u'SBS', u'Small Business Services'],
	[u'STP', u'U.S. State Policy News'],
	[u'VET', u'Veterans'],
	[u'VDM', u'MultiVu Video'],
	[u'ADM', u'MultiVu Audio'],
	[u'PHM', u'MultiVu Photo'],
	[u'BCM', u'Broadcast Minute'],
	[u'CXP', u'Corporate Expansion'],
	[u'ENI', u'Environmental Issues'],
	[u'ENP', u'Environmental Policy'],
	[u'SRI', u'Socially Responsible Investing'],
	[u'VNR', u'Video News Releases'],
	[u'ANW', u'Animal Welfare'],
	[u'AVO', u'Advocacy Group Opinion'],
	[u'OBI', u'Obituaries'],
	[u'FEA', u'Features']]
	subject_dict = {x[0]:x[1] for x in subject_list}
	return subject_dict

def make_industry_dict():
	ind_list = [
	[u'ADV', u'Advertising '],
	[u'ARO', u'Aerospace/Defense'],
	[u'AGR', u'Agriculture'],
	[u'AIR', u'Airlines/Aviation'],
	[u'ART', u'Art'],
	[u'AUT', u'Automotive'],
	[u'FIN', u'Banking/Financial Services'],
	[u'BIO', u'Biotechnology'],
	[u'BKS', u'Books'],
	[u'CHM', u'Chemical'],
	[u'CPR', u'Computer/ Electronics'],
	[u'NET', u'Networks'],
	[u'HRD', u'Computer Hardware'],
	[u'STW', u'Computer Software'],
	[u'CST', u'Construction/Building'],
	[u'CSE', u'Consumer Electronics'],
	[u'EDU', u'Education'],
	[u'EPM', u'Electronics Performance Measurement'],
	[u'ECM', u'Electronic Commerce'],
	[u'ENT', u'Entertainment'],
	[u'ENV', u'Environmental Products & Services'],
	[u'FAS', u'Fashion'],
	[u'FLM', u'Film and Motion Picture'],
	[u'FOD', u'Food & Beverages'],
	[u'CNO', u'Gambling/Casinos'],
	[u'HEA', u'Health Care/Hospitals'],
	[u'HOU', u'Household/Consumer/Cosmetics'],
	[u'INS', u'Insurance'],
	[u'ITE', u'Internet Technology'],
	[u'LEI', u'Leisure & Tourism'],
	[u'MAC', u'Machinery'],
	[u'MAG', u'Magazines'],
	[u'MAR', u'Maritime/Shipbuilding'],
	[u'MTC', u'Medical/Pharmaceuticals'],
	[u'MNG', u'Mining/Metals'],
	[u'MLM', u'Multimedia/Internet'],
	[u'MUS', u'Music'],
	[u'MFD', u'Mutual Funds'],
	[u'OFP', u'Office Products'],
	[u'OIL', u'Oil/Energy'],
	[u'PAP', u'Paper/Forest Products/Containers'],
	[u'PEL', u'Peripherals'],
	[u'PUB', u'Publishing/Information Services'],
	[u'RAD', u'Radio'],
	[u'RLT', u'Real Estate'],
	[u'REA', u'Retail'],
	[u'RST', u'Restaurants'],
	[u'SPT', u'Sports'],
	[u'SUP', u'Supermarkets'],
	[u'SPM', u'Supplementary Medicine'],
	[u'TLS', u'Telecommunications Industry'],
	[u'TVN', u'Television'],
	[u'TEX', u'Textiles'],
	[u'TOB', u'Tobacco'],
	[u'TRN', u'Transportation/Trucking/Railroad'],
	[u'TRA', u'Travel'],
	[u'UTI', u'Utilities'],
	[u'Feature', u'Features'],
	[u'HTS', u'High Tech Security'],
	[u'ECP', u'Electronic Components'],
	[u'EDA', u'Electronic Design Automation'],
	[u'SEM', u'Semiconductors'],
	[u'HED', u'Higher Education'],
	[u'ALC', u'Beers, Wines and Spirits'],
	[u'BIM', u'Biometrics'],
	[u'GAM', u'Electronic Gaming'],
	[u'HMS', u'Homeland Security'],
	[u'IDC', u'Infectious Disease Control'],
	[u'MEN', u'Mobile Entertainment'],
	[u'NAN', u'Nanotechnology'],
	[u'WRK', u'Workforce Management/Human Resources'],
	[u'AIF', u'Air Freight'],
	[u'ALT', u'Alternative Energies'],
	[u'ANW', u'Animal Welfare'],
	[u'ATL', u'Amusement Parks and Tourist Attractions'],
	[u'BEV', u'Beverages'],
	[u'BRI', u'Bridal Services'],
	[u'CPC', u'Cosmetics and Personal Care'],
	[u'CRL', u'Commercial Real Estate'],
	[u'DEN', u'Dentistry'],
	[u'ENS', u'Environmental Products & Services'],
	[u'EUT', u'Electrical Utilities'],
	[u'FRN', u'Furniture and Furnishings'],
	[u'GAS', u'Gas'],
	[u'HHP', u'Household Products'],
	[u'HIN', u'Health Insurance '],
	[u'HMI', u'Home Improvements'],
	[u'HRT', u'Hotels and Resorts'],
	[u'HSC', u'Home Schooling'],
	[u'HVA', u'HVAC'],
	[u'JWL', u'Jewelry'],
	[u'MCT', u'Machine Tools, Metalworking and Metallurgy'],
	[u'MEQ', u'Medical Equipment'],
	[u'MIN', u'Mining'],
	[u'MNH', u'Mental Health'],
	[u'NAB', u'Non-Alcoholic Beverages'],
	[u'ORF', u'Organic Food'],
	[u'ORL', u'Overseas Real Estate (non-US) '],
	[u'OUT', u'Outsourcing Businesses'],
	[u'PAV', u'Passenger Aviation'],
	[u'PHA', u'Pharmaceuticals'],
	[u'PRM', u'Precious Metals'],
	[u'RFI', u'RFID (Radio Frequency ID) Applications & Tech'],
	[u'RIT', u'Railroads & Intermodal Transporation'],
	[u'RRL', u'Residential Real Estate'],
	[u'SMD', u'Social Media'],
	[u'SPE', u'Sports Equipment & Accessories'],
	[u'SSE', u'Sporting Events'],
	[u'SWB', u'Semantic Web'],
	[u'TCS', u'Telecommunications Carriers and Services'],
	[u'TEQ', u'Telecommunications Equipment'],
	[u'TRT', u'Trucking and Road Transportation'],
	[u'VIP', u'VoIP (Voice over Internet Protocol)'],
	[u'WEB', u'Web site'],
	[u'WIC', u'Wireless Communications'],
	[u'WUT', u'Water Utilities'],
	[u'GRE', u'Green Technology'],
	[u'OTC', u'OTC/SmallCap'],
	[u'SRI', u'Socially Responsible Investing'],
	[u'TOY', u'Toys'],
	[u'BRD', u'Broadcast Technology']
	]
	ind_dict = {x[0]:x[1] for x in ind_list}
	return ind_dict

def nmf_component_plot(mse_arr, show=False):
	x = np.arange(1,len(mse_arr) + 1)
	y = [j[1] for j in mse_arr]
	y_percent = []
	for i in xrange(len(mse_arr)-1):
		diff = abs(mse_arr[i+1][1] - mse_arr[i][1])
		y_percent.append(diff / mse_arr[i][1])
	y_percent = [y_percent[0]] + y_percent
	
	f, axarr = plt.subplots(2, sharex=True, figsize=(12,12))
	axarr[0].plot(x, y)
	axarr[0].set_title('MSE vs NMF(n_components)')
	axarr[1].scatter(x, y_percent)
	if show:
		plt.show()

def topic_strength_by_category(df, W, meta_item):
	W = normalize(W, axis=1, norm='l1')
	W = pd.DataFrame(W)
	W.columns = ['Topic' + str(i) for i in xrange(W.shape[1])]
	df.reset_index(drop=True, inplace=True)
	W.reset_index(drop=True, inplace=True)
	df_W = pd.concat([df,W], axis=1)
	if type(meta_item) == list:
		topics_by_meta = df_W.groupby(meta_item).sum().iloc[:,-W.shape[1]:]
	else:
		topics_by_meta = df_W.groupby([meta_item]).sum().iloc[:,-W.shape[1]:]
	topics_by_meta_norm = normalize(topics_by_meta.values, axis=1, norm='l1')
	df_topics_by_meta_norm = pd.DataFrame(topics_by_meta_norm)
	df_topics_by_meta_norm.index = topics_by_meta.index
	df_topics_by_meta_norm.columns = topics_by_meta.columns
	df_topics_by_meta_norm = df_topics_by_meta_norm[df_topics_by_meta_norm.index != 'none']
	return df_topics_by_meta_norm

new_stop_words = ['ha', "\'s", 'tt', 'ireach', "n\'t", 'wo', 'pv', 'tm', 'anite', 'rabichev', 'russell', '603', 'hana', 'atmel', 'radwin', 'se', 'doxee', 'lantto', 'publ', 'fpc1025', '855', 'il', '0344']
subject_dict = make_subject_dict()
industry_dict = make_industry_dict()
df_orig = mongo_to_df('press', 'test_master_1')
df = prep_release_text(df_orig)
df = feature_engineering(df)
df = make_dummies(df, 'industry', 'ind')
df = make_dummies(df, 'subject', 'subj')
release_texts = df['release_text']
tfidf, nmf, W = get_topics(n_components=8, print_output=True)









