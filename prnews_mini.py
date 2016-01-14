#prnews_mini.py

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.grid_search import GridSearchCV
from nltk import tokenize
from nltk.corpus import stopwords
from pymongo import MongoClient
from time import time
import math
import re
from scipy import sparse
from textblob import TextBlob
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score


def mongo_to_df(db, collection):
	connection = MongoClient('127.0.0.1', 27017)
	db = connection[db]
	input_data = db[collection]
	df = pd.DataFrame(list(input_data.find()))
	return df

def make_dummies(df, col, prefix):
	s = df[col]
	dummies = pd.get_dummies(s.apply(pd.Series), prefix=prefix, prefix_sep='_').sum(level=0, axis=1)
	result = pd.concat([df, dummies], axis=1)
	return result

def get_cities(text):
	city_date = text.split('PRNewswire')[0].split()
	temp = []
	for i, word in enumerate(city_date[:-1]):
		if word.isupper():
			if city_date[i+1].isupper():
				temp.append(' '.join((city_date[i], city_date[i+1])))
			elif city_date[i-1].isupper():
				continue
			else:
				temp.append(word)
	if len(temp) >= 1:
		city = temp[0]
		city = re.sub(r',', '', city, flags=re.M)
		city = city.lower()
		return city
	else:
		return 'none'

def state_from_city(city_name):
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
	for k,v in city_states.iteritems():
		if city_name in v:
			return k
	return 'none'

def country_from_state(state_name):
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
	for k,v in state_countries.iteritems():
		if state_name in v:
			return k
	return 'none'

def region_from_state(state_name):
	regions = {
		'EMEA': ['Tel Aviv', 'France', 'UAE', 'Spain', 'Germany', 'Sweden', 'UK', 'Finland', 'Ireland', 'Austria', 'Belgium', 'Russia'],
		'Asia': ['India', 'Singapore'],
		'West Coast': ['CA', 'NV', 'CO', 'OR', 'UT', 'WA', 'AZ', 'BC'],
		'North East': ['ON', 'NY', 'MA', 'DC', 'PA', 'VA', 'MD', 'NJ', 'DE', 'NH', 'CT'],
		'South': ['TX', 'GA', 'TN', 'SC', 'NC', 'FL', 'AL'],
		'Midwest': ['NE', 'MO', 'MN', 'MI', 'WI', 'IL', 'IN', 'OH']
	}
	for k,v in regions.iteritems():
		if state_name in v:
			return k
	return 'none'

def norcal_socal(city_name):
	california = {
		'NorCal': ['san francisco', 'san jose', 'sunnyvale', 'santa clara', 'mountain view', 'san mateo', 'palo alto', 'milpitas', 'fremont', 'campbell', 'redwood city', 'pleasanton', 'menlo park', 'south san'],
		'SoCal': ['los angeles', 'san diego', 'irvine', 'long beach', 'newport beach', 'santa monica', 'santa barbara', 'fountain valley', 'beverly hills', 'aliso viejo', 'foster city', 'anaheim', 'el segundo', 'pasadena']
	}
	for k,v in california.iteritems():
		if city_name in v:
			return k
	return 'none'

def lemmatize(doc_text):
	# doc_text = doc_text.encode('utf-8')
	blob = TextBlob(doc_text)
	temp = []
	for word in blob.words:
		temp.append(word.lemmatize())
	result = ' '.join(temp)
	return result

def prep_text(df):
	df.drop_duplicates(subset=['article_id'], inplace=True)
	df['city'] = df['release_text'].apply(lambda x: get_cities(x))
	df['state'] = df['city'].apply(lambda x: state_from_city(x))
	df['country'] = df['state'].apply(lambda x: country_from_state(x))
	df['region'] = df['state'].apply(lambda x: region_from_state(x))
	df['CA_region'] = df['city'].apply(lambda x: norcal_socal(x))
	df['release_text'] = df['release_text'].apply(lambda x: re.sub(r'\(?(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)\)?', '', x, flags=re.M))
	df['release_text'] = df['release_text'].apply(lambda x: lemmatize(x))
	return df

def print_top_words(model, feature_names, n_top_words):
	for topic_idx, topic in enumerate(model.components_):
		print("Topic #%d:" % topic_idx)
		print(" ".join([feature_names[i]
						for i in topic.argsort()[:-n_top_words - 1:-1]]))
	pass

new_stop_words = ['ha', "\'s", 'tt', 'ireach', "n\'t", 'wo', 'pv', 'tm', 'anite', 'rabichev', 'russell', '603', 'hana', 'atmel', 'radwin', 'se', 'doxee', 'lantto', 'publ', 'fpc1025', '855', 'il', '0344']
def make_stop_words(new_words_list):
	tfidf_temp = TfidfVectorizer(stop_words='english')
	stop_words = tfidf_temp.get_stop_words()
	result = list(stop_words) + new_words_list
	return result

def row_normalize_tfidf(np_matrix):
	return normalize(np_matrix, axis=1, norm='l1')

def get_topics(n_components=10, n_top_words=15, print_output=True, max_features=None):
	custom_stop_words = make_stop_words(new_stop_words)
	tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=custom_stop_words, max_features=max_features)
	tfidf = tfidf_vectorizer.fit_transform(release_texts)
	tfidf = row_normalize_tfidf(tfidf)
	nmf = NMF(n_components=n_components, random_state=1)
	# nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5)
	nmf.fit(tfidf)
	W = nmf.transform(tfidf)
	if print_output:
		print("\nTopics in NMF model:")
		tfidf_feature_names = tfidf_vectorizer.get_feature_names()
		print_top_words(nmf, tfidf_feature_names, n_top_words)
	return tfidf, nmf, W


y_mse = [4.1586519190079731e-07,
 4.0806056611869579e-07,
 4.0635976762741562e-07,
 4.0469315509174745e-07,
 4.031626753591851e-07,
 4.0198547133833227e-07,
 4.0069371691965978e-07,
 3.9989096055799519e-07,
 3.9873993330429839e-07,
 3.9755992608438446e-07,
 3.9668909965925691e-07,
 3.9565356967902645e-07,
 3.9478109096903595e-07,
 3.9395900472461631e-07,
 3.9306251341616297e-07,
 3.922508964184518e-07,
 3.9139907529199026e-07,
 3.9058720813092185e-07,
 3.8985442673013608e-07,
 3.8906441983276564e-07,
 3.8829184088224553e-07,
 3.8795241871128741e-07,
 3.8670916159757251e-07,
 3.8630316709902775e-07,
 3.8551159536125117e-07,
 3.8529296023215857e-07,
 3.8405309135809741e-07,
 3.8349524545279573e-07,
 3.8312979033889216e-07,
 3.8214679236464401e-07,
 3.8163433071433644e-07,
 3.812476213782855e-07,
 3.8037440645891213e-07,
 3.8003752994030476e-07,
 3.7912925338958192e-07,
 3.7869750967188198e-07,
 3.7801548354496874e-07,
 3.7721730759096696e-07,
 3.7667911984978913e-07,
 3.76411057692006e-07,
 3.7596695262405878e-07,
 3.7503280027874852e-07,
 3.74409880730487e-07,
 3.7385335317775083e-07,
 3.7370292291722861e-07,
 3.7324109586296963e-07,
 3.7239348121954796e-07,
 3.7220875451942766e-07,
 3.7157594400522471e-07,
 3.7092184827551988e-07]
x_range = np.arange(1,51)
y_mse_tts_round1 = [(1, 8.9955943708407939e-08),
 (2, 9.3778027752158838e-08),
 (3, 9.3544419976425533e-08),
 (4, 9.5515890208877072e-08), 
 (5, 9.6656460099495444e-08),
 (6, 9.5885980397053374e-08),
 (7, 9.5334438048771455e-08),
 (8, 9.7381168085049762e-08),
 (9, 9.7914818624726174e-08),
 (10, 9.8099492350250766e-08),
 (11, 9.785952599794063e-08),
 (12, 9.8796338877999243e-08),
 (13, 9.8042812898758145e-08),
 (14, 1.0014030681676387e-07),
 (15, 1.0061291988251698e-07),
 (16, 1.0121614033949377e-07),
 (17, 1.0100411437616885e-07)]

y_mse_tts_round2 = [(1, 8.9668696816688193e-08),
 (2, 9.354006756512067e-08),
 (3, 9.3912747795884238e-08),
 (4, 9.4640311785966076e-08),
 (5, 9.573811538314732e-08),
 (6, 9.6150569668228365e-08),
 (7, 9.6823212282909846e-08),
 (8, 9.7862620557820369e-08),
 (9, 9.7807808154723832e-08),
 (10, 9.9052272941647242e-08),
 (11, 9.7887749103781175e-08),
 (12, 9.9018721619947148e-08),
 (13, 1.0006480530688792e-07),
 (14, 1.0013447027686969e-07),
 (15, 1.0061047370872081e-07),
 (16, 1.0133475093254728e-07),
 (17, 1.0221336559036943e-07)]

def tfidf_traintestsplit(tfidf_sparse, test_size=0.2):
	# A = tfidf_sparse.toarray()
	A = tfidf_sparse
	total_entries = A.shape[0] * A.shape[1]
	# print 'total_entries =', total_entries
	train_size = 1. - test_size
	# print 'train_size =', train_size
	ones = np.ones(math.ceil(total_entries * train_size))
	zeros = np.zeros(math.floor(total_entries * test_size))
	# print 'ones length =', len(ones)
	# print 'zeros length =', len(zeros)
	r_temp = np.append(ones, zeros)
	np.random.shuffle(r_temp)
	# print 'total entries = ', total_entries
	# print 'r_temp shape = ', r_temp.shape
	# print 'tfidf shape = ', A.shape
	R = np.reshape(r_temp, (A.shape))
	R_flip = np.logical_not(R)

	R = sparse.csr_matrix(R)
	R_flip = sparse.csr_matrix(R_flip)
	# print 'R.shape =', R.shape
	# print 'R_flip.shape = ', R_flip.shape

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

def add_sentiment_to_df(df):
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

df_orig = mongo_to_df('press', 'test_master_1')
# subject_dict = make_subject_dict()
# industry_dict = make_industry_dict()

df = prep_text(df_orig)
df = make_dummies(df, 'industry', 'ind')
df = make_dummies(df, 'subject', 'subj')
df = add_sentiment_to_df(df)
release_texts = df['release_text']

tfidf, nmf, W = get_topics(n_components=8, print_output=True)


y_mse_tts1 = [(1, 9.0692358748435431e-08),
 (2, 9.3354952077386322e-08),
 (3, 9.3031976312425973e-08),
 (4, 9.4650382176899813e-08),
 (5, 9.5273169824867604e-08),
 (6, 9.5577693466519617e-08),
 (7, 9.6554850072711775e-08),
 (8, 9.6860185893522382e-08),
 (9, 9.777182337779741e-08),
 (10, 9.7198068654839282e-08),
 (11, 9.9179570616898986e-08),
 (12, 9.8435414794084085e-08),
 (13, 9.9729915763610886e-08),
 (14, 1.0057025454177779e-07),
 (15, 1.011253405956895e-07),
 (16, 1.0156648223618776e-07),
 (17, 1.0166909753736873e-07),
 (18, 1.020626092487737e-07),
 (19, 1.0299428837826783e-07),
 (20, 1.0324320688204009e-07),
 (21, 1.0397230634000827e-07),
 (22, 1.0494296231301062e-07),
 (23, 1.0511528348191881e-07),
 (24, 1.0498382799704761e-07),
 (25, 1.0491041860302454e-07)]
y_mse_tts2 = [(1, 8.9920178122599679e-08),
 (2, 9.241997791368585e-08),
 (3, 9.4295576867141506e-08),
 (4, 9.533736345055916e-08),
 (5, 9.5426036661397202e-08),
 (6, 9.5289189162429445e-08),
 (7, 9.7998731922648816e-08),
 (8, 9.7663454307892544e-08),
 (9, 9.7964122477303035e-08),
 (10, 9.7996687595693989e-08),
 (11, 9.8644352378446816e-08),
 (12, 9.9639944945584337e-08),
 (13, 9.9813067493227484e-08),
 (14, 9.9988651463755066e-08),
 (15, 1.0028747660339004e-07),
 (16, 1.010324851747725e-07),
 (17, 1.0135541559518328e-07),
 (18, 1.0166326732647471e-07),
 (19, 1.0212510311692001e-07),
 (20, 1.0335175430165075e-07),
 (21, 1.0414307426080312e-07),
 (22, 1.0442354753425517e-07),
 (23, 1.0454072194454142e-07),
 (24, 1.0556106084624903e-07),
 (25, 1.0522796378226479e-07)]
y_mse_tts3 = [(1, 9.0190656556150848e-08),
 (2, 9.3419199382905457e-08),
 (3, 9.3641722606712041e-08),
 (4, 9.4968516803528593e-08),
 (5, 9.5883421187193439e-08),
 (6, 9.592921845384083e-08),
 (7, 9.6864505710558395e-08),
 (8, 9.727706622775262e-08),
 (9, 9.8108857044629788e-08),
 (10, 9.7928402201978372e-08),
 (11, 9.7838525708802733e-08),
 (12, 9.8399942971601658e-08),
 (13, 9.9766135088303855e-08),
 (14, 1.0074306702218731e-07),
 (15, 1.0044105908534367e-07),
 (16, 1.0174876281476294e-07),
 (17, 1.0134171614323173e-07),
 (18, 1.0222361846948675e-07),
 (19, 1.0333398029797298e-07),
 (20, 1.0270346082971311e-07),
 (21, 1.0394143778363522e-07),
 (22, 1.0416462574741354e-07),
 (23, 1.0413727994908282e-07),
 (24, 1.0533395562565877e-07),
 (25, 1.0571181318772862e-07)]
y_mse_tts4 = [(1, 8.9426402292474586e-08),
 (2, 9.2812499443328966e-08),
 (3, 9.4178962705683908e-08),
 (4, 9.3529454132555265e-08),
 (5, 9.5259937378395284e-08),
 (6, 9.5398234324539e-08),
 (7, 9.6796578756592322e-08),
 (8, 9.6902515790323408e-08),
 (9, 9.7504546807785801e-08),
 (10, 9.8372676717703728e-08),
 (11, 9.8945361006732644e-08),
 (12, 9.9649163100839104e-08),
 (13, 1.001579002457881e-07),
 (14, 1.0025933672657527e-07),
 (15, 1.0050203493995102e-07),
 (16, 1.0085766885878172e-07),
 (17, 1.01291331696572e-07),
 (18, 1.0272144224249718e-07),
 (19, 1.0269227872693379e-07),
 (20, 1.0297934869180299e-07),
 (21, 1.0406891388370278e-07),
 (22, 1.0378104775994881e-07),
 (23, 1.0453246854535985e-07),
 (24, 1.0518066162586831e-07),
 (25, 1.0563110066463136e-07)]
y_mse_tts5 = [(1, 8.9784906512458701e-08),
 (2, 9.4017139054673279e-08),
 (3, 9.3263759676709954e-08),
 (4, 9.4753699049974596e-08),
 (5, 9.5724646303689679e-08),
 (6, 9.6092741545180527e-08),
 (7, 9.6071871839828413e-08),
 (8, 9.7147474628582662e-08),
 (9, 9.8043483473518766e-08),
 (10, 9.7990839922860176e-08),
 (11, 9.7951691494500065e-08),
 (12, 9.8446354003485468e-08),
 (13, 9.9746509166395962e-08),
 (14, 1.0044242578480452e-07),
 (15, 1.0118280976809748e-07),
 (16, 1.0175522041717573e-07),
 (17, 1.0192122384396983e-07),
 (18, 1.0154585412244006e-07),
 (19, 1.0347797937581035e-07),
 (20, 1.0312452836787713e-07),
 (21, 1.0438482224332105e-07),
 (22, 1.0414029129567104e-07),
 (23, 1.0330098207528865e-07),
 (24, 1.0535494217492949e-07),
 (25, 1.0568742565958652e-07)]

y_mse_tts10 = [(1, 5.9804501199301258e-06),
 (2, 5.821254643739843e-06),
 (3, 5.766218953246609e-06),
 (4, 5.7199711771657858e-06),
 (5, 5.6718387889089656e-06),
 (6, 5.6377733141578632e-06),
 (7, 5.6017566140428826e-06),
 (8, 5.5735224196191694e-06),
 (9, 5.5474448905578631e-06),
 (10, 5.5189443199788806e-06),
 (11, 5.4960655029213349e-06),
 (12, 5.4655819361993428e-06),
 (13, 5.4413503055362427e-06),
 (14, 5.4210088433473719e-06),
 (15, 5.3984693063497597e-06),
 (16, 5.3757246429503916e-06),
 (17, 5.3568462368963621e-06),
 (18, 5.3396455817331559e-06),
 (19, 5.3184194724046167e-06),
 (20, 5.2943135969209722e-06)]
y_mse_tts11 = [(1, 5.9804949654718744e-06),
 (2, 5.8220599231756129e-06),
 (3, 5.7669352798572765e-06),
 (4, 5.721051373441099e-06),
 (5, 5.6721014148023085e-06),
 (6, 5.6415280868561967e-06),
 (7, 5.6051597212861769e-06),
 (8, 5.5712561095011194e-06),
 (9, 5.5436978986037166e-06),
 (10, 5.5174499476335196e-06),
 (11, 5.4904367276549139e-06),
 (12, 5.4638932000150867e-06),
 (13, 5.4430467457990721e-06),
 (14, 5.4154081172626882e-06),
 (15, 5.3984359566533473e-06),
 (16, 5.3777020874099655e-06),
 (17, 5.3537907404764393e-06),
 (18, 5.3377713233491328e-06),
 (19, 5.3196610293697841e-06),
 (20, 5.3015054293696969e-06),
 (21, 5.2838120309014353e-06),
 (22, 5.2705793251229962e-06),
 (23, 5.2440764788367329e-06),
 (24, 5.230664040143438e-06),
 (25, 5.2123725904166548e-06),
 (26, 5.20286457272183e-06),
 (27, 5.1828138131626335e-06),
 (28, 5.1704224289779191e-06),
 (29, 5.1591413621288695e-06),
 (30, 5.1382160217699409e-06)]
y_mse_tts12 = [(1, 5.9803915054495377e-06),
 (2, 5.8203760941799226e-06),
 (3, 5.76692497852516e-06),
 (4, 5.7186839728147346e-06),
 (5, 5.6725990544451064e-06),
 (6, 5.6379227467064805e-06),
 (7, 5.6060251600985574e-06),
 (8, 5.5720664340247736e-06),
 (9, 5.5477532072667771e-06),
 (10, 5.5194478315525141e-06),
 (11, 5.4926864447920024e-06),
 (12, 5.4668420502123543e-06),
 (13, 5.4434040290619693e-06),
 (14, 5.4178238628488172e-06),
 (15, 5.3965779517362337e-06),
 (16, 5.3819782266895491e-06),
 (17, 5.3564482913704584e-06),
 (18, 5.3388783988807699e-06),
 (19, 5.3184185391527745e-06),
 (20, 5.3013184575348724e-06),
 (21, 5.2800578529329869e-06),
 (22, 5.2704201686970402e-06),
 (23, 5.2458516599284101e-06),
 (24, 5.236541175352102e-06),
 (25, 5.2147298301569377e-06),
 (26, 5.2070237772225423e-06),
 (27, 5.178942731771308e-06),
 (28, 5.1701871785686259e-06),
 (29, 5.1589371309865449e-06),
 (30, 5.1354607781934522e-06),
 (31, 5.121381649083185e-06),
 (32, 5.1130611824321666e-06),
 (33, 5.1001384566877792e-06),
 (34, 5.0817432304446553e-06),
 (35, 5.0657914747252949e-06),
 (36, 5.0567289608626965e-06),
 (37, 5.0375398480969516e-06),
 (38, 5.0279335532261991e-06),
 (39, 5.007561725412185e-06),
 (40, 4.9916302998904871e-06)]
y_mse_tts13 = [(1, 5.9804414953768397e-06),
 (2, 5.8211435873248396e-06),
 (3, 5.7657094903086123e-06),
 (4, 5.7207500112353087e-06),
 (5, 5.6714457073411649e-06),
 (6, 5.6423050269480776e-06),
 (7, 5.6056319771212346e-06),
 (8, 5.5730617754052197e-06),
 (9, 5.5444519388391911e-06),
 (10, 5.5147196362086441e-06),
 (11, 5.491622504157673e-06),
 (12, 5.4634038961210179e-06),
 (13, 5.4422670737945727e-06),
 (14, 5.4212156510813973e-06),
 (15, 5.3996459239381447e-06),
 (16, 5.3778185939454476e-06),
 (17, 5.3524634861810811e-06),
 (18, 5.3394001561368274e-06),
 (19, 5.3189295507502517e-06),
 (20, 5.2969381284819849e-06),
 (21, 5.2788273304920203e-06),
 (22, 5.2585612122018906e-06),
 (23, 5.2458906222775042e-06),
 (24, 5.2398589800761189e-06),
 (25, 5.2108585622372359e-06),
 (26, 5.1965375094440695e-06),
 (27, 5.181942928320974e-06),
 (28, 5.1661019927848074e-06),
 (29, 5.154735804515089e-06),
 (30, 5.1406148484476383e-06),
 (31, 5.1233343650902206e-06),
 (32, 5.1058389459184671e-06),
 (33, 5.0937038752912318e-06),
 (34, 5.0795604855211152e-06),
 (35, 5.0662744344858008e-06),
 (36, 5.045588987233031e-06),
 (37, 5.0335962253619149e-06),
 (38, 5.0249301758365423e-06),
 (39, 5.0103376168657348e-06),
 (40, 4.9954192448691646e-06),
 (41, 4.9854190314493042e-06),
 (42, 4.9723392971237485e-06),
 (43, 4.9613889671799784e-06),
 (44, 4.9462016907731778e-06),
 (45, 4.9365193768912092e-06),
 (46, 4.9194840802505751e-06),
 (47, 4.9017314872519055e-06),
 (48, 4.8942133942614221e-06),
 (49, 4.8834745288657827e-06),
 (50, 4.8665892947397338e-06)]

#y_mse_tts80: max_features=2500, test_size=0.2(folds=5)
y_mse_tts80 = [(1, 5.9804949654718744e-06),
 (2, 5.8220599231756129e-06),
 (3, 5.7669352798572765e-06),
 (4, 5.721051373441099e-06),
 (5, 5.6721014148023085e-06),
 (6, 5.6415280868561967e-06),
 (7, 5.6051597212861769e-06),
 (8, 5.5712561095011194e-06),
 (9, 5.5436978986037166e-06),
 (10, 5.5174499476335196e-06),
 (11, 5.4904367276549139e-06),
 (12, 5.4638932000150867e-06),
 (13, 5.4430467457990721e-06),
 (14, 5.4154081172626882e-06),
 (15, 5.3984359566533473e-06),
 (16, 5.3777020874099655e-06),
 (17, 5.3537907404764393e-06),
 (18, 5.3377713233491328e-06),
 (19, 5.3196610293697841e-06),
 (20, 5.3015054293696969e-06),
 (21, 5.2838120309014353e-06),
 (22, 5.2705793251229962e-06),
 (23, 5.2440764788367329e-06),
 (24, 5.230664040143438e-06),
 (25, 5.2135063058384106e-06),
 (26, 5.2026266571292814e-06),
 (27, 5.1807668285252553e-06),
 (28, 5.1631404695773333e-06),
 (29, 5.1510923977190903e-06),
 (30, 5.141344351891591e-06),
 (31, 5.1248375376639668e-06),
 (32, 5.1142205623010651e-06),
 (33, 5.0941001878428113e-06),
 (34, 5.0792525343085552e-06),
 (35, 5.0771034868917751e-06),
 (36, 5.0486447649264283e-06),
 (37, 5.0392498670304802e-06),
 (38, 5.027121320895019e-06),
 (39, 5.016371650180071e-06),
 (40, 4.995743857129671e-06),
 (41, 4.9817239611969861e-06),
 (42, 4.9664881452918929e-06),
 (43, 4.9542627274407072e-06),
 (44, 4.9441233771141354e-06),
 (45, 4.9255399161408498e-06),
 (46, 4.9128717255294369e-06),
 (47, 4.9051221210002515e-06),
 (48, 4.8888958851405915e-06),
 (49, 4.8854125866720033e-06),
 (50, 4.8677059697867326e-06),
 (51, 4.8491419127895731e-06),
 (52, 4.8444893032489089e-06),
 (53, 4.8359491007220257e-06),
 (54, 4.8225651072236912e-06),
 (55, 4.8220760640254266e-06),
 (56, 4.7978852604997822e-06),
 (57, 4.7989067742488415e-06),
 (58, 4.7884994892520704e-06),
 (59, 4.7681958803190127e-06),
 (60, 4.7610292507461749e-06),
 (61, 4.7415465905298852e-06),
 (62, 4.7356162862231034e-06),
 (63, 4.7178317294299016e-06),
 (64, 4.706878788107554e-06),
 (65, 4.6967440747031804e-06),
 (66, 4.6920669209447846e-06),
 (67, 4.6948788545266693e-06),
 (68, 4.6758514474735856e-06),
 (69, 4.6686201115949605e-06),
 (70, 4.6505620091690582e-06),
 (71, 4.6492622431940324e-06),
 (72, 4.6236689899040759e-06),
 (73, 4.630419294438639e-06),
 (74, 4.6120787792168723e-06),
 (75, 4.6039603633155458e-06),
 (76, 4.5921932943023555e-06),
 (77, 4.586112660867331e-06),
 (78, 4.5725469262583595e-06),
 (79, 4.5697077037386756e-06),
 (80, 4.5553279705869047e-06)]

#y_mse_tts100: max_features=2500, test_size=0.3333(folds=3)
y_mse_tts100 = [(1, 5.9843051804150497e-06),
 (2, 5.8380688221740067e-06),
 (3, 5.7915998183721377e-06),
 (4, 5.7585539168059717e-06),
 (5, 5.7118665796715357e-06),
 (6, 5.6832413240662698e-06),
 (7, 5.6474357683182926e-06),
 (8, 5.6172884967605943e-06),
 (9, 5.5981858695781749e-06),
 (10, 5.5683143425171122e-06),
 (11, 5.5597532350042762e-06),
 (12, 5.5297319639804819e-06),
 (13, 5.5085389002884943e-06),
 (14, 5.4913498731839419e-06),
 (15, 5.473247836396042e-06),
 (16, 5.4611105714301085e-06),
 (17, 5.4469421933882837e-06),
 (18, 5.4308552221872717e-06),
 (19, 5.404576542467689e-06),
 (20, 5.3848982668491524e-06),
 (21, 5.3755439653405985e-06),
 (22, 5.356330414414384e-06),
 (23, 5.340241582297497e-06),
 (24, 5.3337292731422635e-06),
 (25, 5.3280196529913633e-06),
 (26, 5.291773208642306e-06),
 (27, 5.2885862334775284e-06),
 (28, 5.2701359465267184e-06),
 (29, 5.2577468426406804e-06),
 (30, 5.2439210992543101e-06),
 (31, 5.2354034105593494e-06),
 (32, 5.2269113598520543e-06),
 (33, 5.2197219262135475e-06),
 (34, 5.2067022663543131e-06),
 (35, 5.190200814297149e-06),
 (36, 5.1898374463318984e-06),
 (37, 5.1713527039154726e-06),
 (38, 5.1557542806525737e-06),
 (39, 5.1494044686128257e-06),
 (40, 5.1312748177233788e-06),
 (41, 5.120047357592568e-06),
 (42, 5.1303596978192281e-06),
 (43, 5.1039115254345073e-06),
 (44, 5.101764740761389e-06),
 (45, 5.0710281741183815e-06),
 (46, 5.0785445542561074e-06),
 (47, 5.065410973435611e-06),
 (48, 5.0724009259608663e-06),
 (49, 5.0374374634307759e-06),
 (50, 5.038861067293328e-06)]


# grid_search_nmf_ncomponents(tfidf, 3, 1, 50, y_mse_tts100)
# grid_search_nmf_ncomponents(tfidf, 5, 1, 30, y_mse_tts11)
# grid_search_nmf_ncomponents(tfidf, 5, 1, 40, y_mse_tts12)
# grid_search_nmf_ncomponents(tfidf, 5, 1, 50, y_mse_tts13)
# grid_search_nmf_ncomponents(tfidf, 5, 1, 50, y_mse_tts14)

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







