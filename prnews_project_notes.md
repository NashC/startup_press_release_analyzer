PR Newswire Project Notes

Next Steps:
- +++DONE+++:Merge all smaller MongoDB collections into one master collection with all press releases
- Strip from release text - check out https://regex101.com/ for testing
    + +++DONE+++: strip website URL's
- +++DONE+++: Industries/Subjects - Add dummies to DF
- +++DONE+++:Run topic clustering using these labels
- +++DONE+++:sentiment analysis: spacy.io (https://spacy.io/), nltk.sentiment (http://www.nltk.org/) (http://www.nltk.org/api/nltk.sentiment.html#module-nltk.sentiment.sentiment_analyzer), TextBlob
- +++DONE+++:Call w PR Newswire people to get more API calls per day and larger date range access
- +++DONE+++:Make dictionaries out of Subjects and Industries
- +++DONE+++: Lemmatize words prior to tfidf or other model inputs
- +++DONE+++: download press releases for organizations not classified as 'company'
- +++DONE+++: do value_counts() on 'source' of the releases to see top companies with releases
- +++DONE+++: Check out Word2Vec: https://code.google.com/p/word2vec/
- +++DONE+++: tfidf sparse matrix: normalize across rows so they sum to one

- +++DONE+++: strip 'city, state' from beginning of each release and add to DF
	+ add city, state, country, region
- Do EDA/Plotting on meta data vs topic strength
- +++DONE+++: Manual grid search for optimal 'n_components' in NMF model
- Calculate topic strength by meta item (industry, subject, location, other)
- Build web app
	+ stack?
		* Flask
		* Bootstrap
		* Heroku
		* Plotly
	+ NLP stats to display (polarity, subjectivity, explanation of each stat, word count, standalone stats vs compared to press release corpus, parts of speech, sentence count, sentence sentiment, vocab complexity/reading level)
- draft presentation and slides
	+ Questions to ask:
		* location
		* industry
		* questions that a non-DS person would ask, not model optimization/n_components/etc
	+ visualizations
- draft ReadMe for github repo describing project, process and findings


- If time permites
	+ Word Clouds: about top words per topic
	+ Try to get more press release data
	+ D3 visualizations
	+ Install 'geograpy' library to get city, state names

-Project Deliverables
Model
Powerpoint Presentation Slides
	description
	visualizations
Web App with live update visualizations
GitHub repo with extensive ReadMe explaining entire project
	Project description
	data acquisition pipeline
	Model construction
	Explanation of web app use
Code with extensive comments explaining each step


Potential Sources for more press releases
- Business Wire
- Marketwired
- Factiva
- LexisNexus
- Newsbank



PR Newswire API Subject and Industry Codes

Subjects
[u'ACC', u'Accounting News, Issues']
[u'TNM', u'Acquisitions, Mergers, Takeovers']
[u'BCY', u'Bankruptcy']
[u'RTG', u'Bond/Stock Ratings']
[u'CON', u'Contracts']
[u'DIV', u'Dividends']
[u'ERN', u'Earnings']
[u'ERP', u'Earnings Projects or Forecasts']
[u'ECO', u'Economic News, Trends and Analysis']
[u'FNC', u'Financing Agreements']
[u'JVN', u'Joint Ventures']
[u'LIC', u'Licensing/Marketing Agreements']
[u'PDT', u'New Products/Services']
[u'OFR', u'Offerings']
[u'DSC', u'Oil/Gas Discoveries']
[u'OTC', u'OTC/SmallCap IRW']
[u'PER', u'Personnel Announcements']
[u'RLE', u'Real Estate Transactions']
[u'RCN', u'Restructuring/Recapitalizations']
[u'SLS', u'Sales Reports']
[u'SRP', u"Shareholders' Rights Plans"]
[u'LEG', u'Federal and State Legislation']
[u'EXE', u'Federal Executive Branch, Agency News']
[u'CPN', u'Political Campaigns']
[u'LBR', u'Labor/Union news']
[u'BLK', u'African-American News']
[u'HSP', u'Hispanic-Oriented News']
[u'LAW', u'Legal Issues']
[u'AWD', u'Awards']
[u'NPT', u'Not for Profit']
[u'TDS', u'Tradeshow News']
[u'CCA', u'Conference Call Announcements']
[u'CHI', u'Children-Related News']
[u'WOM', u'Women-related News']
[u'VEN', u'Venture Capital']
[u'BFA', u'Broadcast Feed Announcement']
[u'ASI', u'Asian-Related News']
[u'EGV', u'European Government']
[u'MAV', u'Media Advisory/Invitation']
[u'SVY', u'Surveys, Polls & Research']
[u'INO', u'Investments Opinions']
[u'ZHA', u'Xinhua']
[u'FOR', u'Foreign policy/International affairs']
[u'POL', u'Domestic Policy']
[u'TRD', u'Trade Policy']
[u'REL', u'Religion']
[u'STS', u'Stock Split']
[u'PET', u'Animals/Pets']
[u'TRI', u'Clinical Trials/Medical Discoveries']
[u'RCY', u'Conservation/Recycling']
[u'CSR', u'Corporate Social Responsibility']
[u'FDA', u'FDA Approval']
[u'DIS', u'Handicapped/Disabled']
[u'LGB', u'Lesbian/Gay/Bisexual']
[u'NTA', u'Native American']
[u'PLW', u'Patent Law']
[u'RCL', u'Product Recalls']
[u'PSF', u'Public Safety']
[u'SCZ', u'Senior Citizens']
[u'SBS', u'Small Business Services']
[u'STP', u'U.S. State Policy News']
[u'VET', u'Veterans']
[u'VDM', u'MultiVu Video']
[u'ADM', u'MultiVu Audio']
[u'PHM', u'MultiVu Photo']
[u'BCM', u'Broadcast Minute']
[u'CXP', u'Corporate Expansion']
[u'ENI', u'Environmental Issues']
[u'ENP', u'Environmental Policy']
[u'SRI', u'Socially Responsible Investing']
[u'VNR', u'Video News Releases']
[u'ANW', u'Animal Welfare']
[u'AVO', u'Advocacy Group Opinion']
[u'OBI', u'Obituaries']
[u'FEA', u'Features']

Industries
[u'ADV', u'Advertising ']
[u'ARO', u'Aerospace/Defense']
[u'AGR', u'Agriculture']
[u'AIR', u'Airlines/Aviation']
[u'ART', u'Art']
[u'AUT', u'Automotive']
[u'FIN', u'Banking/Financial Services']
[u'BIO', u'Biotechnology']
[u'BKS', u'Books']
[u'CHM', u'Chemical']
[u'CPR', u'Computer/ Electronics']
[u'NET', u'Networks']
[u'HRD', u'Computer Hardware']
[u'STW', u'Computer Software']
[u'CST', u'Construction/Building']
[u'CSE', u'Consumer Electronics']
[u'EDU', u'Education']
[u'EPM', u'Electronics Performance Measurement']
[u'ECM', u'Electronic Commerce']
[u'ENT', u'Entertainment']
[u'ENV', u'Environmental Products & Services']
[u'FAS', u'Fashion']
[u'FLM', u'Film and Motion Picture']
[u'FOD', u'Food & Beverages']
[u'CNO', u'Gambling/Casinos']
[u'HEA', u'Health Care/Hospitals']
[u'HOU', u'Household/Consumer/Cosmetics']
[u'INS', u'Insurance']
[u'ITE', u'Internet Technology']
[u'LEI', u'Leisure & Tourism']
[u'MAC', u'Machinery']
[u'MAG', u'Magazines']
[u'MAR', u'Maritime/Shipbuilding']
[u'MTC', u'Medical/Pharmaceuticals']
[u'MNG', u'Mining/Metals']
[u'MLM', u'Multimedia/Internet']
[u'MUS', u'Music']
[u'MFD', u'Mutual Funds']
[u'OFP', u'Office Products']
[u'OIL', u'Oil/Energy']
[u'PAP', u'Paper/Forest Products/Containers']
[u'PEL', u'Peripherals']
[u'PUB', u'Publishing/Information Services']
[u'RAD', u'Radio']
[u'RLT', u'Real Estate']
[u'REA', u'Retail']
[u'RST', u'Restaurants']
[u'SPT', u'Sports']
[u'SUP', u'Supermarkets']
[u'SPM', u'Supplementary Medicine']
[u'TLS', u'Telecommunications Industry']
[u'TVN', u'Television']
[u'TEX', u'Textiles']
[u'TOB', u'Tobacco']
[u'TRN', u'Transportation/Trucking/Railroad']
[u'TRA', u'Travel']
[u'UTI', u'Utilities']
[u'Feature', u'Features']
[u'HTS', u'High Tech Security']
[u'ECP', u'Electronic Components']
[u'EDA', u'Electronic Design Automation']
[u'SEM', u'Semiconductors']
[u'HED', u'Higher Education']
[u'ALC', u'Beers, Wines and Spirits']
[u'BIM', u'Biometrics']
[u'GAM', u'Electronic Gaming']
[u'HMS', u'Homeland Security']
[u'IDC', u'Infectious Disease Control']
[u'MEN', u'Mobile Entertainment']
[u'NAN', u'Nanotechnology']
[u'WRK', u'Workforce Management/Human Resources']
[u'AIF', u'Air Freight']
[u'ALT', u'Alternative Energies']
[u'ANW', u'Animal Welfare   ']
[u'ATL', u'Amusement Parks and Tourist Attractions']
[u'BEV', u'Beverages']
[u'BRI', u'Bridal Services']
[u'CPC', u'Cosmetics and Personal Care']
[u'CRL', u'Commercial Real Estate']
[u'DEN', u'Dentistry']
[u'ENS', u'Environmental Products & Services']
[u'EUT', u'Electrical Utilities']
[u'FRN', u'Furniture and Furnishings']
[u'GAS', u'Gas']
[u'HHP', u'Household Products']
[u'HIN', u'Health Insurance ']
[u'HMI', u'Home Improvements']
[u'HRT', u'Hotels and Resorts']
[u'HSC', u'Home Schooling']
[u'HVA', u'HVAC']
[u'JWL', u'Jewelry']
[u'MCT', u'Machine Tools, Metalworking and Metallurgy']
[u'MEQ', u'Medical Equipment']
[u'MIN', u'Mining']
[u'MNH', u'Mental Health']
[u'NAB', u'Non-Alcoholic Beverages']
[u'ORF', u'Organic Food']
[u'ORL', u'Overseas Real Estate (non-US) ']
[u'OUT', u'Outsourcing Businesses']
[u'PAV', u'Passenger Aviation']
[u'PHA', u'Pharmaceuticals']
[u'PRM', u'Precious Metals']
[u'RFI', u'RFID (Radio Frequency ID) Applications & Tech']
[u'RIT', u'Railroads & Intermodal Transporation']
[u'RRL', u'Residential Real Estate']
[u'SMD', u'Social Media']
[u'SPE', u'Sports Equipment & Accessories']
[u'SSE', u'Sporting Events']
[u'SWB', u'Semantic Web']
[u'TCS', u'Telecommunications Carriers and Services']
[u'TEQ', u'Telecommunications Equipment']
[u'TRT', u'Trucking and Road Transportation']
[u'VIP', u'VoIP (Voice over Internet Protocol)']
[u'WEB', u'Web site']
[u'WIC', u'Wireless Communications']
[u'WUT', u'Water Utilities']
[u'GRE', u'Green Technology']
[u'OTC', u'OTC/SmallCap']
[u'SRI', u'Socially Responsible Investing']
[u'TOY', u'Toys']
[u'BRD', u'Broadcast Technology']
