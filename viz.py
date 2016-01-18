#viz.py



def merge_df_W(df, W, topic_list):
	#normalize W matrix, convert to pandas dataframe, assign columns to latent topic names
	W = normalize(W, axis=1, norm='l1')
	W = pd.DataFrame(W)
	W.columns = [topic for topic in topic_list]
	
	#drop/reset indexes, join big press release dataframe with W dataframe
	df.reset_index(drop=True, inplace=True)
	W.reset_index(drop=True, inplace=True)
	df_W = pd.concat([df,W], axis=1)
	
	return df_W

def topic_strength_by_location(df_W, location_col, norm='location'):
	#groupby location name and drop rows without location info
	df_topics_loc = df_W.groupby([location_col]).sum().iloc[:,-W.shape[1]:]
	df_topics_loc = df_topics_loc[df_topics_loc.index != 'none']
	
	#normalize over 'location' or 'topic'
	if norm == 'location':
		np_norm = normalize(df_topics_loc.values, axis=1, norm='l1')
	elif norm == 'topic':
		np_norm = normalize(df_topics_loc.values, axis=0, norm='l1')
		
	#convert to pandas dataframe, assign index and columns
	df_norm = pd.DataFrame(np_norm)
	df_norm.index = df_topics_loc.index
	df_norm.columns = df_topics_loc.columns
	
	return df_norm

def traces_topicRegion(df_norm, x='regions'):
    traces = []
    if x == 'regions':
        for i in xrange(df_norm.shape[1]):
            temp = go.Bar(
                x=df_norm.index,
                y=df_norm.iloc[:,i].values,
                name=df_norm.columns[i]
                )
            traces.append(temp)
    if x == 'topics':
        for i in xrange(df_norm.shape[0]):
            temp = go.Bar(
                x=df_norm.columns,
                y=df_norm.iloc[i,:].values,
                name=df_norm.index[i]
                )
            traces.append(temp)
    return traces



def main():
	latent_topics = {
	'Insurance': ['insurance', 'quote', 'car', 'auto', 'coverage', 'driver', 'online', 'comparing', 'website', 'agency', 'client', 'plan', 'policy', 'multiple', 'carrier'],
	'Enterprise Software': ['data', 'cloud', 'service', 'solution', 'customer', 'business', 'software', 'enterprise', 'management', 'network', 'security', 'technology', 'platform', 'application', 'provider'],
	'Healthcare/Biotech': ['health', 'patient', 'care', 'healthcare', 'medical', 'hospital', 'clinical', 'cancer', 'physician', 'elsevier', 'treatment', 'research', 'disease', 'drug', 'program'],
	'Online Education': ['content', 'video', 'student', 'social', 'game', 'digital', 'marketing', 'medium', 'online', 'brand', 'learning', 'company', 'school', 'new', 'platform'],
	'Renewable Energy': ['solar', 'energy', 'power', 'project', 'renewable', 'module', 'utility', 'sunpower', 'electricity', 'skypower', 'battery', 'smart', 'technology', 'electric', 'home'],
	'3D Printing': ['epson', 'printer', 'label', 'print', 'printing', 'pos', 'color', 'america', 'projector', 'ink', '3d', 'trademark', 'ricoh', 'seiko', 'registered'],
	'Mobile/Apps': ['app', 'mobile', 'user', 'device', 'new', 'android', 'apps', 'store', 'iphone', 'windows', 'available', 'feature', 'apple', 'vehicle', 'phone'],
	'Finance': ['statement', 'looking', 'forward', 'risk', 'company', 'result', 'uncertainty', 'securities', 'release', 'future', 'factor', 'actual', 'materially', 'differ', 'exchange']
	}

	topic_list = ['Insurance', 'Enterprise Software', 'Healthcare/Biotech', 'Online Education', 'Renewable Energy', '3D Printing', 'Mobile/Apps', 'Finance']

	df_W = merge_df_W(df, W, topic_list)
	df_topicRegion_normRegion = topic_strength_by_location(df_W, 'region', norm='location')
	df_topicRegion_normTopic = topic_strength_by_location(df_W, 'region', norm='topic')

	x_regions = traces_topicRegion(df_topicRegion_normRegion, x='regions')
	x_topics = traces_topicRegion(df_topicRegion_normTopic, x='topics')

if __name__ == '__main__':
	main()
