#app.py

import os
# import the Flask class from the flask module
from flask import Flask, render_template, redirect, url_for, request, session, flash, g, Markup
from pymongo import MongoClient
from textblob import TextBlob
import plotly.plotly as py
import numpy as np
import plotly.graph_objs as go
from plotly import tools
import plotly.tools as tls
py.sign_in(os.environ.get('PLOTLY_USERNAME', ''), os.environ.get('PLOTLY_API_KEY', ''))


# create the application object
app = Flask(__name__)
app.database = 'press'
app.collection = 'mini1'

def textblob_sentiment(text):
	'''
		Use TextBlog package to get sentiment values for each press release text
		
		INPUT:
		- text: release text string

		OUTPUT:
		- (polarity, subjectivity): tuple containing the sentiment values
	'''
	#create TextBlog object and use inherent methods to get polarity and subjectivity
	blob = TextBlob(text)
	polarity = blob.sentiment.polarity
	subjectivity = blob.sentiment.subjectivity
	return (polarity, subjectivity)

def make_plot(pol_base, subj_base):
	avg_polarity = 0.175
	avg_subjectivity = 0.444
	
	pol_vs_avg = pol_base - avg_polarity
	subj_vs_avg = subj_base - avg_subjectivity

	y_base = [pol_base, subj_base]
	x_base = ['Polarity', 'Subjectivity']

	y_relative = [pol_vs_avg, subj_vs_avg]
	x_relative = ['Polarity', 'Subjectivity']

	trace1 = go.Bar(
		x=y_base,
		y=x_base,
		name='Baseline Sentiment',
		orientation='h',
		yaxis='y',
	)

	trace2 = go.Bar(
		x=y_relative,
		y=x_base,
		name='Relative Sentiment',
		orientation='h',
		yaxis='y2',
	)

	fig = tls.make_subplots(rows=2, cols=1, subplot_titles=('Sentiment of Your Press Release', 'Relative Sentiment vs Avg Startup Press Release'))
	fig.append_trace(trace1, 1, 1)
	fig.append_trace(trace2, 2, 1)
	
	fig['layout'].update(
	 	showlegend=False,
	 	plot_bgcolor='rgb(217, 217, 217)',
	)
	plot_url = py.plot(fig, filename='flask_test2')
	return plot_url


def make_stock_plot():
	y_saving = [1.3586, 2.2623000000000002, 4.9821999999999997, 6.5096999999999996,
			7.4812000000000003, 7.5133000000000001, 15.2148, 17.520499999999998
			]
	y_net_worth = [93453.919999999998, 81666.570000000007, 69889.619999999995,
				   78381.529999999999, 141395.29999999999, 92969.020000000004,
				   66090.179999999993, 122379.3]
	x_saving = ['Japan', 'United Kingdom', 'Canada', 'Netherlands',
				'United States', 'Belgium', 'Sweden', 'Switzerland']
	x_net_worth = ['Japan', 'United Kingdom', 'Canada', 'Netherlands',
				   'United States', 'Belgium', 'Sweden', 'Switzerland'
				   ]
	trace0 = go.Bar(
		x=y_saving,
		y=x_saving,
		marker=dict(
			color='rgba(50, 171, 96, 0.6)',
			line=dict(
				color='rgba(50, 171, 96, 1.0)',
				width=1,
			),
		),
		name='Household savings, percentage of household disposable income',
		orientation='h',
	)
	trace1 = go.Scatter(
		x=y_net_worth,
		y=x_net_worth,
		mode='lines+markers',
		line=dict(
			color='rgb(128, 0, 128)',
		),
		name='Household net worth, Million USD/capita',
	)
	layout = dict(
		title='Household savings & net worth for eight OECD countries',
		yaxis1=dict(
			showgrid=False,
			showline=False,
			showticklabels=True,
			domain=[0, 0.85],
		),
		yaxis2=dict(
			showgrid=False,
			showline=True,
			showticklabels=False,
			linecolor='rgba(102, 102, 102, 0.8)',
			linewidth=2,
			domain=[0, 0.85],
		),
		xaxis1=dict(
			zeroline=False,
			showline=False,
			showticklabels=True,
			showgrid=True,
			domain=[0, 0.42],
		),
		xaxis2=dict(
			zeroline=False,
			showline=False,
			showticklabels=True,
			showgrid=True,
			domain=[0.47, 1],
			side='top',
			dtick=25000,
		),
		legend=dict(
			x=0.029,
			y=1.038,
			font=dict(
				size=10,
			),
		),
		margin=dict(
			l=100,
			r=20,
			t=70,
			b=70,
		),
		width=600,
		height=600,
		paper_bgcolor='rgb(248, 248, 255)',
		plot_bgcolor='rgb(248, 248, 255)',
	)

	annotations = []

	y_s = np.round(y_saving, decimals=2)
	y_nw = np.rint(y_net_worth)

	# Adding labels
	for ydn, yd, xd in zip(y_nw, y_s, x_saving):
		# labeling the scatter savings
		annotations.append(dict(xref='x2', yref='y2', y=xd, x=ydn - 20000,
									  text='{:,}'.format(ydn) + 'M',
									  font=dict(family='Arial', size=12,
												color='rgb(128, 0, 128)'),
									  showarrow=False,))
		# labeling the bar net worht
		annotations.append(dict(xref='x1', yref='y1', y=xd, x=yd + 3,
									  text=str(yd) + '%',
									  font=dict(family='Arial', size=12,
												color='rgb(50, 171, 96)'),
									  showarrow=False,))
	# Source
	annotations.append(dict(xref='paper', yref='paper', x=-0.2, y=-0.109,
								  text='OECD "' +
								  '(2015), Household savings (indicator), ' +
								  'Household net worth (indicator). doi: ' +
								  '10.1787/cfc6f499-en (Accessed on 05 June 2015)',
								  font=dict(family='Arial', size=10,
											color='rgb(150,150,150)'),
								  showarrow=False,))

	layout['annotations'] = annotations

	# Creating two subplots
	fig = tools.make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
							  shared_yaxes=False, vertical_spacing=0.001)

	fig.append_trace(trace0, 1, 1)
	fig.append_trace(trace1, 1, 2)

	fig['layout'].update(layout)
	plot_url = py.plot(fig, filename='flask_test1')
	return plot_url

# use decorators to link the function to a url
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/analyzer', methods = ['POST'])
def analyzer():
	text = request.form['text']

	avg_polarity = 0.175
	avg_subjectivity = 0.444
	blob = TextBlob(text)
	words = blob.words
	sentences = blob.sentences
	num_words = len(words)
	num_sentences = len(sentences)
	pol_base = blob.sentiment.polarity
	subj_base = blob.sentiment.subjectivity
	pol_vs_avg = pol_base - avg_polarity
	subj_vs_avg = subj_base - avg_subjectivity

	print 'Word Count: ', num_words
	print 'Sentence Count: ', num_sentences
	print 'Baseline Polarity: ', pol_base
	print 'Polarity versus average press release: ', pol_vs_avg
	print 'Baseline Subjectivity: ', subj_base
	print 'Subjectivity versus average press release: ', subj_vs_avg
	graph_url = make_plot(pol_base, subj_base)
	graph_html = tls.get_embed(graph_url)
	graph_html = Markup(graph_html)
	return render_template('index.html', graph_html=graph_html, num_words=num_words, num_sentences=num_sentences)

@app.route('/welcome')
def welcome():
	return render_template('welcome.html')  # render a template

@app.route('/database')
def database():
	g.db = connect_db(app.database, app.collection)
	coll = g.db
	posts = [dict(source=row['source'], headline=row['headline'], release_text=row['release_text']) for row in coll.find()]
	return render_template('database.html', posts=posts)

def connect_db(db_name, coll_name):
	client = MongoClient('127.0.0.1', 27017)
	db = client[db_name]
	coll = db[coll_name]
	return coll

# start the server with the 'run()' method
if __name__ == '__main__':
	app.run(debug=True)