#prnewswire_api.py

import numpy as np
import pandas as pd
import requests
import json
from bs4 import BeautifulSoup
import crunchbase_odm as cb
from pprint import pprint
import pymongo
from pymongo import MongoClient


def get_master_data():
	masterData_url = 'http://api.prnewswire.com/releases/getMasterData'
	masterData_payload = {
		'listName': 'GEOGRAPHY'
	}
	r = requests.get(masterData_url, params=masterData_payload)
	pass

# csv_file = '../crunchbase/cb_odm_csv/organizations.csv'
def make_orgs_df(filename):
	df = pd.read_csv(filename)
	df_filt = df[(df['primary_role'] == 'company') & (df['location_country_code'] == 'USA') & (df['location_region'].isin(['California', 'New York', 'Washington', 'Colorado', 'Texas', 'Connecticut', 'Massachusetts', 'New Jersey']))]
	df_filt_all_locations = df[df['primary_role'] == 'company']
	df_filt_all_locations.drop_duplicates(subset=['name'], inplace=True)
	df_not_company = df[df['primary_role'] == 'investor']
	return df_not_company

# df = make_orgs_df(csv_file)

def make_one_org_query_str(df, start, finish):
	org_query_str = ''
	for i in xrange(start, finish):
		if i >= df.shape[0]:
			continue
		string = '"' + str(df['name'].iloc[i]) + '"'
		if i == start:
			org_query_str += string
		else:
			org_query_str += ' OR ' + string
	return org_query_str

def make_all_org_query_strings(df, total, request_size, start=0, finish=49):
	result = []
	requests_needed = int(total/request_size)
	# start = 0
	# finish = 49
	for i in xrange(requests_needed):
		result.append(make_one_org_query_str(df, start, finish))
		start += 50
		finish += 50
	return result

# all_org_query_strings = make_all_org_query_strings(df, len(df['name'].unique()), 50)

# org_query_str = make_one_org_query_str(df, 2100,2149)

def make_date_strings(start, end):
	result = []
	for year in range(start, end + 1):
		start_date_1 = '01/01/' + str(year) + ' 01:01'
		end_date_1 = '30/04/' + str(year) + ' 01:01'
		start_date_2 = '01/05/' + str(year) + ' 01:01'
		end_date_2 = '31/08/' + str(year) + ' 01:01'
		start_date_3 = '01/09/' + str(year) + ' 01:01'
		end_date_3 = '31/12/' + str(year) + ' 01:01'
		temp1 = [start_date_1, end_date_1]
		temp2 = [start_date_2, end_date_2]
		temp3 = [start_date_3, end_date_3]
		result.append(temp1)
		result.append(temp2)
		result.append(temp3)
	return result

# date_strings = make_date_strings(2015, 2015)

def prep_payload(start_date, end_date, page_index, org_query_str, fixed_start_date=False):
	subject_query_str = 'subject:(TNM OR CON OR FNC OR LIC OR PDT OR SLS OR VEN OR FEA)'
	industry_query_str = 'industry:(BIO OR CPR OR NET OR HRD OR STW OR CSE OR ITE OR MLM OR TLS OR HTS OR SEM OR BIM OR MEN OR NAN OR SMD OR TCS OR TEQ OR VIP WEB OR WIC OR GRE OR BRD)'
	geo_query_str = 'geography:(USA)'
	if fixed_start_date:
		payload = {
		'apikey': 'ded6410542df3e0e',
		'query': '(content:(' + org_query_str + ') + companies:(' + org_query_str + ') + ' + subject_query_str +' + '+ industry_query_str +' + '+ geo_query_str +' + language:en)',
		'pageSize': 100,
		'pageIndex': page_index,
		'startDate': '12/07/2015'
		# 'endDate': end_date
		}
	else:
		payload = {
		'apikey': 'ded6410542df3e0e',
		'query': '(content:(' + org_query_str + ') + companies:(' + org_query_str + ') + ' + subject_query_str +' + '+ industry_query_str +' + '+ geo_query_str +' + language:en)',
		'pageSize': 100,
		'pageIndex': page_index,
		'startDate': start_date,
		'endDate': end_date
		}
	return payload

# payld = prep_payload(date_strings[2][0], date_strings[2][1], 1, org_query_str, fixed_start_date=True)
# print date_strings[2][0]
# print date_strings[2][1]

def prep_request(payload):
	getReleases_url = 'http://api.prnewswire.com/releases/version01/getReleases'
	r = requests.get(getReleases_url, params=payload)
	status = r.json()['feed']['statusCode']
	totalResults = r.json()['feed']['totalResults']
	# responses = r.json()['releases']['release']
	search_params = r.json()['feed']['filter']
	return r, status, totalResults, search_params

# r, status, totalResults, search_params = prep_request(payld)

# responses = r.json()['releases']['release']
# print responses[0]
# print status
# print totalResults

def start_mongo(db_name, coll_name):
	client = MongoClient('127.0.0.1', 27017)
	db = client[db_name]
	coll = db[coll_name]
	return coll

# coll = start_mongo('pr', 'test3')

def mongo_insert_responses(responses, search_params, coll):
	docs = []
	for each in responses:
		html_text = each['releaseContent']
		if html_text == None or len(html_text) == 0:
			continue
		soup = BeautifulSoup(html_text, 'html.parser')
		release_text = soup.get_text()

		doc = {
			'article_id': each['articleId'],
			'date': each['date'],
			'source': each['source'],
			'release_text': release_text,
			'htmltext': each['releaseContent'],
			'headline': each['headline'],
			'subHeadline': each['subHeadline'],
			'link': each['link'],
			'industry': each['industry'],
			'subject': each['subject'],
			'search_params': search_params
		}
		docs.append(doc)
	coll.insert_many(docs)
	pass

# mongo_insert_responses(responses)

def mini(start_index):
	orgs_csv_file = '../crunchbase/cb_odm_csv/organizations.csv'
	df_orgs = make_orgs_df(orgs_csv_file)
	print 'DataFrame successfully created'
	coll = start_mongo('press', 'not_company_1')
	print 'Mongo DB started'
	org_name_sets = make_all_org_query_strings(df=df_orgs, total=df_orgs.shape[0], request_size=50, start=start_index, finish=(start_index+49))
	print 'Org Name Sets successfully created'
	# date_strings = make_date_strings(2014, 2015)
	for index, org_set in enumerate(org_name_sets):
		print 'Org Set #', start_index
		start_index += 50
		payload = prep_payload(start_date='12/07/2015', end_date='07/01/2016', page_index=1, org_query_str=org_set, fixed_start_date=True)
		r, status, totalResults, search_params = prep_request(payload)

		if totalResults == 0:
			continue
		responses = r.json()['releases']['release']
		mongo_insert_responses(responses, search_params, coll)
		if totalResults <= 100:
			continue
		else:
			pages_needed = int(totalResults/100) + 1
			for page_ind in xrange(2, pages_needed + 1):
				payload = prep_payload(start_date='12/07/2015', end_date='07/01/2016', page_index=page_ind, org_query_str=org_set, fixed_start_date=True)
				r, status, totalResults, search_params = prep_request(payload)
				if totalResults == 0:
					continue
				responses = r.json()['releases']['release']
				mongo_insert_responses(responses, search_params, coll)
	pass

def testOne():
	csv_file = '../crunchbase/cb_odm_csv/organizations.csv'
	df = make_orgs_df(csv_file)
	org_query_str = make_one_org_query_str(df,42600,42650)
	payld = prep_payload(start_date='09/07/2015', end_date='04/01/2016', page_index=1, org_query_str=org_query_str, fixed_start_date=True)
	print payld
	r, status, totalResults, search_params = prep_request(payld)
	responses = r.json()['releases']['release']
	print responses[0]
	print status
	print totalResults

def main():
	orgs_csv_file = '../crunchbase/cb_odm_csv/organizations.csv'
	df_orgs = make_orgs_df(orgs_csv_file)
	coll = start_mongo('press', 'test1')
	org_name_sets = make_all_org_query_strings(df_orgs, 500, 50)
	date_strings = make_date_strings(2014, 2015)
	for org_set in org_name_sets:
		for date_set in date_strings:
			# print date_set[0]
			# print date_set[1]
			payload = prep_payload(start_date=date_set[0], end_date=date_set[1], page_index=1, org_query_str=org_set)
			# print pprint(payload)
			r, status, totalResults, responses, search_params = prep_request(payload)
			mongo_insert_responses(responses)
			if totalResults <= 100:
				continue
			else:
				pages_needed = int(totalResults/100) + 1
				for page_ind in xrange(1, pages_needed + 1):
					payload = prep_payload(start_date=date_set[0], end_date=date_set[1], page_index=page_ind, org_query_str=org_set)
					r, status, totalResults, responses, search_params = prep_request(payload)
					mongo_insert_responses(responses)
	pass

if __name__ == '__main__':
	# main()
	mini(8150)
	# testOne()

'''Notes
test_master_1: big_2, big_2_42600, big_2_98600, big_2_269900, big_2_281900, big_2_289400, not_company_1

Last Org Set Counts:
	Org Set # 14350

'''
