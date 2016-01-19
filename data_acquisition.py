# data_acquisition.py

import numpy as np
import pandas as pd
import requests
import json
from bs4 import BeautifulSoup
from pprint import pprint
import pymongo
from pymongo import MongoClient


def get_master_data():
    '''
        Function used to acquire different PR Newswire API parameter values.
        i.e. Geography, Industry and Subject lists.

        INPUT:
        - None

        OUTPUT:
        - None: preps requests object with appropriate parameters submitted to API
    '''
    masterData_url = 'http://api.prnewswire.com/releases/getMasterData'
    masterData_payload = {
        'listName': 'GEOGRAPHY'
    }
    r = requests.get(masterData_url, params=masterData_payload)
    return r


def make_orgs_df(filename):
    '''
        Creates Pandas DF of all organization names from Crunchbase dataset given certain parameters. All crunchbase orgs previously downloaded into local CSV file.

        INPUT:
        - filename: name and location of local organization csv file

        OUTPUT:
        - df_big: dataframe containing all organziation names of interest
    '''
    df = pd.read_csv(filename)

    # get only companies that classify themselves as a company or investor. Merge into one big dataframe.
    df_company = df[df['primary_role'] == 'company']
    df_investor = df[df['primary_role'] == 'investor']
    df_big = pd.concat([df_company, df_investor])

    # Drop duplicate entries as Crunchbase data isn't perfectly clean
    df_big.drop_duplicates(subset=['name'], inplace=True)

    return df_big


def make_one_org_query_str(df, start, finish):
    '''
        Makes single organization name parameter string for PR Newswire API in API specific format.

        INPUT:
        - df: pandas dataframe including orgnaization names from make_orgs_df()
        - start: index of dataframe at which to start range of organization names
        - finish: index of dataframe at which to finish range of organization names

        OUTPUT:
        - org_query_str: correctly formated string for API parameter submission
    '''
    org_query_str = ''
    for i in xrange(start, finish):
        # checks if index is out of range of dataframe size to prevent errors upon submission
        if i >= df.shape[0]:
            continue

        # company names must be surrounded by double quotes in the actual string
        string = '\"' + str(df['name'].iloc[i]) + '\"'

        # if it's the start index, put only company name. Otherwise, put ' OR ' in between all names.
        if i == start:
            org_query_str += string
        else:
            org_query_str += ' OR ' + string

    return org_query_str


def make_all_org_query_strings(df, total, request_size=50, start=0, finish=49):
    '''
        Makes list of single organization name parameter strings for PR Newswire API in API specific format. Batches strings in list to loop over later for repeated API requests.

        INPUT:
        - df: pandas dataframe including orgnaization names from make_orgs_df()
        - total: total number of organization names that need to be assembled into API strings
        - request_size: batch size for single organization name strings. Default set to 50 as API errors with larger batch size.
        - start: index of dataframe at which to start range of org names
        - finish: index of dataframe at which to finish range of org names

        OUTPUT:
        - result: list of single org name submission strings
    '''
    result = []

    # calculate number of requests needed
    requests_needed = int(total/request_size)

    # create requests and append to list
    for i in xrange(requests_needed):
        result.append(make_one_org_query_str(df, start, finish))
        start += request_size
        finish += request_size

    return result


def make_date_strings(start, end):
    '''
        Make list of date strings for API submission over the requested range of years. Date strings are batched into three sets of four-month long strings for each year.

        INPUT:
        - start: first year to start assembling date strings
        - end: last year to finish assembling date strings

        OUTPUT:
        - result: list of date strings for API submission
    '''
    result = []
    for year in range(start, end + 1):
        # January-April. API format is 'DD/MM/YYYY hh:mm'
        start_date_1 = '01/01/' + str(year) + ' 01:01'
        end_date_1 = '30/04/' + str(year) + ' 01:01'
        # May-August
        start_date_2 = '01/05/' + str(year) + ' 01:01'
        end_date_2 = '31/08/' + str(year) + ' 01:01'
        # September-December
        start_date_3 = '01/09/' + str(year) + ' 01:01'
        end_date_3 = '31/12/' + str(year) + ' 01:01'

        # puts date strings into pairs of [start, finish]
        temp1 = [start_date_1, end_date_1]
        temp2 = [start_date_2, end_date_2]
        temp3 = [start_date_3, end_date_3]

        result.append(temp1)
        result.append(temp2)
        result.append(temp3)
    return result


def prep_payload(start_date, end_date, page_index, org_query_str, fixed_start_date=False):
    '''
        Preps API request parameters (payload) into single dictionary

        INPUT:
        - start_date: start date for API request
        - end_date: end date for API request
        - page_index: page index for API request
        - org_query_str: from org_query_str()
        - fixed_start_date: whether or not you want a fixed or variable start_date

        OUTPUT:
        - payload: dictionary containing all API request parameters
    '''
    # subjects, industries and geographies from PR Newswire API docs for which we want to get press releases
    subject_query_str = 'subject:(TNM OR CON OR FNC OR LIC OR PDT OR SLS OR VEN OR FEA)'
    industry_query_str = 'industry:(BIO OR CPR OR NET OR HRD OR STW OR CSE OR ITE OR MLM OR TLS OR HTS OR SEM OR BIM OR MEN OR NAN OR SMD OR TCS OR TEQ OR VIP WEB OR WIC OR GRE OR BRD)'
    geo_query_str = 'geography:(USA)'

    # if/else only difference is fixed_start_date or not. Create payload dictionary and return.
    if fixed_start_date:
        payload = {
        'apikey': 'ded6410542df3e0e',
        'query': '(content:(' + org_query_str + ') + companies:(' + org_query_str + ') + ' + subject_query_str + ' + ' + industry_query_str + ' + ' + geo_query_str + ' + language:en)',
        'pageSize': 100,
        'pageIndex': page_index,
        'startDate': '12/07/2015'
        # 'endDate': end_date
        }
    else:
        payload = {
        'apikey': 'ded6410542df3e0e',
        'query': '(content:(' + org_query_str + ') + companies:(' + org_query_str + ') + ' + subject_query_str + ' + ' + industry_query_str + ' + ' + geo_query_str + ' + language:en)',
        'pageSize': 100,
        'pageIndex': page_index,
        'startDate': start_date,
        'endDate': end_date
        }
    return payload


def prep_request(payload):
    '''
        Preps API requests object with appropriate URL and payload. Returns request response object and various submission responses for later debug printing.

        INPUT:
        - payload: API parameters payload from prep_payload() function

        OUTPUT:
        - r: API reponse stored in a variable
        - status: status response code returned from API
        - totalResults: number of total results received in API response
        - search_params: search parameters from API request
    '''
    # submits API request and stores response into 'r' variable
    getReleases_url = 'http://api.prnewswire.com/releases/version01/getReleases'
    r = requests.get(getReleases_url, params=payload)

    # API response elements. 'responses' is commented out for debugging purposes.
    status = r.json()['feed']['statusCode']
    totalResults = r.json()['feed']['totalResults']
    # responses = r.json()['releases']['release']
    search_params = r.json()['feed']['filter']
    return r, status, totalResults, search_params


def start_mongo(db_name, coll_name):
    '''
        Creates connection to local MongoDB database. Returns 'collection' variable for later writing API responses to database.

        INPUT:
        - db_name: MongoDB database name
        - coll_name: MongoDB collection name

        OUTPUT:
        - coll: pymongo collection object connected to appropriate local database
    '''
    # create MongoClient instance connected to local database
    client = MongoClient('127.0.0.1', 27017)

    # access desired database and collection, and return collection variable
    db = client[db_name]
    coll = db[coll_name]
    return coll


def mongo_insert_responses(responses, search_params, coll):
    '''
        Write API responses to MongoDB database.

        INPUT:
        - responses: list of API responses, each API response is a dictionary with different information stored on each key
        - search_params: search_params submitted with this batch of responses
        - coll: MongoDB collection name to write responses to

        OUTPUT:
        - None: function writes to MongoDB database and returns nothing
    '''
    docs = []
    for each in responses:
        # html string of press release content
        html_text = each['releaseContent']

        # if response is empty, skip this one
        if html_text is None or len(html_text) == 0:
            continue

        # create BeautifulSoup object with html_text for parsing
        soup = BeautifulSoup(html_text, 'html.parser')

        # use BeautifulSoup's .get_text() method to get only the press release text from the html
        release_text = soup.get_text()

        # create dictionary of response items in prep for writing to MongoDB. MongoDB takes the keys as column names and the values as entries for a given row.
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
        # batch set of responses by appending to exterior list
        docs.append(doc)

    # insert this batch of dictionaries into MongoDB
    coll.insert_many(docs)
    pass


def testOne(start_index, end_index):
    '''
        Debug function to test a single set of organziation names, instead of a large list of them.

        INPUT:
        - start_index: start index of organziation dataframe to begin range of org names
        - end_index: end index of organziation dataframe to end range of org names

        OUTPUT:
        - None: Returns nothing. Prints several items from API response.
    '''
    # create orgs dataframe
    csv_file = '../crunchbase/cb_odm_csv/organizations.csv'
    df = make_orgs_df(csv_file)

    # create org query string and payload
    org_query_str = make_one_org_query_str(df, start_index, end_index)
    payld = prep_payload(start_date='09/07/2015', end_date='04/01/2016', page_index=1, org_query_str=org_query_str, fixed_start_date=True)
    print payld

    # submit API request and store responses
    r, status, totalResults, search_params = prep_request(payld)
    responses = r.json()['releases']['release']

    # print API responses for debugging expenses
    print responses[0]
    print status
    print totalResults


def main(start_index):
    '''
        Main function that calls all other functions in succession. Goes from list of organizations, prepping and submitting API request, and writing results to MongoDB.

        Contains several print statements to monitor status of program while running.

        Max API calls per day is 5000. Therefore, function allows for setting of new start index of organization dataframe, this way you can set the start_index of today's requests as one greater than the final index of yesterday's requests.

        INPUT:
        - start_index: start index of organizations dataframe at which to begin today's requests

        OUTPUT:
        - None: function writes to MongoDB database and returns nothing
    '''
    # create orgs df
    orgs_csv_file = 'data/organizations.csv'
    df_orgs = make_orgs_df(orgs_csv_file)
    print 'DataFrame successfully created'

    # create MongoDB connection object
    coll = start_mongo('press', 'spra_master')
    print 'Mongo DB started'

    # create list of API organization query strings
    org_name_sets = make_all_org_query_strings(df=df_orgs, total=df_orgs.shape[0], request_size=50, start=start_index, finish=(start_index+49))
    print 'Org Name Sets successfully created'

    # create date_strings
    # date_strings = make_date_strings(2014, 2015)

    # loop through all sets of organization names
    for index, org_set in enumerate(org_name_sets):
        print 'Org Set #', start_index

        # increase start index for this request
        start_index += 50

        # prep API request payload with this set of org names
        payload = prep_payload(start_date='12/07/2015', end_date='07/01/2016', page_index=1, org_query_str=org_set, fixed_start_date=True)

        # submit API request and return response variables
        r, status, totalResults, search_params = prep_request(payload)

        # if API response was empty, skip this set of org names. Created to prevent errors.
        if totalResults == 0:
            continue

        # set responses variable in prep for MongoDB writing function
        responses = r.json()['releases']['release']

        # write responses to MongoDB
        mongo_insert_responses(responses, search_params, coll)

        # each API response can only return 100 results, this checks to see if multiple requests are needed for a single set of org names
        if totalResults <= 100:
            continue
        else:
            # calculates number of requests needed to get all responses from this set of org names. Responses >100 require increasing the 'page_index' parameter of the API request.
            pages_needed = int(totalResults/100) + 1

            # loop through number of requests needed
            for page_ind in xrange(2, pages_needed + 1):
                # create request payload
                payload = prep_payload(start_date='12/07/2015', end_date='07/01/2016', page_index=page_ind, org_query_str=org_set, fixed_start_date=True)

                # sets response variables
                r, status, totalResults, search_params = prep_request(payload)

                # checks if response is empty to prevent errors
                if totalResults == 0:
                    continue

                # creates response variable and writes to MongoDB
                responses = r.json()['releases']['release']
                mongo_insert_responses(responses, search_params, coll)
    pass


if __name__ == '__main__':
    main(1)
