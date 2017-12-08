import wikipedia
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from pandas import read_csv
from csv import writer

def get_wiki_plot(query):
    try:
        page = wikipedia.page(query)
    except Exception as E:
        return "---;"+str(E)

    body = wikipedia.page(query).content
    try: 
        plot = body.split('\n\n\n== Plot ==\n')[1].split('\n\n')[0]
    except IndexError:
        try:
            plot = body.split('\n\n\n== PlotEdit ==\n')[1].split('\n\n')[0]
        except Exception as E:
            return "---;"+str(E)

    return plot
        

data = read_csv('movie_metadata.csv', na_values='', keep_default_na=False)
movie_list = data['movie_title'].values
year_list = data['title_year'].values

with open('actual_wiki_plots.csv','a') as outFile:
    writer = writer(outFile)
    for index in range(-1, len(movie_list)):
        if index==-1:
            writer.writerow(['id','movie_title','wiki_plot'])
            continue
        movie_name = movie_list[index]
        movie_year = year_list[index]
        if str(movie_year)=='nan':
            print 'gotcha'
            movie_year=''
        else:
            movie_year = str(int(movie_year))
        query = ' '.join([movie_name, movie_year, 'film'])
        print 'looking plot for',query
        plot = get_wiki_plot(query)
        print movie_name,plot
        writer.writerow([index,movie_name,plot])
        print index,"Done"
        print "*"*18
