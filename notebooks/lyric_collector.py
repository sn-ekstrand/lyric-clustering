import time
from pymongo import MongoClient
import json
import requests
import ast

client = MongoClient('localhost', 27017)
db = client['songs']
song_lyrics = db['lyrics']

for chart in db['song_collection'].find():
    songs = chart['songs']
    for k, v in songs.items():
        song_dict = dict()
        artist_lookup = v[0].split(' Featuring ')[0]
        title = v[1]
        try:
            r = requests.get('https://api.lyrics.ovh/v1/{}/{}'.format(artist_lookup, title))
            lyrics = ast.literal_eval(r.text)['lyrics']
        except:
            lyrics = None
        time.sleep(22)
        song_dict = {'week': chart['week'], 
                     'standing': k, 
                     'artist': v[0], 
                     'title': title, 
                     'lyrics': lyrics}
        song_lyrics.insert_one(song_dict)