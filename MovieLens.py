import os
import csv
import sys
import re
from surprise import Dataset
from surprise import Reader
from collections import defaultdict
import numpy as np
import pandas as pd

class MovieLens:
    self.movie_id_to_name = {}
    self.name_to_movie_id = {}
    self.ratings_path = 'ml-latest-small/ratings.csv'
    moviespath = 'ml-latest-small/movies.csv'
    def load_movie_data(self):
        rating_data = Dataset.load_from_file(self.ratings_path)
        df_movie = pd.read_csv(self.moviespath)
        df_movie['movieId'].head(5)
        self.movie_id_to_name = dict(zip(df_movie['movieId'], df_movie['title']))
        self.name_to_movie_id = dict(zip(df_movie['title'], df_movie['movieId']))
        return rating_data
    
    def get_user_rating(self,user):
        df_rating = pd.read_csv(self.ratings_path)
        user_rating = list(zip(df_rating.query('userId=='+user)['movieId'],df_rating.query('userId=='+user)['rating']))
        return user_rating
    
    def get_popular_ranks(self):
        df_popular = pd.read_csv(self.ratings_path)
        y = df_popular.groupby('movieId')['movieId'].value_counts()
        movieId = [x for x,y in list(y.keys())]
        frequency = list(y.values)
        ratings = dict(zip(movieId,frequency))
        return ratings
    
    def get_genres(self):
        genres = defaultdict(list)
        genre_ids = {}
        max_genre_id = 0
        df_movie = pd.read_csv(self.moviespath)
        movie_id_list = df_movie['movieId']
        movie_title_list = df_movie['title']
        movie_genres_str = df_movie['genres']
        
        for movie_id,genre in zip(movie_id_list,movie_genres_str):
            genre_list = genre.split('|')
            genre_id_list = []
            for genre in genre_list:
                if genre in genre_ids:
                    genre_id = genre_ids[genre]
                else:
                    genre_id = max_genre_id
                    genre_ids[genre] = genre_id
                    max_genre_id = max_genre_id + 1
                genre_id_list.append(genre_id)
            genres[movie_id] = genre_id_list
        
        for (movie_id,genre_id_list) in genres.items():
            bitfield = [0]*max_genre_id
            for genre_id in genre_id_list:
                bitfield[genre_id] = 1
            
            genres[movie_id] = bitfield
        
        return genres
    
    def get_years(self):
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)
        df_movie = pd.read_csv(self.moviespath)
        movie_title_list = df_movie['title']
        movie_id_list = df_movie['movieId']
        for title,movie_id in zip(movie_title_list,movie_id_list):
            temp = p.search(title)
            year = temp.group(1)
            if year:
                years[movie_id] = int(year)
        return years
    
    def get_mise_en_scene(self):
        mes = defaultdict(list)
        df_movie = pd.read_csv(self.moviespath)
        movie_id_list = df_movie['ML_Id']
        avg_shot_length_list = df_movie['f1']
        mean_color_variance_list = df_movie['f2']
        std_dev_color_variance_list = df_movie['f3']
        mean_motion_list = df_movie['f4']
        std_dev_motion_list = df_movie['f5']
        mean_lighting_key_list = df_movie['f6']
        num_shots_list = df_movie['f7']
        for a,b,c,d,e,f,g,h in zip(movie_id_list,avg_shot_length_list,mean_color_variance_list,std_dev_color_variance_list,mean_motion_list,std_dev_motion_list,mean_lighting_key_list,num_shots_list):
            mes[a] = [b,c,d,e,f,g,h]
        return mes
    
    def get_movie_name(self,movie_id):
        if movie_id in self.movie_id_name:
            return self.movie_id_name[movie_id]
        else:
            return ""
    
    def get_movie_id(self,movie_name):
        if movie_name in self.name_to_movie_id:
            return self.name_to_movie_id[movie_name]
        else:
            return 0