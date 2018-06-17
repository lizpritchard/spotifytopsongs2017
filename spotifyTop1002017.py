# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 16:14:29 2018

@author: Liz Pritchard

Project: Spotify Top 100 Songs of 2017 Data Exploration from Kaggle
"""
#%%

import pandas as pd 
import os
import seaborn as sns; sns.set()
sns.palplot(sns.color_palette("colorblind"))

import matplotlib.pyplot as plt

os.chdir("C:/Users/Liz/Documents/Kaggle/Spotify/Top 100 Songs of 2017")

#%%
# Create dataframe
topSongs = pd.read_csv("featuresdf.csv")

#%%
# summarize dataset
topSongs.describe()

#%%
# data analyzation - danceability
# majority of top 100 songs from 2017 have high danceability value
print("Mean value of danceability", topSongs["danceability"].mean())
sns.distplot(topSongs["danceability"])
plt.show()

#%% data analyzation - mode (major/minor key)

print("Mean value of mode", topSongs["mode"].mean())
# mean value of 0.58
# majority of users prefer songs in a major key 

# mode is binary with 1 as major and 0 as minor
# mode map of binary values for music mode
mode_map = {1.0: "majorKey", 0.0: "minorKey"}
topSongs["mode"] = topSongs["mode"].map(mode_map)

# countplot of minor vs major songs
sns.countplot(x = "mode", data=topSongs)
plt.title("Number of Minor vs Major Songs")
plt.show()

#%% valence
# users tend to prefer songs in a major key
# does major key = happy songs? 

print("Mean value of valence in songs", topSongs["valence"].mean())
sns.distplot(topSongs["valence"])
plt.show()

# mean value of 0.5170489999999999 > even distribution
# though majority of songs are in major key, that does not mean they're nececssarily always happy

#%%
# new dataset - music features 

musicFeatures = topSongs.drop(["name", "artists", "id", "duration_ms"], axis=1)

# correlation heatmap of song features 
plt.figure(figsize = (16,5))
sns.heatmap(musicFeatures.corr(), cmap="Spectral", annot=True)
plt.show()

#%%
"""
Ideas for future exploration: 
    Positive/upbeat songs vs. sad/lowkey songs (mode, valence, energy, key)
    Instrumental vs speechiness   
"""
