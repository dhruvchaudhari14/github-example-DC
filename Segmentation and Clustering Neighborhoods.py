#!/usr/bin/env python
# coding: utf-8

# ## Table of Contents
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
# 
# <font size = 3>
# 
# 1. <a href="#item1">To Create the Required DataFrame consisting of different Boroughs in Toronto,Canada using BeautifulSoap package.</a>
# 
# 2. <a href="#item2">Merging the Data containing Borough and Neighborhood data with its respective Latitude and Longitude.</a>
# 
# 3. <a href="#item3">Exploring the dataset and extracting required data.</a>
# 
# 4. <a href="#item4">Importing Required Packages.</a>
# 
# 5. <a href="#item5">Exploring Neighborhoods containing Toronto in them.</a> 
# 
# 6. <a href="#item5">Analysing each Neighborhood.</a> 
#     
# 7. <a href="#item5">Clustering each Neighborhood.</a>     
# </font>
# </div>

# ## 1. To Create the Required DataFrame consisting of different Boroughs in Toronto,Canada  using BeautifulSoap package.

# In[1]:


import numpy as np 

import pandas as pd 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json 

get_ipython().system('conda install -c conda-forge geopy --yes ')
from geopy.geocoders import Nominatim 

import requests 
from pandas.io.json import json_normalize
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.cluster import KMeans

get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes ')
import folium 

print('Libraries imported.')


# In[2]:


import urllib.request


# In[3]:


url = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"


# In[4]:


page = urllib.request.urlopen(url)


# In[5]:


from bs4 import BeautifulSoup


# In[6]:


soup = BeautifulSoup(page, "lxml")


# In[7]:


print(soup.prettify())


# In[8]:


soup.title


# In[9]:


soup.title.string


# In[10]:


all_tables=soup.find_all("table")
all_tables


# In[11]:


right_table=soup.find('table', class_='wikitable sortable')
right_table


# In[12]:


A=[]
B=[]
C=[]

for row in right_table.findAll('tr'):
    cells=row.findAll('td')
    if len(cells)==3:
        A.append(cells[0].find(text=True))
        B.append(cells[1].find(text=True))
        C.append(cells[2].find(text=True))


# In[13]:


import pandas as pd
df=pd.DataFrame(A,columns=['Postal Code'])
df['Borough']=B
df['Neighborhood']=C
df.head()


# In[14]:


df = df.replace('\n','', regex=True)
df.head()


# In[15]:


df=df[df.Borough!='Not assigned']
df.head()


# In[16]:


df.head()


# In[17]:


df.reset_index(inplace = True) 
df.head()


# In[18]:


df.head()


# In[19]:


df=df.drop(columns="index")
df.head()


# In[20]:


df.head()


# ## 2. Merging the Data containing Borough and Neighborhood data with its respective Latitude and Longitude.

# In[21]:


latlong=pd.read_csv('C:/Users/Info/OneDrive/Desktop/IBM/Data\\Geospatial_Coordinates.csv')


# In[22]:


latlong.head()


# In[23]:


df=df.sort_values(by=['Postal Code'])
df.head()


# In[24]:


df.reset_index(inplace = True) 
df=df.drop(columns="index")
df.head()


# In[25]:


df=pd.concat([df,latlong['Latitude'],latlong['Longitude']],axis=1)
df


# ## 3. Exploring the dataset and extracting required data.

# In[26]:


print('The dataframe has {} boroughs and {} Neighborhoods.'.format(
        len(df['Borough'].unique()),
        df.shape[0]
    )
)


# In[27]:


print(df['Borough'].unique())


# In[39]:


df1=df.loc[df['Borough']=='Downtown Toronto']
df2=df.loc[df['Borough']=='Central Toronto']
df3=df.loc[df['Borough']=='West Toronto']
df4=df.loc[df['Borough']=='East Toronto']
t_neighbor=pd.concat([df1,df2,df3,df4],axis=0)
t_neighbor


# In[40]:


t_neighbor.reset_index(inplace = True) 
t_neighbor=t_neighbor.drop(columns="index")
t_neighbor.head()


# In[44]:


print('The dataframe has {} boroughs and {} Neighborhoods.'.format(
        len(t_neighbor['Borough'].unique()),
        t_neighbor.shape[0]
    )
)


# ## 4. Importing Required Packages.

# In[41]:


import folium
print("Folium is imported")


# In[42]:


import geopy
address = 'Toronto,Canada'

geolocator = Nominatim(user_agent="ca_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto is {}, {}.'.format(latitude, longitude))


# In[43]:


map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(t_neighbor['Latitude'],t_neighbor['Longitude'], t_neighbor['Borough'], t_neighbor['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# In[50]:


CLIENT_ID = '3YKX4FGPPKCE410LKNZBFPNLXENNJJABG1CD0SMGO5VP5TA0' 
CLIENT_SECRET = '4XNX1BG01ZDK3OUFOXF4Y2TYGECQDN1VV4DLPAT3LG4UPKBU' 
VERSION = '20180605' 

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# ## 5. Exploring Neighborhoods  containing Toronto in them.

# In[51]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
   
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        results = requests.get(url).json()["response"]['groups'][0]['items']
        

        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[52]:


toronto_venues = getNearbyVenues(names=t_neighbor['Neighborhood'],
                                   latitudes=t_neighbor['Latitude'],
                                   longitudes=t_neighbor['Longitude']
                                  )


# In[53]:


print(toronto_venues.shape)
toronto_venues.head()


# In[54]:


toronto_venues.groupby('Neighborhood').count()


# In[55]:


print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))


# ## 6. Analysing each Neighborhood. 

# In[58]:


toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

toronto_onehot['Neighborhood'] = toronto_venues['Neighborhood'] 

fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()


# In[59]:


toronto_onehot.shape


# #### Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category

# In[60]:


toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped


# In[61]:


toronto_grouped.shape


# #### Let's print each neighborhood along with the top 5 most common venues

# In[62]:


num_top_venues = 5

for hood in toronto_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[63]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[159]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# ## 7. Clustering each Neighborhood.

# In[160]:


kclusters = 4

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

kmeans.labels_[0:10] 


# In[161]:


neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = t_neighbor

toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_merged.head() 


# In[162]:


toronto_merged.tail()


# In[163]:


map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'],toronto_merged['Longitude'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# ## Cluster 1

# In[148]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 0, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# ## Cluster 2

# In[142]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 1, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# ## Cluster 3

# In[152]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 2, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# ## Cluster 4

# In[153]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 3, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# # From above clusters we can see that Cluster 3 has much more venues in it as compared to all the other clusters, even after trying different k values, No values could show a better map and distributed clusters.
