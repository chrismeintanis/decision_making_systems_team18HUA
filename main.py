import mysql.connector
from scipy import stats
from mysql.connector import errorcode
import requests
import random
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import networkx as nx
from math import sqrt
from sqlalchemy import create_engine
from deap import creator, base, tools, algorithms

API_KEY = 'b1825b9288e6c9ea2613df2ecf29e416'
API_SECRET = '8242e0b24ce3fe63489cf398401799d5'
USERNAME = 'meintanis'
PASSWORD = 'Mei258749.'




DB_NAME = 'users'

cnx = mysql.connector.connect(user='christos', password='password',
                              host='127.0.0.1',
                              database='users', raise_on_warnings=True)
cursor = cnx.cursor()

TABLES = {}
TABLES['user'] = (
    "CREATE TABLE `user` ("
    "   `id` int(3) NOT NULL AUTO_INCREMENT,"
    "   `first_name` varchar(24) NOT NULL,"
    "   `last_name` varchar(24) NOT NULL,"
    "   `gender` enum('M','F') NOT NULL,"
    "   `birth_date` DATE NOT NULL,"
    "    PRIMARY KEY(`id`)"
    ")ENGINE=InnoDB"
)

TABLES['artist'] = (
    "CREATE TABLE `artist` ("
    "  `id` varchar(255) NOT NULL,"
    "  `name` varchar(255) NOT NULL,"
    "  `listeners` int(11) NOT NULL,"
    "  `playcount` int(11) NOT NULL,"
    "  `bio` text,"
    "  PRIMARY KEY (`id`)"
    ") ENGINE=InnoDB"
)

TABLES['album'] = (
    "CREATE TABLE `album` ("
    "  `id` varchar(255) NOT NULL,"
    "  `title` varchar(255) NOT NULL,"
    "  `artist` varchar(255) NOT NULL,"
    "  `artistId` varchar(255) NOT NULL,"
    "  `playcount` int(11) NOT NULL,"
    "  PRIMARY KEY (`id`)"
    ") ENGINE=InnoDB"
)

TABLES['user_albums'] = (
    "CREATE TABLE `user_albums` ("
    "  `uid` int(3) NOT NULL,"
    "  `last_name` varchar(24) NOT NULL,"
    "  `album_title` varchar(255) NOT NULL,"
    "  `album_id` varchar(255) NOT NULL,"
    "  PRIMARY KEY (`uid`,`album_id`)"
    ") ENGINE=InnoDB"
)

TABLES['user_artists'] = (
    "CREATE TABLE `user_artists` ("
    "  `uid` int(3) NOT NULL,"
    "  `last_name` varchar(24) NOT NULL,"
    "  `artist_name` varchar(255) NOT NULL,"
    "  `artist_id` varchar(255) NOT NULL,"
    "  PRIMARY KEY (`uid`,`artist_id`)"
    ") ENGINE=InnoDB"
)


def create_database(crs):
    try:
        crs.execute("CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'".format(DB_NAME))
    except mysql.connector.Error as err:
        print("Failed creating database: {}".format(err))
        exit(1)

    try:
        crs.execute("USE {}".format(DB_NAME))
    except mysql.connector.Error as err:
        print("Database {} does not exists.".format(DB_NAME))
        if err.errno == errorcode.ER_BAD_DB_ERROR:
            create_database(crs)
            print("Database {} created succesfully.".format(DB_NAME))
            cnx.database = DB_NAME
        else:
            print(err)
            exit(1)


for table_name in TABLES:
    table_desc = TABLES[table_name]
    try:
        print("Creating table {}: ".format(table_name), end='')
        cursor.execute(table_desc)
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
            print("already exists.")
        else:
            print(err.msg)
    else:
        print("OK")

users = [
    ("Noah", "Williams", "M", "1985-07-22"),
    ("Olivia", "Jones", "F", "1990-05-11"),
    ("Liam", "Brown", "M", "1988-01-02"),
    ("Ava", "Davis", "F", "1993-08-19"),
    ("Mason", "Garcia", "M", "1986-06-12"),
    ("Isabella", "Rodriguez", "F", "1989-04-23"),
    ("Jacob", "Martinez", "M", "1991-09-29"),
    ("Sophia", "Hernandez", "F", "1987-11-17"),
    ("Ethan", "Lopez", "M", "1994-02-05"),
    ("Mia", "Gonzalez", "F", "1997-10-08"),
    ("Michael", "Perez", "M", "1996-12-27"),
    ("Emily", "Rivera", "F", "1984-03-01"),
    ("William", "Lee", "M", "1983-09-18"),
    ("Madison", "Walker", "F", "1999-07-14"),
    ("Alexander", "Hall", "M", "1982-04-21"),
    ("Grace", "Young", "F", "1998-06-25"),
    ("Daniel", "Allen", "M", "1995-11-03"),
    ("Chloe", "King", "F", "1981-12-09"),
    ("Larry", "Hoover", "M", "1991-03-20"),
    ("Conor", "McGregor", "M", "1987-07-14")
]

for user in users:
    check_user = "SELECT id FROM user WHERE first_name=%s AND last_name=%s AND gender=%s AND birth_date=%s"
    cursor.execute(check_user, user[0:4])
    result = cursor.fetchone()
    if result is None:
        insert_user = "INSERT INTO user (first_name, last_name, gender, birth_date) VALUES (%s, %s, %s, %s)"
        cursor.execute(insert_user, user)


def insert_user_artists(user_id,user_name , artists, crs):
    random.shuffle(artists)
    selected_artists = artists[:5]  # Select the first 3 artists

    for artist in selected_artists:
        artist_id = artist[0]
        artist_name = artist[1]

        query = "INSERT INTO user_artists (uid, last_name, artist_name, artist_id) VALUES (%s, %s, %s, %s)"
        values = (user_id, user_name, artist_name, artist_id)

        try:
            crs.execute(query, values)
            print("User Artist inserted successfully: User ID: {}, Artist ID: {}, Name: {}".format(user_id, artist_id,
                                                                                                   artist_name))
        except mysql.connector.Error as err:
            if err.errno == mysql.connector.errorcode.ER_DUP_ENTRY:
                print("Duplicate entry. Skipping insert.")
            else:
                print("Error inserting user artist:", err)


def insert_user_albums(user_id,user_name, albums, crs):
    random.shuffle(albums)
    selected_albums = albums[:5]  # Select the first 3 albums

    for album in selected_albums:
        album_id = album[0]
        album_title = album[1]

        query = "INSERT INTO user_albums (uid, last_name, album_title, album_id) VALUES (%s, %s, %s, %s)"
        values = (user_id, user_name, album_title, album_id)

        try:
            crs.execute(query, values)
            print("User Album inserted successfully: User ID: {}, Album ID: {}, Title: {}".format(user_id, album_id,
                                                                                                  album_title))
        except mysql.connector.Error as err:
            if err.errno == mysql.connector.errorcode.ER_DUP_ENTRY:
                print("Duplicate entry. Skipping insert.")
            else:
                print("Error inserting user album:", err)


# ...

def retrieve_albums(crs):
    query = "SELECT id, title FROM album"

    try:
        crs.execute(query)
        albums = crs.fetchall()
        return albums
    except mysql.connector.Error as err:
        print("Error retrieving albums:", err)
        return []


def retrieve_artists(crs):
    query = "SELECT id, name FROM artist"

    try:
        crs.execute(query)
        artists = crs.fetchall()
        return artists
    except mysql.connector.Error as err:
        print("Error retrieving artists:", err)
        return []




def insert_album(album_id1 ,title1 ,artist1 ,artistId1 ,playcount1 ,crs):
    check_album = "SELECT id FROM album WHERE id=%s AND title=%s AND artist=%s AND artistId=%s AND playcount=%s"
    values = ( album_id1, title1, artist1, artistId1, playcount1)
    crs.execute(check_album, values)
    row = crs.fetchone()

    query = "INSERT INTO album (id, title, artist, artistId, playcount) VALUES (%s, %s, %s, %s, %s)"
    values = (album_id, album_title, album_artist, album_artistId, album_playcount)
    # crs.execute(query, values)
    # print("Album {} inserted successfully.".format(album_title))
    try:
        crs.execute(query, values)
        print("Album {} inserted successfully.".format(title1))
    except mysql.connector.Error as err:
        if err.errno == mysql.connector.errorcode.ER_DUP_ENTRY:
            print("Duplicate entry. Skipping insert.")
        else:
            print("Error inserting Album:", err)


def insert_artist(artist_id1,artist_name1, playcount1, listeners1, bio1, crs):
    check_artist = "SELECT id FROM artist WHERE id=%s AND name=%s AND playcount=%s AND listeners=%s AND bio=%s"
    values = (artist_id1, artist_name1, playcount1, listeners1, bio1)
    crs.execute(check_artist, values)
    row = crs.fetchone()

    query = "INSERT INTO artist (id ,name, listeners, playcount, bio) VALUES (%s, %s, %s, %s, %s)"
    values = (artist_id, artist_name, listeners, playcount, bio)
    # crs.execute(query, values)
    # print("Artist {} inserted successfully.".format(artist_name))
    try:
        crs.execute(query, values)
        print("Artist {} inserted successfully.".format(artist_name1))
    except mysql.connector.Error as err:
        if err.errno == mysql.connector.errorcode.ER_DUP_ENTRY:
            print("Duplicate entry. Skipping insert.")
        else:
            print("Error inserting artist:", err)





getTopArtists = requests.get('http://ws.audioscrobbler.com/2.0/?method=chart.gettopartists&api_key=b1825b9288e6c9ea2613df2ecf29e416&format=json')
resp_desc = getTopArtists.json()
topArtists = resp_desc["artists"]["artist"]


counter = 0
for artist in topArtists:
    artist_name = artist["name"]
    getArtistInfo = requests.get(f'http://ws.audioscrobbler.com/2.0/?method=artist.getinfo&artist={artist_name}&api_key=b1825b9288e6c9ea2613df2ecf29e416&format=json')
    artistInfoResponse = getArtistInfo.json()

    resp = artistInfoResponse["artist"]

    artist_id = resp["mbid"]
    playcount = resp["stats"]["playcount"]
    listeners = resp["stats"]["listeners"]
    bio = resp["bio"]["summary"]

    insert_artist(artist_id, artist_name, playcount, listeners, bio, cursor)
    artist_name = artist["name"]
    artist_name = artist_name.lower()
    artist_name = artist_name.replace(' ', '+')
    getTopAlbums = requests.get(f'http://ws.audioscrobbler.com/2.0/?method=artist.gettopalbums&artist={artist_name}&api_key=b1825b9288e6c9ea2613df2ecf29e416&format=json&limit=15')
    albums =getTopAlbums.json()["topalbums"]["album"]
    for album in albums:
        album_title = album["name"]
        album_id = album.get("mbid")
        if album_id is None:
            continue
        album_artist = album["artist"]["name"]
        album_artistId = album["artist"]["mbid"]
        album_playcount = album["playcount"]
        album_release_date = None


        insert_album(album_id, album_title, album_artist, album_artistId, album_playcount, cursor)
        album_values = (album_id, album_title, album_artist, album_artistId, album_playcount)

    if counter == 9:
        break

    counter += 1


# Retrieve albums and artists
albums = retrieve_albums(cursor)
artists = retrieve_artists(cursor)
def retrieve_user(crs):
    query = "SELECT id, last_name FROM user"
    crs.execute(query)
    rows = crs.fetchall()

    if rows is None:
        return None

    return rows

usersTmp=retrieve_user(cursor)

query = "DELETE FROM user_albums"
cursor.execute(query)
query = "DELETE FROM user_artists"
cursor.execute(query)

# Insert 3 random artists and 3 random albums for each user
for user in usersTmp:
    user_id = user[0]
    user_name = user[1]

    insert_user_artists(user_id,user_name, artists, cursor)
    insert_user_albums(user_id,user_name, albums, cursor)



# Retrieve user albums and artists
query = "SELECT u.uid, ua.album_id, u.artist_id FROM user_artists u JOIN user_albums ua ON u.uid = ua.uid"
cursor.execute(query)
user_albums = cursor.fetchall()


# engine = create_engine('mysql+mysqlconnector://christos:password@localhost/users')
#
# # Execute the SQL query to load the data
# query = "SELECT * FROM user"
# df = pd.read_sql(query, con=engine)
#
# # Handle missing values
# df.fillna(df.select_dtypes(include=np.number).mean(), inplace=True)
#
# # Handle outlier values using Z-score
# numeric_columns = df.select_dtypes(include=np.number).columns
# z_scores = np.abs(stats.zscore(df[numeric_columns]))
# df = df[(z_scores < 3).all(axis=1)]
#
# # Execute the SQL query to load the data
# query = "SELECT * FROM artist"
# df = pd.read_sql(query, con=engine)
#
# # Handle missing values
# df.fillna(df.select_dtypes(include=np.number).mean(), inplace=True)
#
# # Handle outlier values using Z-score
# numeric_columns = df.select_dtypes(include=np.number).columns
# z_scores = np.abs(stats.zscore(df[numeric_columns]))
# df = df[(z_scores < 3).all(axis=1)]
#
# # Execute the SQL query to load the data
# query = "SELECT * FROM album"
# df = pd.read_sql(query, con=engine)
#
# # Handle missing values
# df.fillna(df.select_dtypes(include=np.number).mean(), inplace=True)
#
# # Handle outlier values using Z-score
# numeric_columns = df.select_dtypes(include=np.number).columns
# z_scores = np.abs(stats.zscore(df[numeric_columns]))
# df = df[(z_scores < 3).all(axis=1)]
#
# # Execute the SQL query to load the data
# query = "SELECT * FROM user_artists"
# df = pd.read_sql(query, con=engine)
#
# # Handle missing values
# df.fillna(df.select_dtypes(include=np.number).mean(), inplace=True)
#
# # Handle outlier values using Z-score
# numeric_columns = df.select_dtypes(include=np.number).columns
# z_scores = np.abs(stats.zscore(df[numeric_columns]))
# df = df[(z_scores < 3).all(axis=1)]
#
# # Execute the SQL query to load the data
# query = "SELECT * FROM user_albums"
# df = pd.read_sql(query, con=engine)
#
# # Handle missing values
# df.fillna(df.select_dtypes(include=np.number).mean(), inplace=True)
#
# # Handle outlier values using Z-score
# numeric_columns = df.select_dtypes(include=np.number).columns
# z_scores = np.abs(stats.zscore(df[numeric_columns]))
# df = df[(z_scores < 3).all(axis=1)]
#
# # Display the processed data
# print(df)



# Create an empty graph
G = nx.Graph()

# Add nodes for users
users = set([user[0] for user in user_albums])
G.add_nodes_from(users)

# Add edges between users who have the same album_id and artist_id
for user_album in user_albums:
    user_id = user_album[0]
    album_id = user_album[1]
    artist_id = user_album[2]

    # Find other users who have the same album_id and artist_id
    connected_users = [user[0] for user in user_albums if user[1] == album_id and user[2] == artist_id]

    # Add edges between the current user and connected users
    for connected_user in connected_users:
        if user_id != connected_user:
            # Set the weight of the edge to 1
            G.add_edge(user_id, connected_user, weight=1)

# Draw the graph
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_color='b', node_size=1000, alpha=0.8)
nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
nx.draw_networkx_labels(G, pos, font_color='white', font_size=10)
plt.axis('off')
plt.title('User Connections based on Album and Artist')
plt.show()




def retrieve_top_artists_by_playcount(crs, limit):
    query = "SELECT name, playcount FROM artist ORDER BY playcount DESC LIMIT %s"
    values = (limit,)

    try:
        crs.execute(query, values)
        top_artists = crs.fetchall()
        return top_artists
    except mysql.connector.Error as err:
        print("Error retrieving top artists by playcount:", err)
        return []

def retrieve_top_artists_by_listeners(crs, limit):
    query = "SELECT name, listeners FROM artist ORDER BY listeners DESC LIMIT %s"
    values = (limit,)

    try:
        crs.execute(query, values)
        top_artists = crs.fetchall()
        return top_artists
    except mysql.connector.Error as err:
        print("Error retrieving top artists by listeners:", err)
        return []

def retrieve_top_albums_by_playcount(crs, limit):
    query = "SELECT title, playcount FROM album ORDER BY playcount DESC LIMIT %s"
    values = (limit,)

    try:
        crs.execute(query, values)
        top_albums = crs.fetchall()
        return top_albums
    except mysql.connector.Error as err:
        print("Error retrieving top albums by playcount:", err)
        return []

# ...


def retrieve_top_artists_from_user_artists(crs, limit):
    query = "SELECT artist_name, COUNT(*) as playcount FROM user_artists GROUP BY artist_name ORDER BY playcount DESC LIMIT %s"
    values = (limit,)

    try:
        crs.execute(query, values)
        top_artists = crs.fetchall()
        return top_artists
    except mysql.connector.Error as err:
        print("Error retrieving top artists from user_artists:", err)
        return []

def retrieve_top_albums_from_user_albums(crs, limit):
    query = "SELECT album_title, COUNT(*) as playcount FROM user_albums GROUP BY album_title ORDER BY playcount DESC LIMIT %s"
    values = (limit,)

    try:
        crs.execute(query, values)
        top_albums = crs.fetchall()
        return top_albums
    except mysql.connector.Error as err:
        print("Error retrieving top albums from user_albums:", err)
        return []

# ...



# Retrieve top artists by playcount
top_artists_playcount = retrieve_top_artists_by_playcount(cursor, 10)
top_artists_names_playcount = [artist[0] for artist in top_artists_playcount]
top_artists_playcount_values = [artist[1] for artist in top_artists_playcount]

# Plot top artists by playcount
plt.figure(figsize=(10, 6))
plt.bar(top_artists_names_playcount, top_artists_playcount_values)
plt.title('Top Artists by Playcount')
plt.xlabel('Artist')
plt.ylabel('Playcount')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Retrieve top artists by listeners
top_artists_listeners = retrieve_top_artists_by_listeners(cursor, 10)
top_artists_names_listeners = [artist[0] for artist in top_artists_listeners]
top_artists_listeners_values = [artist[1] for artist in top_artists_listeners]

# Plot top artists by listeners
plt.figure(figsize=(10, 6))
plt.bar(top_artists_names_listeners, top_artists_listeners_values)
plt.title('Top Artists by Listeners')
plt.xlabel('Artist')
plt.ylabel('Listeners')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Retrieve top albums by playcount
top_albums_playcount = retrieve_top_albums_by_playcount(cursor, 10)
top_albums_titles_playcount = [album[0] for album in top_albums_playcount]
top_albums_playcount_values = [album[1] for album in top_albums_playcount]

# Plot top albums by playcount
plt.figure(figsize=(10, 6))
plt.bar(top_albums_titles_playcount, top_albums_playcount_values)
plt.title('Top Albums by Playcount')
plt.xlabel('Album')
plt.ylabel('Playcount')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Retrieve top artists from user_artists
top_artists_user_artists = retrieve_top_artists_from_user_artists(cursor, 10)
top_artists_names_user_artists = [artist[0] for artist in top_artists_user_artists]
top_artists_playcount_user_artists = [artist[1] for artist in top_artists_user_artists]

# Plot top artists from user_artists
plt.figure(figsize=(10, 6))
plt.bar(top_artists_names_user_artists, top_artists_playcount_user_artists)
plt.title('Top Artists (user_artists)')
plt.xlabel('Artist')
plt.ylabel('Playcount')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Retrieve top albums from user_albums
top_albums_user_albums = retrieve_top_albums_from_user_albums(cursor, 10)
top_albums_titles_user_albums = [album[0] for album in top_albums_user_albums]
top_albums_playcount_user_albums = [album[1] for album in top_albums_user_albums]

# Plot top albums from user_albums
plt.figure(figsize=(10, 6))
plt.bar(top_albums_titles_user_albums, top_albums_playcount_user_albums)
plt.title('Top Albums (user_albums)')
plt.xlabel('Album')
plt.ylabel('Playcount')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



date_rng = pd.date_range(start='1/1/2018', end='12/31/2022', freq='D')
val = 40 + 15 * np.tile(np.sin(np.linspace(-np.pi, np.pi, 365)), 5)
val = np.append(val, val[1824]) + 5 * np.random.rand(1826)
series = pd.DataFrame({
    'values': val
}, index=pd.DatetimeIndex(date_rng))

# Plotting the time series
plt.figure(figsize=(10, 6))
plt.plot(series.index, series['values'])
plt.title('Time Series')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# Decomposing the time series into trend, seasonality, and residuals
decomposition = seasonal_decompose(series['values'], model='additive')

# Plotting the decomposed components
plt.figure(figsize=(10, 8))
plt.subplot(4, 1, 1)
plt.plot(series.index, decomposition.trend)
plt.title('Trend')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(series.index, decomposition.seasonal)
plt.title('Seasonality')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(series.index, decomposition.resid)
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)

plt.tight_layout()
plt.show()


X = series['values'].values
size = int(len(X) * 0.76)
train, test = X[:size], X[size:]

# Fitting the ARIMA model
history = list(train)
predictions = []

for t in range(len(test)):
    model = ARIMA(history, order=(1, 0, 0))  # Adjust the order as needed
    model_fit = model.fit()
    output = model_fit.forecast()
    pred = output[0]
    predictions.append(pred)
    true_value = test[t]
    history.append(true_value)
    print(f"True Value: {true_value:.2f}, Predicted Value: {pred:.2f}")

# Evaluating the model performance
mse = mean_squared_error(test, predictions)
print(f"Mean Squared Error (MSE): {mse:.2f}")

mse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % mse)
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


# Erwthma h# Retrieve users


# # Εύρεση των γειτόνων του επιλεγμένου χρήστη
# neighbors = list(G.neighbors(user_id))
#
# # Εάν ο επιλεγμένος χρήστης βρίσκεται στη λίστα των γειτόνων, αφαίρεσή του
# if user_id in neighbors:
#     neighbors.remove(user_id)
#
# # Προτεινόμενοι δίσκοι για κάθε χρήστη
# recommended_albums = {}
#
# for user_id in users:
#     # Εύρεση του χρήστη στον πίνακα `user` με βάση το user_id
#     query = "SELECT * FROM user WHERE id = %s"
#     cursor.execute(query, (user_id,))
#     user_data = cursor.fetchone()
#
#     if user_data:
#         # Αποσυσκευοφορία των στοιχείων του χρήστη
#         user_id, first_name, last_name, gender, birth_date = user_data
#
#         # Εύρεση των γειτόνων του επιλεγμένου χρήστη
#         neighbors = list(G.neighbors(user_id))
#
#         # Εάν ο επιλεγμένος χρήστης βρίσκεται στη λίστα των γειτόνων, αφαίρεσή του
#         if user_id in neighbors:
#             neighbors.remove(user_id)
#
#         # Υπολογισμός της Συνολικής Βαρύτητας Ακμής για κάθε γείτονα του επιλεγμένου χρήστη
#         total_edge_weights = {}
#         for neighbor in neighbors:
#             total_edge_weights[neighbor] = sum([data['weight'] for _, _, data in G.edges(neighbor, data=True)])
#
#         # Ταξινόμηση των γειτόνων βάσει της Συνολικής Βαρύτητας Ακμής σε φθίνουσα σειρά
#         sorted_neighbors = sorted(total_edge_weights, key=total_edge_weights.get, reverse=True)
#
#         # Εύρεση του πρώτου δίσκου που ο χρήστης δεν έχει επιλέξει από τον πιο συνδεδεμένο γείτονά του
#         recommended_album = None
#         for neighbor in sorted_neighbors:
#             common_albums = set(G.neighbors(neighbor)) - set(G.neighbors(user_id))
#             if common_albums:
#                 recommended_album = common_albums.pop()
#                 break
#
#         # Εάν υπάρχει προτεινόμενος δίσκος, εκτύπωση του τίτλου του
#         if recommended_album:
#             query = "SELECT title FROM album WHERE id = %s"
#             cursor.execute(query, (recommended_album,))
#             album_title = cursor.fetchone()[0]
#             recommended_albums[user_id] = album_title
#
# # Εκτύπωση των προτεινόμενων δίσκων για κάθε χρήστη
# for user_id, album_title in recommended_albums.items():
#     print("User ID:", user_id, "- Recommended Album Title:", album_title)


# Definition of the fitness function
def fitness_function(individual):
    total_desire = np.sum(user_money_rates[:, 1:] * individual[:, np.newaxis])  # We remove the first column of user_money_rates
    return total_desire,

users_money = 200 + np.ceil(100 * np.random.rand(100))
user_money_rates = np.empty_like(np.append(users_money[0], np.random.randint(5, size=50) + 1))
for i in users_money:
    user_money_rates = np.vstack([user_money_rates, np.append(i, np.random.randint(5, size=50) + 1)])
user_money_rates = np.delete(user_money_rates, (0), axis=0)

album_price = np.random.randint(50, size=100) + 1

# Definition of the parameters of the genetic algorithm
population_size = 100
num_generations = 50
crossover_probability = 0.8
mutation_probability = 0.2

# Creating the genetic algorithm objects
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.randint, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(album_price))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness_function)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Initialize the initial population distribution
population = toolbox.population(n=population_size)

# Running the genetic algorithm
for generation in range(num_generations):
    offspring = algorithms.varAnd(population, toolbox, cxpb=crossover_probability, mutpb=mutation_probability)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

# Finding the best result
best_individual = tools.selBest(population, k=1)[0]
best_disk_selection = np.where(best_individual == 1)[0]

# Random selection of discs
random_selection = np.random.choice(len(album_price), size=len(best_disk_selection), replace=False)

# Print the selected disks from the genetic algorithm
print("Genetic algorithm selection:")
for disk_index in best_disk_selection:
    print("Disk", disk_index + 1, "- Price:", album_price[disk_index])

# Print the selected disks from the random selection
print("\nRandom selection of discs:")
for disk_index in random_selection:
    print("Disk", disk_index + 1, "- Price:", album_price[disk_index])

# Comparison of results
common_disks = np.intersect1d(best_disk_selection, random_selection)
print("\nShared disks:")
for disk_index in common_disks:
    print("Disk", disk_index + 1, "- Price:", album_price[disk_index])

genetic_only_disks = np.setdiff1d(best_disk_selection, random_selection)
print("\nDrives only by genetic algorithm:")
for disk_index in genetic_only_disks:
    print("Disk", disk_index + 1, "- Price:", album_price[disk_index])

random_only_disks = np.setdiff1d(random_selection, best_disk_selection)
print("\nDiscs only from random selection:")
for disk_index in random_only_disks:
    print("Disk", disk_index + 1, "- Price:", album_price[disk_index])

# Calculation of total value of discs by the genetic algorithm
genetic_total_price = np.sum(album_price[best_disk_selection])

# Calculation of total value of discs from the random selection
random_total_price = np.sum(album_price[random_selection])

# Print results
print("\nTotal disk value from the genetic algorithm:", genetic_total_price)
print("Total value of discs from the random selection:", random_total_price)

if genetic_total_price > random_total_price:
    print("Genetic algorithm provides better results.")
elif genetic_total_price < random_total_price:
    print("Random selection provides better results.")
else:
    print("Genetic algorithm and random selection provide identical results.")



cnx.commit()
cursor.close()
cnx.close()