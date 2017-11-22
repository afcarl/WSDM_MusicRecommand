import numpy as np

song2com = {}
song2id = {}
id2song = []
genre2id = {}
id2genre = []
artist2id = {}
id2artist = []
composer2id = {}
id2composer = []
lyricist2id = {}
id2lyricist = []
language2id = {}
id2language = []

# 0 : PAD or END, 1 : UNK
genre2id['pad'] = 0
genre2id['Nan'] = 1
id2genre.append('pad')
id2genre.append('Nan')
artist2id['pad'] = 0
artist2id['Nan'] = 1
id2artist.append('pad')
id2artist.append('Nan')
composer2id['pad'] = 0
composer2id['Nan'] = 1
id2composer.append('pad')
id2composer.append('Nan')
lyricist2id['pad'] = 0
lyricist2id['Nan'] = 1
id2lyricist.append('pad')
id2lyricist.append('Nan')

genre_max = 8
artist_max = 23
composer_max = 23
lyricist_max = 23

song_f = open("../dataset/filtered_songs.csv", "r")
lines = song_f.readlines()

for i, line in enumerate(lines):
    if i == 0:
        continue

    genre_list = []
    artist_list = []
    composer_list = []
    lyricist_list = []

    tokens = line.strip().split(',')

    song_id = tokens[0]
    if song2id.get(song_id) == None:
        song2id[song_id] = len(id2song)
        id2song.append(song_id)

    song_length = int(tokens[1])

    # genre numbering meaning??
    genre_idx = tokens[2].split('|') if len(tokens[2])>0 else ['Nan']
    for genre_id in genre_idx:
        genre_id = genre_id.strip()
        if genre2id.get(genre_id) == None:
            genre2id[genre_id] = len(genre2id) 
            id2genre.append(genre_id)
        genre_list.append(genre2id.get(genre_id))
    genre_list = genre_list + [0] * (genre_max - len(genre_idx)) if len(genre_idx) < genre_max else genre_list
    genre_list += [len(genre_idx)]

    # group artist eg) MFBTY Treatment
    artist_names = tokens[3].split('|') if len(tokens[3])>0 else ['Nan']
    for artist_name in artist_names:
        artist_name = artist_name.strip()
        if artist2id.get(artist_name) == None:
            artist2id[artist_name] = len(artist2id) 
            id2artist.append(artist_name)
        artist_list.append(artist2id.get(artist_name))
    artist_list = artist_list + [0] * (artist_max - len(artist_names)) if len(artist_names) < artist_max else artist_list
    artist_list += [len(artist_names)]

    composers = tokens[4].split('|') if len(tokens[4])>0 else ['Nan']
    for composer in composers:
        composer = composer.strip()
        if composer2id.get(composer) == None:
            composer2id[composer] = len(composer2id) 
            id2composer.append(composer)
        composer_list.append(composer2id.get(composer))
    composer_list = composer_list + [0] * (composer_max - len(composers)) if len(composers) < composer_max else composer_list
    composer_list += [len(composers)]

    lyricists = tokens[5].split('|') if len(tokens[5])>0 else ['Nan']
    for lyricist in lyricists:
        lyricist = lyricist.strip()
        if lyricist2id.get(lyricist) == None:
            lyricist2id[lyricist] = len(lyricist2id) 
            id2lyricist.append(lyricist)
        lyricist_list.append(lyricist2id.get(lyricist))
    lyricist_list = lyricist_list + [0] * (lyricist_max - len(lyricists)) if len(lyricists) < lyricist_max else lyricist_list
    lyricist_list += [len(lyricists)]

    language = tokens[6]
    if language2id.get(language) == None:
        language2id[language] = len(language2id)
        id2language.append(language)
    language = language2id.get(language)

    song2com[song_id] = [song_length] + genre_list + artist_list + composer_list + lyricist_list + [language]

'''
np.save("npy/all_song2id.npy", song2id)
np.save("npy/all_id2song.npy", id2song)
np.save("npy/genre2id.npy", genre2id)
np.save("npy/id2genre.npy", id2genre)
'''
np.save("npy/filtered_artist2id.npy", artist2id)
np.save("npy/filtered_id2artist.npy", id2artist)
'''
np.save("npy/composer2id.npy", composer2id)
np.save("npy/id2composer.npy", id2composer)
np.save("npy/lyricist2id.npy", lyricist2id)
np.save("npy/id2lyricist.npy", id2lyricist)
np.save("npy/language2id.npy", language2id)
np.save("npy/id2language.npy", id2language)
np.save("npy/all_song2com.npy", song2com)
'''
