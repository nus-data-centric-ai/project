import numpy as np
import random
import pandas as pd

class IMDB:
    def __init__(self, plot_file_path, genre_file_path, yahoo_file_path):
        self.plot_file_path = plot_file_path
        self.genre_file_path = genre_file_path
        self.yahoo_file_path = yahoo_file_path

    def get_comedy_and_horror_movie_data(self):
        imdb_plot = self.load_plot_data()
        imdb_tags = self.load_genre_data()
        imdb_data = self.join_plot_tags(imdb_plot, imdb_tags)
        # Process Yahoo dataset which is used as reference to provide cleaner tags to IMDB dataset
        yahoo_data = self.load_yahoo_data()
        # All initial data is treated as dirty
        dirty_data = []
        for imdb_tuple in imdb_data:
            dirty_data.append(self.clean(imdb_tuple, yahoo_data))
        # Get movies labelled comedy or horror -- not really to features -- more of to filter out
        x, y = self.filter_movies(dirty_data)
        return x, y

    def load_plot_data(self):
        """
        Processes data from a list of a database of movies and their plots from IMDB.
        :return: Returns a dictionary of key = title and value = plot.
        :rtype: dict
        """
        filename = self.plot_file_path
        f = open(filename, "r", encoding="latin1")
        line = f.readline()
        lookup = {}

        cur_movie = ""
        while line != "":
            comps = line.split(":")
            if len(comps) > 1:
                if comps[0] == "MV":
                    cur_movie = self.clean_title(":".join(comps[1:]))

                if cur_movie not in lookup:
                    lookup[cur_movie] = ""

                if comps[0] == "PL":
                    lookup[cur_movie] = (
                            lookup[cur_movie] + " " + ":".join(comps[1:]).rstrip()
                    )

            line = f.readline()
        return lookup

    def load_genre_data(self):
        """
        Processes data from a list of a database of movies and their genres from IMDB.
        :return: Returns a dictionary of key = title and value = list of genres a movie is tagged with.
        :rtype: dict
        """
        filename = self.genre_file_path
        f = open(filename, "r", encoding="latin1")
        line = f.readline()
        start = False
        lookup = {}

        while line != "":
            if "==================" in line:
                start = True

            if start:
                comps = line.split("\t")
                if len(comps) > 2:
                    title = " ".join(comps[0: len(comps) - 1])
                    realtitle = self.clean_title(title)

                    if realtitle not in lookup:
                        lookup[realtitle] = [comps[-1].rstrip()]
                    else:
                        lookup[realtitle].append(comps[-1].rstrip())

            line = f.readline()

        return lookup

    def join_plot_tags(self, plot, tags):
        """
        Filters movies that were found in both the plot and tag dataset then combines the data together.

        :param plot: A dictionary of key = title and value = plot
        :type plot: dict
        :param tags:  A dictionary of key = title and value = list of genres a movie is tagged with
        :type tags: dict
        :return: A list of tuples where each tuple is (title, plot, genre list a movie is tagged with) if title is in both plot and tag dataset
        :rtype: list
        """
        join_keys = [k for k in plot if k in tags]
        result = []
        for i in join_keys:
            result.append((i, plot[i], tags[i]))
        return result

    def clean_title(self, title):
        """
          Cleans movie title extracted from IMDB data.
          :param title: Dirty title
          :type data: str
          :return: A cleaned movie title
          :rtype: str
        """
        realtitlec = title.split("(")[0]
        realtitle = realtitlec.strip('"')
        realtitle = realtitle.strip()

        # deal with the empty string problem
        if len(realtitle) == 0:
            realtitle = "N\A" + str(random.random())

        return realtitle.strip().lower()

    def load_yahoo_data(self):
        """
        Processes movie data from Yahoo dataset.

        :return: Returns a dictionary where the key = movie title and value = (plot, genre list)
        :rtype: dict
        """
        filename = self.yahoo_file_path
        f = open(filename, "r", encoding="latin1")
        line = f.readline()
        dataset = {}
        while line != "":
            comps = line.split("\t")
            title, plot, tags = comps[1], comps[2], comps[10]
            dataset[self.clean_title(title)] = (plot, tags)
            line = f.readline()
        return dataset

    def clean(self, imdb_movie_tuple, yahoo_movie_dict):
        """
        Cleans one tuple of the IMDB movie data by cross-referencing to a smaller but cleaner dataset from Yahoo.

        When there was a sufficiently close textual match (in terms of title string similarity),
        the Yahoo dataset’s category list and plot content was incorporated into existing IMDB data.

        :param imdb_movie_tuple: (movie_title, plot, genre list) extracted from IMDB data
        :type imdb_movie_tuple: genre
        :return: Returns a dictionary where the key = movie title and value = (plot, genre list) where plot and genre list could
        have supplemented data from Yahoo dataset
        :rtype: dict
        """
        imdb_title = imdb_movie_tuple[0]
        # if title from imdb in yahoo dataset and there is a genre tagged to the movie in the yahoo dataset
        if imdb_title in yahoo_movie_dict and yahoo_movie_dict[imdb_title][1][0] != "\\N":
            clean = yahoo_movie_dict[imdb_title]  # (yahoo_plot, yahoo_tags)
            yahoo_plot, yahoo_tags = clean[0], clean[1]
            imdb_plot = imdb_movie_tuple[1]
            return (imdb_title, imdb_title + " " + imdb_plot + " " + yahoo_plot, yahoo_tags)
        elif len(imdb_movie_tuple[2]) > 5:  # number of genres in imdb dataset > 5
            imdb_plot = imdb_movie_tuple[1]
            return (imdb_title, imdb_plot, ["\\N"])
        else:
            return imdb_movie_tuple

    ###################
    # DATA TO FEATURES#
    ###################

    def filter_movies(self, data, label1="Comedy", label2="Horror"):
        """
        Converts data processed to features for SGDClassifier.
        :param data: A list of tuples where each tuple is (title, plot, genre list a movie is tagged with) if title is in both plot and tag dataset
        :type data: list
        :param label1:  Comedy genre
        :type label1: str
        :param label2: Horror genre
        :type label2: str
        :return:
            - X - A list of (title, plot, genres) for movies which have comedy or horror as their genre
            - y - A list that contains 0 if a movie contained Comedy and 1 if a movie contained Horror as a genre
            - index - A list of int consisting of the indices of movies in data that either have Comedy or Horror as a genre
        """
        Y = []
        index = []
        inc = 0

        for i in data:
            tagstring = " ".join(i[2])
            if label2 in tagstring:
                Y.append(-1)
                index.append(inc)
            elif label1 in tagstring:
                Y.append(1)
                index.append(inc)

            inc = inc + 1

        X = [data[i] for i in index]
        return X, Y

    def translate_indices(self, indices_list_1, indices_list_2):
        """
        Gets a list of positions in indices_list_2 where indices_list_2[positions] is in indices_list_1.

        :param indices_list_1: A list of int representing indices
        :type indices_list_1: list
        :param indices_list_2:  A list of int representing indices
        :type indices_list_2: list
        :return: A list of positions (int) in indices_list_2 if indices_list_2[positions] is in indices_list_1.
        :rtype: list
        """
        indices_1_set = set(indices_list_1)
        return [absolute_indices for absolute_indices, relative_indices in enumerate(indices_list_2)
                if relative_indices in indices_1_set]
