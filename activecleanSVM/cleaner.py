import pandas as pd

class Cleaner:
    def __init__(self, process_cleaned_df, own_filepath):
        """
        Initialises a Cleaner object, which is used to output the sampled dirty data to the user
        and process cleaned data from user

        :param process_cleaned_df: A user-provided function to process a cleaned excel sheet based on format of data
        :type process_cleaned_df: function
        :param own_filepath: A string consisting of the absolute path to the folder which contains
        the excel sheet output to user (user-specified filepath)
        """
        self.process_cleaned_df = process_cleaned_df #user defined function
        self.cleaned_data_filepath = own_filepath + "SampleToClean_Iteration.xlsx"

    def update_with_cleaned_data(self, X_full, Y_full, dirty_sample_indices):
        """
        Takes in cleaned data from user to update data
        :param X_full: Full X data (unfeaturised)
        :type X_full: list
        :param Y_full: Full Y data, which is the labels for the X data
        :type Y_full: list
        :param dirty_sample_indices: List of integers that show that indices of data that was selected to be
        cleaned by user
        :type dirty_sample_indices: list
        """
        # Process cleaned sample from user
        unprocessed_cleaned_sample_df = pd.read_excel(self.cleaned_data_filepath, index_col=0)
        unprocessed_cleaned_sample_df.index.name = 'Index'
        cleaned_sample_df = self.process_cleaned_df(unprocessed_cleaned_sample_df)

        # Update existing data with cleaned data
        full_data_df = pd.DataFrame(X_full, columns=["Movie", "Plot", "Genres"])
        full_data_df["Y label"] = Y_full
        full_data_df.loc[dirty_sample_indices, :] = cleaned_sample_df[:]
        Y_full = full_data_df["Y label"].values.tolist()
        X_data_df = full_data_df.drop(columns=["Y label"])
        X_full = list(X_data_df.itertuples(index=False))
        X_full = [tuple(namedtuple) for namedtuple in X_full]
        return (X_full, Y_full)

    def provide_sample(self,X_full, Y_full, dirty_sample_indices):
        """
        Takes in sampled indices from Sampler to output to user in an Excel sheet for cleaning
        :param X_full: Full X data (unfeaturised)
        :type X_full: list
        :param Y_full: Full Y data, which is the labels for the X data
        :type Y_full: list
        :param dirty_sample_indices: List of integers that show that indices of data that was selected to be
        cleaned by user
        :type dirty_sample_indices: list
        """
        full_data = pd.DataFrame(X_full, columns=["Movie", "Plot", "Genres"])
        full_data["Y label"] = Y_full
        dirty_sample_df = full_data.filter(items=dirty_sample_indices, axis=0)
        dirty_sample_df.to_excel(self.cleaned_data_filepath)
        print("Please clean the data in " + self.cleaned_data_filepath)
