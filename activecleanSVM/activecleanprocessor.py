from sklearn.linear_model import SGDClassifier
from activecleanSVM.detector import Detector
from activecleanSVM.sampler import DetectorSampler, UniformSampler
from activecleanSVM.cleaner import Cleaner
from activecleanSVM.updater import Updater

class ActiveCleanProcessor:
    def __init__(self, user_clf, dirty_data, indices, batch_size, ownFilepath,
                 step_size, vectorizer, featuriser, process_cleaned_df):
        """
        Initialises an ActiveClean object

        :param user_clf: A classifier that the user hopes to improve
        :param dirty_data: A tuple of (X_dirty,Y_dirty) which is the data to be cleaned
        :type dirty_data: tuple
        :param indices: A tuple (dirty_indices, clean_indices). Initially, all data is considered to be dirty
        :type indices: tuple
        :param batch_size: Size of data to sample
        :type batch_size: int
        :param ownFilepath: User defined filepath to store xlsx files
        :param step_size: learning rate in SGD algorithm
        :type: float
        :param vectorizer: Used to featurise the data
        :param featuriser: A function that transforms the data that can be input into the vectorizer
        :type featuriser: function
        :param process_cleaned_df: A user-provided function to process a cleaned excel sheet based on format of data
        :type process_cleaned_df: function
        """
        self.clf = user_clf

        self.dirty_indices = indices[0]
        self.clean_indices = indices[1]

        # training data to clean
        self.X_full = dirty_data[0]
        self.Y_full = dirty_data[1]

        self.batch_size = batch_size
        self.step_size = step_size
        self.ownFilepath = ownFilepath

        self.vectorizer = vectorizer
        self.featurise_text = featuriser
        self.process_cleaned_df = process_cleaned_df

        self.detector = Detector(SGDClassifier(loss="log", alpha=1e-6, max_iter=200, fit_intercept=True))
        self.detectorsampler = DetectorSampler(self.featurise_text(self.X_full, self.vectorizer), self.dirty_indices,
                                               self.detector, batch_size)
        self.uniformsampler = UniformSampler(dirty_data, self.dirty_indices, batch_size)
        self.cleaner = Cleaner(process_cleaned_df, ownFilepath)

        self.total_labels = [] # a list of (indice, indice in clean_indice)
        self.dirty_sample_indices = []
        self.sampling_prob = []

        self.iteration_count = 0

    def start(self, num_records_to_clean):
        """
        Starts the ActiveClean object by sampling and providing the sampled data to the user

        :param num_records_to_clean: Number of datapoints that user wants to clean
        :type num_records_to_clean: int
        """
        if num_records_to_clean < self.batch_size:
            return Exception("Please provide smaller batch size")

        featurised_x_dirty = self.featurise_text(self.X_full, self.vectorizer)

        # Train user clf first
        self.clf = self.clf.fit(featurised_x_dirty, self.Y_full)

        # Sample dirty data
        self.dirty_sample_indices, self.sampling_prob = self.uniformsampler.sample()

        # Provide sample to clean to user in Excel file
        self.cleaner.provide_sample(self.X_full, self.Y_full, self.dirty_sample_indices)

        print("Done initialising")

    def runNextIteration(self):
        """
        Runs the process of updating the model, sampling dirty data and providing sampled data to user
        after the user has cleaned the previous sampled data
        """
        if len(self.dirty_indices) < self.batch_size:
            print("Not enough dirty data to sample")
            return self.clf, self.X_full, self.Y_full

        print("Updating data after cleaning...")

        self.X_full, self.Y_full = self.cleaner.update_with_cleaned_data(self.X_full, self.Y_full,
                                                                         self.dirty_sample_indices)
        print("Preparing data for Updater...")
        num_cleaned = len(self.clean_indices)
        total_data_size = len(self.Y_full)
        proportion_cleaned = num_cleaned / total_data_size
        proportion_dirty = 1 - proportion_cleaned
        proportions = (proportion_dirty, proportion_cleaned)

        # already cleaned
        if len(self.clean_indices) != 0:
            cleaned_X = [self.X_full[clean_indice] for clean_indice in self.clean_indices]
            cleaned_Y = [self.Y_full[clean_indice] for clean_indice in self.clean_indices]
            cleaned_data = (self.featurise_text(cleaned_X, self.vectorizer), cleaned_Y)
            num_cleaned = len(cleaned_Y)
        else:
            cleaned_data = ([], [])  # empty data will cause error with featuriser
            num_cleaned = 0

        # newly-cleaned
        sample_X = [self.X_full[sample_indice] for sample_indice in self.dirty_sample_indices]
        sample_Y = [self.Y_full[sample_indice] for sample_indice in self.dirty_sample_indices]
        sample_data = (self.featurise_text(sample_X, self.vectorizer), sample_Y)

        self.iteration_count += 1

        print("Updating updater...")
        updater = Updater(self.clf, self.batch_size, self.step_size)
        # self.clf = updater.retrain(self.featuriser(self.X_full), self.Y_full)
        self.clf = updater.update(sample_data, self.sampling_prob, cleaned_data, proportions,
                                  self.iteration_count, num_cleaned)

        # Update cleaned data
        self.dirty_indices = [index for index in self.dirty_indices if index not in self.dirty_sample_indices]
        for index in self.dirty_sample_indices:
            self.clean_indices.append(index)

        print("Updating detector...")
        # Update detector sampler
        self.total_labels.extend([(index, True) for index in self.dirty_sample_indices])
        self.detector.update_classifier(self.total_labels, self.featurise_text(self.X_full, self.vectorizer))
        self.detectorsampler.updateDetector(self.detector)

        print("Starting sampling process again...")
        # Sample dirty data
        self.dirty_sample_indices, self.sampling_prob = self.detectorsampler.sample()

        # Provide dirty samples to user to clean
        self.cleaner.provide_sample(self.X_full, self.Y_full, self.dirty_sample_indices)

        print("Number of records cleaned: " + str(len(self.clean_indices)))
        print("Iterations of retraining done: " + str(self.iteration_count))

        return self.clf, self.X_full, self.Y_full

    def getClassifier(self):
        return self.clf