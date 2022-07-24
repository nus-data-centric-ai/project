class Updater:
    def __init__(self, clf, batch_size, step_size):
        """
        Initialises an Updater object, which is used to update the parameters of the user given classifier
        :param clf: A model from user
        :param batch_size:  Size of data in each sample given to user
        :type batch_size: int
        :param step_size: learning rate in SGD algorithm
        :type: float
        """
        self.clf = clf
        self.batch_size = batch_size
        self.step_size = step_size

    def retrain(self, cleaned_X, cleaned_Y):
        """
        A function that can be used if the user is unaware of how to derive gradient functions for a more complicated model

        :param cleaned_X: A list of cleaned X values that have been featurised into TFIDF vectors
        :type cleaned_X: CSR Sparse Matrix of size (1, num_of_features)
        :param cleaned_Y: A list of integer labels consisting of 1 (Horror) and -1 (Comedy)
        :type cleaned_Y: list

        :returns: A updated classifier
        """
        return self.clf.partial_fit(cleaned_X, cleaned_Y)

    def update_step_size(self, batch_size, iteration_count):
        """
        Updates step size with every iteration based on equation proposed in research paper

        :param batch_size: Size of sample given to user to clean
        :type batch_size: int
        :param iteration_count: Number of retrainings done
        :type iteration_count: int
        """

        self.step_size = self.step_size / (batch_size * iteration_count)  # inverse scaling is used

    def get_gradients_weight_cleaned(self, data):
        """
        Calculates gradient based on already cleaned data

        :param data: A tuple of (X,Y) where X is featurised cleaned data and Y contains the corresponding labels
        :type data: tuple
        :returns: A list of gradients to update the weights
        :rtype:list
        """
        # hinge loss = max(0, 1 - y * f(x)) where f(x) = wT*x + b where y= -1 or 1
        # L = 1 - y*x*wT -y*b
        # dw/dL = -y*x
        X, Y = data  # X is featurised
        num_training_examples = len(Y)
        weights = self.clf.coef_[0]
        final_gradient = [0] * len(weights)
        if num_training_examples == 0:
            return final_gradient  # all 0 -- happens initially for the "already clean" dataset
        predictions = self.clf.predict(X)
        for i in range(num_training_examples):
            y_ = Y[i]  # (num_training_examples,)
            x_ = X[i]  # row of TFIDF vector (1, num_features)
            v = y_ * predictions[i]  # scalar
            gradient = [0] * len(weights)
            if v < 1:
                # x_ is (1, num_features) and -y is a scalar so all_gradients is (1, num_features)
                gradient = -y_ * x_  # (grad_w1, grad_w2, ...) -- csr_matrix
                gradient = gradient.toarray()[0]
            for i in range(len(final_gradient)):
                final_gradient[i] = final_gradient[i] + gradient[i] # [grad_w1, grad_w2 ...]

        weighted_final_gradient = [1 / num_training_examples * gradient for gradient in final_gradient]

        assert len(weighted_final_gradient) == len(weights)

        return weighted_final_gradient

    def get_gradients_weight_sample(self, data, sampling_probs):
        """
        Calculates gradient based on the cleaned sampled data

        :param data: A tuple of (X,Y) where X is featurised cleaned sample data and Y contains the corresponding labels
        :type data: tuple
        :returns: A list of gradients to update the weights
        :rtype:list
        """
        # hinge loss = max(0, 1 - y * f(x)) where f(x) = wT*x + b where y= -1 or 1
        # L = 1 - y*x*wT -y*b
        # dw/dL = -y*x
        X, Y = data  # X is featurised
        num_training_examples = len(Y)
        weights = self.clf.coef_[0]
        final_gradient = [0] * len(weights)
        if num_training_examples == 0:
            return final_gradient  # all 0 -- happens initially for the "already clean" dataset
        predictions = self.clf.predict(X)
        for i in range(num_training_examples):
            sampling_prob = sampling_probs[i]
            y_ = Y[i]  # (num_training_examples,)
            x_ = X[i]  # row of TFIDF vector (1, num_features)
            v = y_ * predictions[i]  # scalar
            gradient = [0] * len(weights)
            if v < 1:
                # x_ is (1, num_features) and -y is a scalar so all_gradients is (1, num_features)
                gradient = sampling_prob * -y_ * x_  # (grad_w1, grad_w2, ...) -- csr_matrix
                gradient = gradient.toarray()[0]
            for i in range(len(final_gradient)):
                final_gradient[i] = final_gradient[i] + gradient[i] # [grad_w1, grad_w2 ...]

        weighted_final_gradient = [1 / num_training_examples * gradient for gradient in final_gradient]

        assert len(weighted_final_gradient) == len(weights)

        return weighted_final_gradient

    def get_gradients_intercept_cleaned(self, data):
        """
        Calculates gradient based on already cleaned data

        :param data: A tuple of (X,Y) where X is featurised cleaned data and Y contains the corresponding labels
        :type data: tuple
        :returns: A list of gradient to update the intercept
        :rtype:list
        """
        # hinge loss = max(0, 1 - y * f(x)) where f(x) = wT*x + b
        # L = 1 - y*x*wT -y*b
        # db/dL = -y
        X, Y = data  # X is featurised
        num_training_examples = len(Y)
        weights = self.clf.intercept_
        final_gradient = 0
        if num_training_examples == 0:
            return [final_gradient]  # all 0 -- happens initially for the "already clean" dataset
        predictions = self.clf.predict(X)
        for i in range(num_training_examples):
            y_ = Y[i]  # (num_training_examples,)
            x_ = X[i]  # row of TFIDF vector (1, num_features)
            v = y_ * predictions[i]  # scalar
            gradient = 0
            if v < 1:
                # x_ is (1, num_features) and -y is a scalar so all_gradients is (1, num_features)
                gradient = -y_ # (grad_w1, grad_w2, ...) -- csr_matrix
            final_gradient = final_gradient + gradient # intercept is a scalar

        weighted_final_gradient = 1 / num_training_examples * final_gradient

        return [weighted_final_gradient]

    def get_gradients_intercept_sample(self, data, sampling_probs):
        """
        Calculates gradient for intercept based on the cleaned sampled data

        :param data: A tuple of (X,Y) where X is featurised cleaned sample data and Y contains the corresponding labels
        :type data: tuple
        :returns: A list of gradients to update the intercept
        :rtype:list
        """
        # hinge loss = max(0, 1 - y * f(x)) where f(x) = wT*x + b
        # L = 1 - y*x*wT -y*b
        # db/dL = -y
        X, Y = data  # X is featurised
        num_training_examples = len(Y)
        weights = self.clf.intercept_
        final_gradient = 0
        predictions = self.clf.predict(X)
        for i in range(num_training_examples):
            sampling_prob = sampling_probs[i]
            y_ = Y[i]  # (num_training_examples,)
            x_ = X[i]  # row of TFIDF vector (1, num_features)
            v = y_ * predictions[i]  # scalar
            gradient = 0
            if v < 1:
                # x_ is (1, num_features) and -y is a scalar so all_gradients is (1, num_features)
                gradient = sampling_prob * -y_  # (grad_w1, grad_w2, ...) -- csr_matrix
            final_gradient = final_gradient + gradient  # intercept is a scalar

        weighted_final_gradient = 1 / num_training_examples * final_gradient

        return [weighted_final_gradient]

    def evaluate_sample_gradient(self, sample_data, sampling_probs, proportion_dirty, get_gradients):
        """
        Calculates sample gradient based on the cleaned sampled data based on proposed equation from research paper

        :param sample_data: A tuple of (X,Y) where X is featurised newly-cleaned sample data and Y contains the corresponding labels
        :type sample_data: tuple
        :param sampling_probs: The sampling probability for each data point that was sampled
        :type sampling_probs: list
        :param proportion_dirty: Number of data points that are considered dirty out of all data points
        :type proportion_dirty: float
        :param get_gradients: A function to calculate the gradients to update a specific model parameter
        :type get_gradients: function

        :returns: A list of gradients which are floats
        :rtype: list
        """
        gradients = get_gradients(sample_data, sampling_probs)

        processed_sample_grads = [(1 / self.batch_size) * proportion_dirty * gradient for gradient in gradients]

        return processed_sample_grads

    def evaluate_cleaned_gradient(self, cleaned_data, num_cleaned, proportion_cleaned, get_gradients):
        """
        Calculates cleaned gradient based on the cleaned sampled data based on proposed equation from research paper

        :param cleaned_data: A tuple of (X,Y) where X is featurised already-cleaned data and Y contains the corresponding labels
        :type cleaned_data: tuple
        :param num_cleaned: Number of datapoints that has been cleaned by the user
        :type num_cleaned: int
        :param proportion_cleaned: Number of data points that are considered clean out of all data points
        :type proportion_cleaned: float
        :param get_gradients: A function to calculate the gradients to update a specific model parameter
        :type get_gradients: function

        :returns: A list of gradients which are floats
        :rtype: list
        """
        gradients = get_gradients(cleaned_data)
        if num_cleaned == 0:
            return gradients  # zero division error because of 1 / num_cleaned -- num_cleaned will initially be 0
        processed_cleaned_grads = [(1 / num_cleaned) * proportion_cleaned * gradient for gradient in gradients]
        return processed_cleaned_grads

    def evaluate_final_gradient(self, sample_data, sampling_probs, cleaned_data, proportions, get_gradient_cleaned,
                                get_gradient_sample, num_cleaned):
        """
        Calculates sample gradient based on the cleaned sampled data based on proposed equation from research paper

        :param sample_data: A tuple of (X,Y) where X is featurised newly-cleaned sample data and Y contains the corresponding labels
        :type sample_data: tuple
        :param sampling_probs: The sampling probability for each data point that was sampled
        :type sampling_probs: list
        :param proportion_dirty: Number of data points that are considered dirty out of all data points
        :type proportion_dirty: float
        :param get_gradients: A function to calculate the gradients to update a specific model parameter
        :type get_gradients: function

        :returns: A list of gradients/gradient which are floats
        :rtype: list
        """
        proportion_cleaned, proportion_dirty = proportions

        sample_gradients = self.evaluate_sample_gradient(sample_data, sampling_probs,
                                                         proportion_dirty, get_gradient_sample)

        clean_gradients = self.evaluate_cleaned_gradient(cleaned_data, num_cleaned, proportion_cleaned,
                                                         get_gradient_cleaned)

        assert len(sample_gradients) == len(clean_gradients)

        final_gradient = [sample_gradient + clean_gradient for (sample_gradient, clean_gradient) in
                          zip(sample_gradients, clean_gradients)]

        return final_gradient

    def get_regularisation_term(self):
        pass

    def update(self, sample_data, sampling_probs, cleaned_data, proportions, iteration_count, num_cleaned):
        """
        Updates model parameter for SVM based on proposed equation from research paper

        :param sample_data: A tuple of (X,Y) where X is featurised newly-cleaned sample data and Y contains the corresponding labels
        :type sample_data: tuple
        :param sampling_probs: The sampling probability for each data point that was sampled
        :type sampling_probs: list
        :param cleaned_data: A tuple of (X,Y) where X is featurised cleaned data and Y contains the corresponding labels
        :type cleaned_data: tuple
        :param proportions: A tuple of floats representing (proportion of data that is dirty, proportion of data that is clean)
        :type proportions: tuple
        :param iteration_count: Number of retraining that has been done
        :type iteration_count: int
        :param num_cleaned: Number of datapoints that have been cleaned
        :type num_cleaned: int

        :returns: An updated classifier
        """

        self.update_step_size(self.batch_size, iteration_count)

        weight_gradients = self.evaluate_final_gradient(sample_data, sampling_probs,
                                                       cleaned_data, proportions, self.get_gradients_weight_cleaned,
                                                       self.get_gradients_weight_sample, num_cleaned)

        adjusted_weight_gradients = [self.step_size * gradient for gradient in weight_gradients]

        self.clf.coef_ = self.clf.coef_ - adjusted_weight_gradients

        intercept_gradient = self.evaluate_final_gradient(sample_data, sampling_probs,
                                                       cleaned_data, proportions, self.get_gradients_intercept_cleaned,
                                                       self.get_gradients_intercept_sample, num_cleaned)

        adjusted_intercept_gradient = [self.step_size * gradient for gradient in intercept_gradient]

        self.clf.intercept_ = self.clf.intercept_ - adjusted_intercept_gradient[0]

        return self.clf
