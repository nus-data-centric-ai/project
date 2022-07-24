import numpy as np

class BaseSamplerInterface:
    '''BaseSampler acts as an interface for all the samplers'''
    def __init__(self, full_data, dirty_indices, batch_size=50) -> None:
        self.full_data = full_data
        self.batch_size = batch_size
        self.dirty_indices = dirty_indices

    def sample(self):
        return NotImplementedError

    def remove_dirty(self, dirty_indices):
        self.dirty_indices = [i for i in self.dirty_indices if i not in dirty_indices]

    '''update the dirty indices'''
    def update_dirty(self, new_dirty_indices):
        self.dirty_indices = new_dirty_indices

class UniformSampler(BaseSamplerInterface):
    '''Uniform sampler samples the data uniformly'''
    def __init__(self, full_data, dirty_indices, batch_size=50) -> None:
        super().__init__(full_data, dirty_indices, batch_size)

    def sample(self):
        samples = np.random.choice(self.dirty_indices, self.batch_size)
        sampling_prob = [1/len(self.dirty_indices)] * self.batch_size
        return samples, sampling_prob

#TODO improve the code
class DetectorSampler(BaseSamplerInterface):
    '''Detectorsampler uses the detector provided to
    find the dirty samples and prioritizes sampling them'''

    def __init__(self, full_data, dirty_indices, detector, batch_size=50) -> None:
        super().__init__(full_data, dirty_indices, batch_size)
        self.detector = detector

    def updateDetector(self, detector):
        self.detector = detector

    def sample(self):
        dirty_probability = self.detector.get_error_prob(self.dirty_indices, self.full_data)
        if dirty_probability is None:
            samples = np.random.choice(self.dirty_indices, self.batch_size)
            sampling_prob = [1 / len(self.dirty_indices)] * self.batch_size # uniform
        else:
            samples = np.random.choice(self.dirty_indices, self.batch_size, p = dirty_probability)
            sampling_prob = dirty_probability[samples]

        super().remove_dirty(samples)
        return samples, sampling_prob

