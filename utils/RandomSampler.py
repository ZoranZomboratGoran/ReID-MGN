import random
import collections
from torch.utils.data import sampler
import copy
from opt import opt

class RandomSampler(sampler.Sampler):
    def __init__(self, data_source, batch_id, batch_image):
        super(RandomSampler, self).__init__(data_source)

        self.data_source = data_source
        self.batch_image = batch_image
        self.batch_id = batch_id

        self._id2index = collections.defaultdict(list)
        for idx, path in enumerate(data_source.imgs):
            _id = data_source.id(path)
            self._id2index[_id].append(idx)

    def __iter__(self):
        unique_ids = self.data_source.unique_ids
        id2index = copy.deepcopy(self._id2index)

        imgs = []
        while len(unique_ids) >= self.batch_id:
            uid_batch = random.sample(unique_ids, self.batch_id)
            for uid in uid_batch:
                imgs.extend(self._sample(id2index[uid], self.batch_image))
                if id2index[uid] == []:
                    unique_ids.remove(uid)

        if self.data_source.dtype == 'test':
            test_samples = 200 * self.batch_image * self.batch_id
            return iter(imgs[:test_samples])

        return iter(imgs)

    def __len__(self):
        return len(self._id2index) * self.batch_image

    @staticmethod
    def _sample(population, k):
        population_ = population
        while len(population_) < k:
            population_ = population_ + population_[:k - len(population_)]
        sample = random.sample(population_, k)
        for s in sample:
            try:
                population.remove(s)
            except ValueError:
                pass
        return sample
