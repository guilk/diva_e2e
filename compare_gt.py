import os
import cPickle as pickle
import numpy as np


if __name__ == '__main__':
    with open('./new_data.pkl', 'rb') as input_file:
        new_data = pickle.load(input_file)

    with open('./old_data.pkl', 'rb') as input_file:
        old_data = pickle.load(input_file)

    print(len(new_data['num_gts']))
    for new_num_gt, old_num_gt in zip(new_data['num_gts'], old_data['num_gts']):
        assert new_num_gt[0] == old_num_gt[0]

    for new_gt_wins, old_gt_wins in zip(new_data['gt_twins'], old_data['gt_twins']):
        # assert new_num_gt[0] == old_num_gt[0]
        assert np.allclose(new_gt_wins[0], old_gt_wins[0])