import os
import argparse
import torch
from networks.unet_proto_3d import UNetProto
from utils.test_3d_patch import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/LA', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='', help='exp_name')
parser.add_argument('--model', type=str,  default='vnet', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-procssing?')
parser.add_argument('--det', type=int, default=0, help='whether use deterministic representation')
parser.add_argument('--labelnum', type=int, default=8, help='labeled data')
parser.add_argument('--stage_name',type=str, default='self_train', help='self_train or pre_train')
parser.add_argument('--temp', type=float, default=0.5, help='temperature for softmax in proto matching')
parser.add_argument('--sim_mode', type=str, default='dist', help='similarity computation method [dist,euclidean]')
parser.add_argument('--embed_dim', type=int, default=32, help='the dimension of the mu')
parser.add_argument('--sigma_mode', type=str, default='diagonal',help='the type of covariance matrix [radius,diagonal]')
parser.add_argument('--sigma_trans_mode', type=str, default='sigmoid',help='the way to transform sigma_raw to sigma')  # softplus, sigmoid, sigmoidLearn

FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = ""
num_classes = 2
test_save_path = "../conmodel/LA/{}/{}_labeled/predictions/".format(FLAGS.exp,FLAGS.labelnum)
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)
with open(FLAGS.root_path + '/test.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path + "/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]


def test_calculate_metric():

    model =UNetProto(
            backbone=FLAGS.model,
            inchannel=1,
            nclasses=num_classes,
            proto_mu=None,
            proto_sigma=None,
            embed_dim=FLAGS.embed_dim,
            sigma_mode=FLAGS.sigma_mode,
            sigma_trans_mode=FLAGS.sigma_trans_mode,
            sim_mode=FLAGS.sim_mode,
            temp=FLAGS.temp,
            momentum=False,
        ).cuda()

    save_model_path = ''
    model.load_state_dict(torch.load(save_model_path))
    save_mu_path = os.path.join(snapshot_path, 's_proto_mu.pt'.format(FLAGS.model))
    save_sigma_path = os.path.join(snapshot_path, 's_proto_sigma.pt'.format(FLAGS.model))
    model.prototypes_sigma = torch.load(save_sigma_path)
    model.prototypes_mu = torch.load(save_mu_path)

    print("init weight from {}".format(save_model_path))

    model.eval()

    avg_metric = test_all_case(model, image_list, num_classes=num_classes,
                           patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                           save_result=True, test_save_path=test_save_path,
                           metric_detail=FLAGS.detail, nms=FLAGS.nms)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)

# python test_LA.py --model 0214_re01 --gpu 0
