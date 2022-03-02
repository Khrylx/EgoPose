import os
import argparse
from mujoco_py import load_model_from_path, MjSim
from envs.common.mjviewer import MjViewer
from mocap.skeleton import Skeleton

parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', default=True)
parser.add_argument('--mocap-id', type=str, default='1205')
parser.add_argument('--template-id', type=str, default='humanoid_template')
parser.add_argument('--bvh-id', type=str, default='ROM')
parser.add_argument('--model-id', type=str, default='humanoid_1205_orig')
args = parser.parse_args()

bvh_file = '~/datasets/egopose/bvh/%s_%s.bvh' % (args.mocap_id, args.bvh_id)
template_file = 'assets/mujoco_models/template/%s.xml' % args.template_id
model_file = 'assets/mujoco_models/%s.xml' % args.model_id

exclude_bones = {'Thumb', 'Index', 'Middle', 'Ring', 'Pinky', 'End', 'Toe'}
spec_channels = {'LeftForeArm': ['Zrotation'], 'RightForeArm': ['Zrotation'],
                 'LeftLeg': ['Xrotation'], 'RightLeg': ['Xrotation']}
skeleton = Skeleton()
skeleton.load_from_bvh(os.path.expanduser(bvh_file), exclude_bones, spec_channels)
skeleton.write_xml(model_file, template_file)

model = load_model_from_path(model_file)
sim = MjSim(model)
viewer = MjViewer(sim)

while args.render:
    viewer.render()
