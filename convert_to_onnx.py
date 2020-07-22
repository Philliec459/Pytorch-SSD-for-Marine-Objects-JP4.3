from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite

import sys
import torch.onnx
import onnx


if len(sys.argv) < 3:
    print('Usage: python convert_to_caffe2_models.py <net type: mobilenet-v1-ssd|others>  <model path>')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]

label_path = sys.argv[3]

# set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('running on device ' + str(device))

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)
net.to(device)
net.eval()

model_path = f"models/{net_type}.onnx"
init_net_path = f"models/{net_type}_init_net.pb"
init_net_txt_path = f"models/{net_type}_init_net.pbtxt"
predict_net_path = f"models/{net_type}_predict_net.pb"
predict_net_txt_path = f"models/{net_type}_predict_net.pbtxt"

batch_size = 1 # 4

dummy_input = torch.randn(batch_size, 3, 300, 300).cuda()
torch.onnx.export(net, dummy_input, model_path, verbose=True, input_names=['input_0'], output_names=['scores', 'boxes'])

