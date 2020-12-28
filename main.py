from PIL import Image
from torchvision import models, transforms
import torch
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torchsnooper

x = Image.open("bird.jpg")
# preprocess = transforms.Compose([
#     transforms.Resize((400, 400)),
#     transforms.ToTensor()
# ])
# xp = preprocess(x) #(3, 400, 400)
# xp = torch.unsqueeze(xp, dim=0)
#
# net = models.resnet34()
# path = "resnet34-333f7ec4.pth"
# net.load_state_dict(torch.load(path))
#
# features = []
#
# def hook_feature(module, input, output):
#     features.append(output.data.numpy())
#
# net.layer2.register_forward_hook(hook=hook_feature)
# net.layer3.register_forward_hook(hook=hook_feature)
# net.layer4.register_forward_hook(hook=hook_feature)
#
# net(xp)
#
#
# np.save("feature_maps2", features[0])
# np.save("feature_maps3", features[1])
# np.save("feature_maps4", features[2])


feature_maps_name = ["feature_maps2.npy", "feature_maps3.npy", "feature_maps4.npy"]


# plt.figure()
# for i in range(3):
#     feature_map = np.load(feature_maps_name[i])
#     features_map = np.squeeze(feature_map)
#     feature_map = features_map.transpose(1, 2, 0)
#     for j in range(8):
#         plt.subplot(3, 8, i*8+j+1)
#         plt.imshow(feature_map[:,:,j])
# plt.show()
with torchsnooper.snoop():
    feature_map = np.load(feature_maps_name[1])
    feature_map = np.squeeze(feature_map, axis=0)
    feature_map_hw = feature_map.reshape((feature_map.shape[0], -1))
    similar = feature_map_hw @ feature_map_hw.transpose()
    similar_sum = np.sum(similar, axis=1, keepdims=True)
    similar_m = np.divide(similar, similar_sum)
    # fig = plt.figure()
    # plt.imshow(similar)
    # plt.show()
    fig = plt.figure()
    weight = similar_m[:,:,np.newaxis,np.newaxis]
    feature_attrs = np.sum(weight * feature_map, axis=1)

    # for i, weight_v in enumerate(similar_m):
    #     weight =weight_v[:, np.newaxis, np.newaxis]
    #     feature_attr = np.sum(feature_map * weight, axis=0)
    #     if i < 16:
    #         plt.subplot(4, 4, i+1)
    #         plt.imshow(feature_attr)
    feature_attrs_show = feature_attrs.transpose(1, 2, 0)
    for i in range(len(similar_m)):
        if i < 16:
            plt.subplot(4, 4, i+1)
            plt.imshow(feature_attrs_show[:,:,i])
        else:
            break

    avg_struc = np.divide(np.sum(feature_attrs_show, axis=2), feature_attrs_show.shape[2])
    fig1plt.figure()
    plt.imshow(avg_struc)
    plt.show()
