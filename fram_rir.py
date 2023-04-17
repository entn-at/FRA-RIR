"""!
Author: Rongzhi Gu (lorrygu), Yi Luo (oulyluo)
Copyright: Tencent AI Lab
"""

import numpy as np
import torch
from torchaudio.transforms import Resample
from torchaudio.functional import highpass_biquad


def sample_room_dim(min_room=None, max_room=None):
    if min_room is None:
        min_room = [3, 3, 2.5]
    if max_room is None:
        max_room = [10, 10, 4]
    return np.random.uniform(np.array(min_room), np.array(max_room))


def sample_mic_array_pos(mic_arch, array_pos=None, room_dim=None,
                            min_dis_wall=None):
    """
    Generate the microphone array position according to the given microphone architecture (geometry)
    :param mic_arch: np.array with shape [n_mic, 3]
                    the relative 3D coordinate to the array_pos in (m)
                    e.g., 2-mic LA (left->right) [[-0.1, 0, 0], [0.1, 0, 0]];
                    e.g., 4-mic CA (north->clockwise) [[0, 0.035, 0], [0.035, 0, 0], [0, -0.035, 0], [-0.035, 0, 0]]
    :param array_pos: array CENTER position in (m)
    :param room_dim: room dimension in (m)
    :param min_dis_wall: minimum distance from the wall in (m)
    :return
        mic_pos: microphone array position in (m) with shape [n_mic, 3]
        array_pos: array CENTER position in (m) with shape [1, 3]
    """
    if min_dis_wall is None:
        min_dis_wall = [0.5, 0.5, 0.5]
    if room_dim is None:
        room_dim = [9, 7.5, 3.5]

    def rotate(angle, valuex, valuey):
        rotate_x = valuex * np.cos(angle) + valuey * np.sin(angle)  # [nmic]
        rotate_y = valuey * np.cos(angle) - valuex * np.sin(angle)
        return np.stack([rotate_x, rotate_y, np.zeros_like(rotate_x)], -1)  # [nmic, 3]

    mic_array_center = np.mean(mic_arch, 0, keepdims=True)  # [1, 3]
    max_radius = max(np.linalg.norm(mic_arch - mic_array_center, axis=-1))
    array_pos = np.random.uniform(np.array(min_dis_wall) + max_radius,
                                  np.array(room_dim) - np.array(min_dis_wall) - max_radius).reshape(1, 3)
    mic_pos = array_pos + mic_arch
    # assume the array is always horizontal
    rotate_azm = np.random.uniform(-np.pi, np.pi)
    mic_pos = array_pos + rotate(rotate_azm, mic_arch[:, 0], mic_arch[:, 1])  # [n_mic, 3]

    return mic_pos, array_pos


def sample_src_pos(room_dim=None, num_src=2, array_pos=None,
                   min_mic_dis=0.5, max_mic_dis=5, min_dis_wall=None):
    if min_dis_wall is None:
        min_dis_wall = [0.5, 0.5, 0.5]
    if array_pos is None:
        array_pos = [4.5, 3.75, 1.75]
    if room_dim is None:
        room_dim = [9, 7.5, 3.5]

    # random sample the source positon
    src_pos = []
    for _ in range(num_src):
        while True:
            pos = np.random.uniform(np.array(min_dis_wall), np.array(
                room_dim) - np.array(min_dis_wall))
            dis = np.linalg.norm(pos - np.array(array_pos))

            if dis >= min_mic_dis and dis <= max_mic_dis:
                src_pos.append(pos)
                break
    return src_pos


def sample_mic_arch(n_mic, mic_spacing=None, bounding_box=None):
    if mic_spacing is None:
        mic_spacing = [0.02, 0.10]
    if bounding_box is None:
        bounding_box = [0.08, 0.12, 0]

    sample_n_mic = np.random.randint(n_mic[0], n_mic[1] + 1)
    if sample_n_mic == 1:
        mic_arch = np.array([[0, 0, 0]])
    else:
        mic_arch = []
        while len(mic_arch) < sample_n_mic:
            this_mic_pos = np.random.uniform(
                np.array([0, 0, 0]), np.array(bounding_box))

            if len(mic_arch) != 0:
                ok = True
                for other_mic_pos in mic_arch:
                    this_mic_spacing = np.linalg.norm(this_mic_pos - other_mic_pos)
                    if this_mic_spacing < mic_spacing[0] or this_mic_spacing > mic_spacing[1]:
                        ok = False
                        break
                if ok:
                    mic_arch.append(this_mic_pos)
            else:
                mic_arch.append(this_mic_pos)
        mic_arch = np.stack(mic_arch, 0)  # [nmic, 3]
    return mic_arch


def FRA_RIR_ADHOC(mic_arch, sr=16000, rt60=None, room_dim=None,
                  array_pos=None, src_pos=None, mic_pos=None,
                  num_src=2, direct_range=None,
                  alpha=None, image_coeff=None, a=-2.0, b=2.0, tau=0.25,
                  diff_ratio=True,
                  ):
    """multichannel FRA-RIR simulation for ad-hoc array,
    * mic_arch must be given, including
        - n_mic range
        - bounding box: within a room
        - mic spacing
    """
    assert mic_arch is not None
    if rt60 is None:
        rt60 = [0.1, 0.7]
    if direct_range is None:
        direct_range = [-6, 50]
    if image_coeff is None:
        image_coeff = ['T60', 2.0]
    if alpha is None:
        alpha = [0.25]

    if isinstance(mic_arch, dict):  # ADHOC ARRAY
        # sample mic_arch
        bounding_box = mic_arch["bounding_box"]
        mic_spacing = mic_arch["spacing"]
        n_mic = mic_arch["n_mic"]
        mic_arch = sample_mic_arch(n_mic, mic_spacing, bounding_box)
    else:   # FIXED ARRAY
        mic_arch = np.array(mic_arch)  # [nmic, 3]

    # sample room statistics
    if room_dim is None:
        room_dim = sample_room_dim()
    elif len(room_dim) == 2:
        room_dim = sample_room_dim(min_room=room_dim[0], max_room=room_dim[1])

    R = torch.tensor(
        1. / (2 * (1./room_dim[0]+1./room_dim[1] + 1./room_dim[2])))

    # sample RT60
    if len(rt60) == 2:
        T60 = torch.tensor(1.).uniform_(rt60[0], rt60[1])
    else:
        T60 = torch.tensor(rt60[0])

    # sample mic_pos
    if mic_pos is None:
        mic_arch = np.array(mic_arch)
        if array_pos is None:
            mic_pos, array_pos = sample_mic_array_pos(mic_arch, array_pos, room_dim=room_dim)
        else:
            mic_pos = array_pos + mic_arch
    else:
        if array_pos is None:
            array_pos = np.mean(mic_pos, 0)

    if src_pos is None:
        src_pos = sample_src_pos(room_dim=room_dim, array_pos=array_pos,
                                    num_src=num_src)
    else:
        assert len(src_pos) == num_src

    eps = np.finfo(np.float16).eps
    mic_position = torch.from_numpy(mic_pos)
    src_position = torch.from_numpy(np.array(src_pos))  # [nsource, 3]
    array_center = torch.from_numpy(array_pos).squeeze()  # [3]
    n_mic = mic_position.shape[0]

    # [nmic, nsource]
    direct_dist = torch.sum((mic_position.unsqueeze(
        1) - src_position.unsqueeze(0)) ** 2, -1).sqrt()
    # [nsource]
    direct_center = torch.sum(
        (array_center.unsqueeze(0) - src_position) ** 2, -1).sqrt()

    ns = n_mic * num_src

    # random set the #images
    if isinstance(image_coeff[0], float):
        random_image_coeff = torch.FloatTensor(
            1).uniform_(image_coeff[0], image_coeff[1])
    else:
        random_image_coeff = torch.FloatTensor(
            1).uniform_(T60 * 2, image_coeff[1])
    image = int(T60 * sr * random_image_coeff)

    ratio = 64
    sample_sr = sr*ratio
    velocity = 340.

    direct_idx = torch.ceil(direct_dist * sample_sr /
                            velocity).long().view(ns,)
    rir_length = int(np.ceil(sample_sr * T60))

    resample1 = Resample(sample_sr, sample_sr//int(np.sqrt(ratio)))
    resample2 = Resample(sample_sr//int(np.sqrt(ratio)), sr)

    reflect_coef = (1 - (1 - torch.exp(-0.16*R/T60)).pow(2)).sqrt()
    dist_range = [torch.linspace(1., velocity*T60/direct_center[i]-1, image)
                  for i in range(num_src)]

    if len(alpha) == 2:
        random_alpha = torch.FloatTensor(1).uniform_(alpha[0], alpha[1]).item()
    else:
        random_alpha = alpha[0]

    dist_prob = torch.linspace(random_alpha, 1., image).pow(2)
    dist_prob /= dist_prob.sum()
    dist_select_idx = dist_prob.multinomial(num_samples=int(image*num_src),
                                            replacement=True).view(num_src, image)

    dist_center_ratio = torch.stack(
        [dist_range[i][dist_select_idx[i]] for i in range(num_src)], 0)

    if not diff_ratio:
        # apply the same dist ratio to all microphones
        # nmic, nsource, nimage
        dist = direct_dist.unsqueeze(2) * dist_center_ratio.unsqueeze(0)
    else:
        # apply different dist ratios to mirophones
        azm = torch.FloatTensor(num_src, image).uniform_(-np.pi, np.pi)
        ele = torch.FloatTensor(num_src, image).uniform_(-np.pi/2, np.pi/2)
        # [nsource, nimage, 3]
        unit_3d = torch.stack([torch.sin(
            ele) * torch.cos(azm), torch.sin(ele) * torch.sin(azm), torch.cos(ele)], -1)
        # [nsource] x [nsource, T] x [nsource, nimage, 3] => [nsource, nimage, 3]
        image2center_dist = direct_center[...,
                                          None, None] * dist_center_ratio[..., None]
        image_position = array_center[None,
                                      None, ...] + image2center_dist * unit_3d
        # [nmic, nsource, nimage]
        dist = torch.sum((mic_position.unsqueeze(1).unsqueeze(
            1) - image_position[None, ...]) ** 2, -1) ** 0.5

    max_reflect, _ = torch.max(direct_dist, 0)  # [nsource]
    reflect_max = (torch.log10(velocity*T60) -
                   torch.log10(max_reflect) - 3) / torch.log10(reflect_coef + eps)
    reflect_ratio = (dist / (velocity*T60)).pow(2) *  \
        (reflect_max.view(-1, 1) - 1) + 1
    reflect_pertub = torch.FloatTensor(
        num_src, image).uniform_(a, b) * dist_center_ratio.pow(tau)
    reflect_ratio = torch.maximum(
        reflect_ratio + reflect_pertub.unsqueeze(0), torch.ones(1))

    # [nmic, nsrc, 1 + sr*2]
    dist = torch.cat([direct_dist.unsqueeze(2), dist], 2).view(ns, -1)
    reflect_ratio = torch.cat(
        [torch.zeros(n_mic, num_src, 1), reflect_ratio], 2).view(ns, -1)

    rir = torch.zeros(ns, rir_length)
    delta_idx = torch.minimum(torch.ceil(
        dist * sample_sr / velocity), torch.ones(1)*rir_length-1).long()
    delta_decay = reflect_coef.pow(reflect_ratio) / dist
    for i in range(ns):
        rir[i][delta_idx[i]] += delta_decay[i]

    direct_mask = torch.zeros(ns, rir_length).float()

    for i in range(ns):
        direct_mask[i, max(direct_idx[i]+sample_sr*direct_range[0]//1000, 0):
                    min(direct_idx[i]+sample_sr*direct_range[1]//1000, rir_length)] = 1.

    rir_direct = rir * direct_mask

    all_rir = torch.stack([rir, rir_direct], 1).view(ns*2, -1)
    rir_downsample = resample1(all_rir)
    rir_hp = highpass_biquad(
        rir_downsample, sample_sr // int(np.sqrt(ratio)), 80.)
    rir = resample2(rir_hp).float().view(n_mic, num_src, 2, -1)

    return rir[:, :, 0].data.numpy(), rir[:, :, 1].data.numpy()


if __name__ == "__main__":

    # === single-channel ===
    mic_arch = {
        'n_mic': [1, 1],
        'spacing': None,
        'bounding_box': None
    }
    rir, rir_direct = FRA_RIR_ADHOC(mic_arch)
    # n_mic, n_src, rir_len
    print(rir.shape, rir_direct.shape)

    # === multi-channel (fixed) ===
    mic_arch = [[-0.05, 0, 0], [0.05, 0, 0]]
    rir, rir_direct = FRA_RIR_ADHOC(mic_arch)
    # n_mic, n_src, rir_len
    print(rir.shape, rir_direct.shape)

    # === multi-channel (adhoc) ===
    mic_arch = {
        'n_mic': [1, 3],
        'spacing': [0.02, 0.05],
        'bounding_box': [0.5, 1.0, 0],  # x, y, z
    }
    rir, rir_direct = FRA_RIR_ADHOC(mic_arch)
    # n_mic, n_src, rir_len
    print(rir.shape, rir_direct.shape)
    