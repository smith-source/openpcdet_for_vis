import argparse
import glob
import os
from pathlib import Path

import numpy

# import mayavi.mlab as mlab
try:
    import open3d
    from visual_utils import my_open3d_vis_utils_1 as V                # #
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils, object3d_kitti,calibration_kitti                         # #g
# from visual_utils import visualize_utils as V   # #


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin', label_path=None, calib_path=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path


        if label_path is not None:
            self.label_path = Path(label_path)                       # #g
            self.calib_path = Path(calib_path)

        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        # label = DemoDataset.get_label(self, index)                     # #g

        input_dict = {
            'points': points,
            'frame_id': index,
            # 'gt_boxes': label                                          # #g

        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


    def get_calib(self, gt_name):
        calib_file = self.calib_path / gt_name
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_label(self, gt_name):                                                                                       # #g


        # label_file = self.label_path / ('%s.txt' % index)
        label_file = self.label_path / gt_name

        assert label_file.exists()
        # object_list = object3d_kitti.get_objects_from_label(label_file)
        obj_list = object3d_kitti.get_objects_from_label(label_file)
        # for i in object_list:
        #     label_list.append(i.to_openpcdet_format())

        annotations = {}
        calib = self.get_calib(gt_name)

        annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format       # # ?
        annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
        annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
        num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
        loc = annotations['location'][:num_objects]
        dims = annotations['dimensions'][:num_objects]
        rots = annotations['rotation_y'][:num_objects]
        loc_lidar = calib.rect_to_lidar(loc)                                                   # # 转化为OpenPCDet坐标
        l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
        loc_lidar[:, 2] += h[:, 0] / 2
        gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
        annotations['gt_boxes_lidar'] = gt_boxes_lidar
        return annotations['gt_boxes_lidar']


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    parser.add_argument('--gt_boxes', type=str, default=None, help='the gt_boxes path')                                 # #g
    parser.add_argument('--calib_path', type=str, default=None, help='the calib txt files path')                        # #g
    parser.add_argument('--preds_path', type=str, default=None, help='where to store the predictions results')  # #p

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    label_path = args.gt_boxes                                                                                          # #g
    calib_path = args.calib_path                                                                                        # #g
    preds_path = args.preds_path                                                                                        # #p

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger, label_path=label_path, calib_path=calib_path       # #g
    )

    gt_index_list = demo_dataset.sample_file_list                                                                       # #g

    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    # if args.gt_boxes is not None:
    #     data_path = args.data_path
    #     gt_boxes_path = args.gt_boxes
    #     gt_list = []
    #     kitti = KittiDataset()
    #     for i in os.listdir(data_path):
    #         file_name = os.path.splittext(i)[0]
    #         KittiDataset.get_label(file_name)
    #         gt_path = os.path.join(gt_boxes_path, file_name)
    #         gt_list.append(gt_path)



    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            label_name = gt_index_list[idx].split('/')[-1].replace('bin', 'txt')                                        # #g

    # *** the following code is to convert the lidar_preds to rect_preds, and store it. *** #
            label_dict = {1:'Car', 2:'Pedestrian', 3:'Cyclist'}
            if preds_path is not None:
                pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()
                preds_gt_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
                scores = pred_dicts[0]['pred_scores'].cpu().numpy()

                preds_file_path = os.path.join(preds_path, label_name)
                with open(preds_file_path, 'w') as f:
                    for b in range(scores.shape[0]):
                        pred_label = label_dict[pred_labels[b]]
                        [x, y, z, l, w, h, ry] = preds_gt_boxes[b]
                        cache_x = -y
                        score = scores[b]
                        igonre = '0 0 0 0 0 0 0'

                        image_loc = [x, z, z-h/2]
                        image_loc = torch.tensor(image_loc)
                        image_loc = image_loc.reshape(1,3)
                        calib = demo_dataset.get_calib(gt_name=label_name)
                        new_loc = calib.lidar_to_rect(image_loc)
                        # new_loc = calib.rect_to_img(new_loc)
                        [_, y, z] = new_loc[0]
                        x = cache_x

                        line = ' '.join([str(i) for i in [pred_label, igonre, h, w, l, x, y, z, - ry-np.pi/2, score]])
                        f.write(line)
                        f.write('\n')
    # ***    end                                                                       ***   #


            if label_path is not None:                                                                                  # #g
                gt_boxes = demo_dataset.get_label(label_name)
                gt_boxes = torch.Tensor(gt_boxes)
            else:
                gt_boxes = None

            V.draw_scenes(
                points=data_dict['points'][:, 1:], gt_boxes=gt_boxes, ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            # mlab.show(stop=True)   # #

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
