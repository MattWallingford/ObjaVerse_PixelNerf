import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
import cv2
from util import get_image_to_tensor_balanced, get_mask_to_tensor


class ObjaVerseDataset(torch.utils.data.Dataset):
    """
    """

    def __init__(
        self,
        path,
        stage="train",
        list_prefix="softras_",
        image_size=512,
        max_imgs=100000,
        z_near=1.2,
        z_far=4.0,
    ):
        """
        :param path dataset root path, contains metadata.yml
        :param stage train | val | test
        :param list_prefix prefix for split lists: <list_prefix>[train, val, test].lst
        :param image_size result image size (resizes if different); None to keep original size
        :param sub_format shapenet | dtu dataset sub-type.
        :param scale_focal if true, assume focal length is specified for
        image of side length 2 instead of actual image size. This is used
        where image coordinates are placed in [-1, 1].
        """
        super().__init__()
        self.base_path = path
        assert os.path.exists(self.base_path)

        cats = [x for x in glob.glob(os.path.join(path, "*")) if os.path.isdir(x)]
        

        obj_folders = [os.path.join(path, x) for x in os.listdir(path) 
                        if os.path.isdir(os.path.join(path, x))]

        total_objs = len(obj_folders)
        self.total_objs = total_objs
        if stage == "train":
            file_lists = obj_folders[:int(total_objs*.9)]
        elif stage == "val":
            file_lists = file_lists = obj_folders[int(total_objs*.9):int(total_objs)]
        elif stage == "test":
            print(int(total_objs*.9))
            file_lists = obj_folders[int(total_objs*.9):]
        

        all_render_paths = []
        all_cam_paths = []
        for obj_path in file_lists:
            if not os.path.exists(obj_path):
                continue
            full_path = os.path.join(path,obj_path)
            render_paths = np.sort([os.path.join(full_path, x) for x in os.listdir(obj_path) if '.png' in x])
            cam_paths = np.sort([os.path.join(full_path, x) for x in os.listdir(obj_path) if '.npy' in x])
            all_cam_paths.append(cam_paths)
            all_render_paths.append(render_paths)
        self.all_cam_paths = all_cam_paths
        self.all_render_paths = all_render_paths
        self.stage = stage

        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()
        print(
            "Loading ObjaVerse dataset",
            self.base_path,
            "stage",
            stage,
            len(file_lists),
            "objs",
        )
        self.image_size = image_size
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

        self._coord_trans_cust = torch.diag(
            torch.tensor([1, -1, -1], dtype=torch.float32)
        )

        self.coord_trans_world = torch.tensor(
                [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32
            )

        self.z_near = .6
        self.z_far = 3
        self.lindisp = False

    def __len__(self):
        return len(self.all_cam_paths)

    def __getitem__(self, index):

        rgb_paths = self.all_render_paths[index]
        pose_paths = self.all_cam_paths[index]

        assert len(rgb_paths) == len(pose_paths)
        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        for rgb_path, pose_path in zip(rgb_paths, pose_paths):
            img = imageio.imread(rgb_path)[..., :3]
            img_tensor = self.image_to_tensor(img)
            alpha = imageio.imread(rgb_path)[..., 3]
            mask = (alpha != 0).astype(np.uint8) * 255
            # mask = (img != 0).all(axis=-1)[..., None].astype(np.uint8) * 255
            mask_tensor = self.mask_to_tensor(mask)
            # cv2.imwrite('../test.png', mask)
            # cv2.imwrite('../test_img.png', img)

            pose = torch.from_numpy(
                np.load(pose_path) 
            ).float()
            #Transform pose from BlenderToOpenCV
            pose = self._coord_trans_cust@pose
            R = pose[:3,:3]
            R_inv = R.T
            t = -R_inv@pose[:3,3]
            c_2_w = np.column_stack([R_inv,t])
            c_2_w = torch.tensor(np.vstack([c_2_w,[0,0,0,1]])).float()

            c_2_w = c_2_w@self._coord_trans


            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            if len(rnz) == 0:
                rmin, rmax = 0, 0
                cmin, cmax = 0, 0
            else:
                rmin, rmax = rnz[[0, -1]]
                cmin, cmax = cnz[[0, -1]]
            bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)

            all_imgs.append(img_tensor)
            all_poses.append(c_2_w)
            all_masks.append(mask_tensor)
            all_bboxes.append(bbox)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_masks = torch.stack(all_masks)
        all_bboxes = torch.stack(all_bboxes)

        focal = torch.tensor((560, 560), dtype=torch.float32)
        if all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size / all_imgs.shape[-2]
            focal *= scale
            all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")
        #print(all_bboxes)


        result = {
            #"path": dir_path,
            "img_id": index,
            "focal": focal,
            "images": all_imgs,
            # "bbox": all_bboxes,
            # "masks": all_masks,
            "poses": all_poses,
        }
        return result



