import torch
import numpy as np
import cv2

class MaskToBBSmoothed:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "masks": ("MASK",),
                "images": ("IMAGE",),
                "resolution": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
                "smoothing_factor": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0, "step": 0.01}),
                "padding": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
            },
        }

    RETURN_TYPES = ("MASK", "BBOX", "IMAGE")
    RETURN_NAMES = ("cropped_masks", "bboxes", "cropped_images")
    FUNCTION = "process"
    CATEGORY = "Image/Processing"

    def process(self, masks, images, resolution, smoothing_factor, padding):
        mask_frames = masks.cpu().numpy()
        img_frames = images.cpu().numpy()
        
        batch_size = min(mask_frames.shape[0], img_frames.shape[0])
        height, width = mask_frames.shape[1], mask_frames.shape[2]
        
        # --- PASS 1: Calculate Global Aspect Ratio ---
        all_mask_coords = []
        ratios = []
        for i in range(batch_size):
            m_bin = (mask_frames[i] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                pts = np.concatenate(contours)
                mx, my, mw, mh = cv2.boundingRect(pts)
                all_mask_coords.append(np.array([mx, my, mx + mw, my + mh], dtype=np.float32))
                ratios.append(mw / mh if mh > 0 else 1.0)
            else:
                all_mask_coords.append(np.array([width/2-32, height/2-32, width/2+32, height/2+32], dtype=np.float32))
                ratios.append(1.0)

        global_ratio = sum(ratios)/len(ratios)

        # --- PASS 2: Smooth and Crop ---
        output_m_crops, output_bboxes, output_crops = [], [], []
        cam_center, cam_size = None, None
        
        for i in range(batch_size):
            m_rect = all_mask_coords[i]
            m_center = np.array([(m_rect[0] + m_rect[2])/2, (m_rect[1] + m_rect[3])/2])
            m_dim = np.array([m_rect[2] - m_rect[0], m_rect[3] - m_rect[1]])
            
            target_size_raw = m_dim + (padding * 2)
            if global_ratio > (target_size_raw[0] / target_size_raw[1]):
                target_size = np.array([target_size_raw[1] * global_ratio, target_size_raw[1]])
            else:
                target_size = np.array([target_size_raw[0], target_size_raw[0] / global_ratio])

            if cam_center is None:
                cam_center, cam_size = m_center, target_size
            else:
                cam_center = cam_center * (1 - smoothing_factor) + m_center * smoothing_factor
                cam_size = cam_size * (1 - smoothing_factor*0.5) + target_size * (smoothing_factor*0.5)

            w_32, h_32 = ((int(cam_size[0]) + 31) // 32) * 32, ((int(cam_size[1]) + 31) // 32) * 32
            ix1, iy1 = int(max(0, cam_center[0] - w_32//2)), int(max(0, cam_center[1] - h_32//2))
            
            ix1 = min(ix1, width - w_32) if width > w_32 else 0
            iy1 = min(iy1, height - h_32) if height > h_32 else 0
            ix1, iy1 = max(0, ix1), max(0, iy1)

            output_bboxes.append((ix1, iy1, w_32, h_32))
            
            img_u8 = (img_frames[i] * 255).astype(np.uint8)
            crop = img_u8[iy1:iy1+h_32, ix1:ix1+w_32]
            
            if crop.shape[1] < w_32 or crop.shape[0] < h_32:
                canvas = np.zeros((h_32, w_32, 3), dtype=np.uint8)
                canvas[:crop.shape[0], :crop.shape[1]] = crop
                crop = canvas

            output_crops.append(cv2.resize(crop, (resolution, resolution), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0)
            
            m_u8 = (mask_frames[i] * 255).astype(np.uint8)
            m_crop = m_u8[iy1:iy1+h_32, ix1:ix1+w_32]
            if m_crop.shape[1] < w_32 or m_crop.shape[0] < h_32:
                m_canvas = np.zeros((h_32, w_32), dtype=np.uint8)
                m_canvas[:m_crop.shape[0], :m_crop.shape[1]] = m_crop
                m_crop = m_canvas
            output_m_crops.append(cv2.resize(m_crop, (resolution, resolution), interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.0)

        return (torch.from_numpy(np.stack(output_m_crops)), output_bboxes, torch.from_numpy(np.stack(output_crops)))

class MaskBBoxStitcher:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "processed_images": ("IMAGE",),
                "cropped_masks": ("MASK",),
                "bboxes": ("BBOX",),
                "feathering": ("INT", {"default": 10, "min": 0, "max": 128, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch"
    CATEGORY = "Image/Processing"

    def stitch(self, original_images, processed_images, cropped_masks, bboxes, feathering):
        orig = original_images.cpu().numpy()
        proc = processed_images.cpu().numpy()
        masks = cropped_masks.cpu().numpy()
        
        out_frames = []
        for i in range(min(len(orig), len(proc))):
            canvas = (orig[i] * 255).astype(np.uint8)
            x, y, w, h = bboxes[i]
            pi = (proc[i] * 255).astype(np.uint8)
            mi = (masks[i] * 255).astype(np.uint8)
            
            # --- SUB-PIXEL ALIGNMENT ---
            src_pts = np.float32([[0, 0], [pi.shape[1]-1, 0], [0, pi.shape[0]-1]])
            dst_pts = np.float32([[x, y], [x + w - 1, y], [x, y + h - 1]])
            matrix = cv2.getAffineTransform(src_pts, dst_pts)
            
            # Warp both image and mask back to full resolution space
            warped_proc = cv2.warpAffine(pi, matrix, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_CUBIC)
            warped_mask = cv2.warpAffine(mi, matrix, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_LINEAR)
            
            alpha = warped_mask.astype(np.float32) / 255.0
            
            if feathering > 0:
                f = feathering * 2 + 1
                alpha = cv2.GaussianBlur(alpha, (f, f), 0)
            
            alpha = np.expand_dims(alpha, axis=-1)
            blended = (warped_proc.astype(np.float32) * alpha + canvas.astype(np.float32) * (1.0 - alpha))
            out_frames.append(blended.astype(np.uint8).astype(np.float32) / 255.0)

        return (torch.from_numpy(np.stack(out_frames)),)

NODE_CLASS_MAPPINGS = {
    "MaskToBBSmoothed": MaskToBBSmoothed,
    "MaskBBoxStitcher": MaskBBoxStitcher
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskToBBSmoothed": "PC Video Mask Smooth Crop",
    "MaskBBoxStitcher": "PC Video Mask Stitcher"
}