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
                "smoothing_factor": ("FLOAT", {"default": 0.03, "min": 0.01, "max": 1.0, "step": 0.01}),
                "padding": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
            },
        }

    RETURN_TYPES = ("MASK", "BBOX", "IMAGE")
    RETURN_NAMES = ("cropped_masks", "bboxes", "cropped_images")
    FUNCTION = "process"
    CATEGORY = "Image/Processing"

    def process(self, masks, images, resolution, smoothing_factor, padding):
        # MASK tensor is [B, H, W], IMAGE tensor is [B, H, W, C]
        mask_frames = masks.cpu().numpy()
        img_frames = images.cpu().numpy()
        
        batch_size = min(mask_frames.shape[0], img_frames.shape[0])
        height, width = mask_frames.shape[1], mask_frames.shape[2]
        
        output_m_crops = []
        output_bboxes = []
        output_crops = []
        
        prev_bbox = None
        
        for i in range(batch_size):
            # 1. Prepare Mask and Detect Contours
            mask_binary = (mask_frames[i] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                x_min, y_min = width, height
                x_max, y_max = 0, 0
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x + w), max(y_max, y + h)
            else:
                x_min, y_min = width // 2 - 32, height // 2 - 32
                x_max, y_max = width // 2 + 32, height // 2 + 32

            # 2. Apply Padding
            x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
            x_max, y_max = min(width, x_max + padding), min(height, y_max + padding)

            # 3. Calculate Square BBox (32px aligned)
            curr_w, curr_h = x_max - x_min, y_max - y_min
            side = max(curr_w, curr_h)
            new_side = ((side + 31) // 32) * 32
            
            center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
            s_x_min, s_y_min = center_x - new_side // 2, center_y - new_side // 2
            s_x_max, s_y_max = s_x_min + new_side, s_y_min + new_side
            
            # Shift back into frame if square exceeds edges
            if s_x_min < 0:
                s_x_max -= s_x_min
                s_x_min = 0
            if s_y_min < 0:
                s_y_max -= s_y_min
                s_y_min = 0
            if s_x_max > width:
                s_x_min -= (s_x_max - width)
                s_x_max = width
            if s_y_max > height:
                s_y_min -= (s_y_max - height)
                s_y_max = height

            s_x_min, s_y_min = max(0, s_x_min), max(0, s_y_min)
            s_x_max, s_y_max = min(width, s_x_max), min(height, s_y_max)

            # 4. Smoothing
            curr_coords = np.array([s_x_min, s_y_min, s_x_max, s_y_max])
            if prev_bbox is None:
                smoothed_bbox = curr_coords
            else:
                smoothed_bbox = prev_bbox * (1 - smoothing_factor) + curr_coords * smoothing_factor
            
            prev_bbox = smoothed_bbox
            sb = smoothed_bbox.astype(int)
            
            # 5. BBOX Data (x, y, w, h)
            output_bboxes.append((sb[0], sb[1], sb[2] - sb[0], sb[3] - sb[1]))
            
            # 6. Coordinate clipping for cropping
            y1, y2, x1, x2 = max(0, sb[1]), min(height, sb[3]), max(0, sb[0]), min(width, sb[2])

            # 7. Square Crop Output (Images)
            src_frame = (img_frames[i] * 255).astype(np.uint8)
            crop = src_frame[y1:y2, x1:x2]
            if crop.size > 0:
                resized_crop = cv2.resize(crop, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
            else:
                resized_crop = np.zeros((resolution, resolution, 3), dtype=np.uint8)
            output_crops.append(resized_crop.astype(np.float32) / 255.0)

            # 8. Square Crop Output (Masks)
            m_crop = mask_binary[y1:y2, x1:x2]
            if m_crop.size > 0:
                resized_m_crop = cv2.resize(m_crop, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
            else:
                resized_m_crop = np.zeros((resolution, resolution), dtype=np.uint8)
            output_m_crops.append(resized_m_crop.astype(np.float32) / 255.0)

        return (
            torch.from_numpy(np.stack(output_m_crops)), 
            output_bboxes, 
            torch.from_numpy(np.stack(output_crops))
        )

class MaskBBoxStitcher:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "processed_images": ("IMAGE",), # Renamed for clarity
                "bboxes": ("BBOX",),
                "feathering": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch"
    CATEGORY = "Image/Processing"

    def stitch(self, original_images, processed_images, bboxes, feathering):
        orig_frames = original_images.cpu().numpy()
        proc_frames = processed_images.cpu().numpy()
        
        batch_size = min(orig_frames.shape[0], proc_frames.shape[0], len(bboxes))
        out_frames = []

        for i in range(batch_size):
            orig_img = (orig_frames[i] * 255).astype(np.uint8)
            proc_img = (proc_frames[i] * 255).astype(np.uint8)
            x, y, w, h = bboxes[i]

            if w <= 0 or h <= 0:
                out_frames.append(orig_frames[i])
                continue

            # Resize processed crop back to original bbox size
            resized_crop = cv2.resize(proc_img, (w, h), interpolation=cv2.INTER_LANCZOS4)

            if feathering > 0:
                mask = np.ones((h, w), dtype=np.uint8) * 255
                feather_size = max(3, feathering if feathering % 2 != 0 else feathering + 1)
                mask = cv2.copyMakeBorder(mask, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
                mask = cv2.GaussianBlur(mask, (feather_size, feather_size), 0)
                mask = mask[10:-10, 10:-10][:, :, np.newaxis] / 255.0

                roi = orig_img[y:y+h, x:x+w]
                blended_roi = (resized_crop * mask + roi * (1 - mask)).astype(np.uint8)
                orig_img[y:y+h, x:x+w] = blended_roi
            else:
                orig_img[y:y+h, x:x+w] = resized_crop

            out_frames.append(orig_img.astype(np.float32) / 255.0)

        return (torch.from_numpy(np.stack(out_frames)),)

NODE_CLASS_MAPPINGS = {
    "MaskToBBSmoothed": MaskToBBSmoothed,
    "MaskBBoxStitcher": MaskBBoxStitcher
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskToBBSmoothed": "PC Video Mask Smooth Crop",
    "MaskBBoxStitcher": "PC Video Mask Stitcher"
}