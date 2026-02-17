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

        all_mask_coords = []
        frame_ratios = []
        
        # --- PASS 1: Identify Largest Contour Only ---
        for i in range(batch_size):
            m_bin = (mask_frames[i] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # FIX: Pick only the largest contour by area to ignore stray noise pixels
                main_contour = max(contours, key=cv2.contourArea)
                # Ignore tiny specks (less than 10 pixels) entirely
                if cv2.contourArea(main_contour) > 10:
                    mx, my, mw, mh = cv2.boundingRect(main_contour)
                    all_mask_coords.append(np.array([mx, my, mx + mw, my + mh], dtype=np.float32))
                    frame_ratios.append((mw + padding*2) / (mh + padding*2))
                    continue
            
            # Fallback if no valid mask found
            all_mask_coords.append(np.array([width/2-32, height/2-32, width/2+32, height/2+32], dtype=np.float32))
            frame_ratios.append(1.0)

        global_target_ratio = sum(frame_ratios) / len(frame_ratios)

        output_m_crops, output_bboxes, output_crops = [], [], []
        cam_center, cam_size = None, None
        
        for i in range(batch_size):
            m_rect = all_mask_coords[i]
            m_center = np.array([(m_rect[0] + m_rect[2])/2, (m_rect[1] + m_rect[3])/2])
            
            mask_w, mask_h = (m_rect[2] - m_rect[0]), (m_rect[3] - m_rect[1])
            raw_w, raw_h = mask_w + (padding * 2), mask_h + (padding * 2)
            
            if raw_w / raw_h > global_target_ratio:
                target_w, target_h = raw_w, raw_w / global_target_ratio
            else:
                target_h, target_w = raw_h, raw_h * global_target_ratio
            
            target_size = np.array([target_w, target_h])

            if cam_center is None:
                cam_center, cam_size = m_center, target_size
            else:
                # Smooth the float coordinates
                cam_center = cam_center * (1 - smoothing_factor) + m_center * smoothing_factor
                cam_size = cam_size * (1 - smoothing_factor * 0.2) + target_size * (smoothing_factor * 0.2)

            hw, hh = cam_size[0]/2, cam_size[1]/2
            x1, y1, x2, y2 = cam_center[0]-hw, cam_center[1]-hh, cam_center[0]+hw, cam_center[1]+hh
            
            # Sub-pixel nudge for mask containment
            if x1 > m_rect[0]: x1 = m_rect[0] - 2.0
            if y1 > m_rect[1]: y1 = m_rect[1] - 2.0
            if x2 < m_rect[2]: x2 = m_rect[2] + 2.0
            if y2 < m_rect[3]: y2 = m_rect[3] + 2.0
            
            f_w, f_h = x2 - x1, (x2 - x1) / global_target_ratio
            f_x1, f_y1 = (x1 + x2)/2 - f_w/2, (y1 + y2)/2 - f_h/2

            output_bboxes.append((f_x1, f_y1, f_w, f_h))
            
            # AI resolution calculation
            if global_target_ratio > 1:
                tw, th = resolution, int(resolution / global_target_ratio)
            else:
                th, tw = resolution, int(resolution * global_target_ratio)
            tw, th = (tw // 32) * 32, (th // 32) * 32 

            src_pts = np.float32([[f_x1, f_y1], [f_x1 + f_w, f_y1], [f_x1, f_y1 + f_h], [f_x1 + f_w, f_y1 + f_h]])
            dst_pts = np.float32([[0, 0], [tw, 0], [0, th], [tw, th]])
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            img_u8 = (img_frames[i] * 255).astype(np.uint8)
            crop = cv2.warpPerspective(img_u8, M, (tw, th), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
            output_crops.append(crop.astype(np.float32)/255.0)
            
            m_u8 = (mask_frames[i] * 255).astype(np.uint8)
            m_crop = cv2.warpPerspective(m_u8, M, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            output_m_crops.append(m_crop.astype(np.float32)/255.0)

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
            fx, fy, fw, fh = bboxes[i]
            pi = (proc[i] * 255).astype(np.uint8)
            mi = (masks[i] * 255).astype(np.uint8)
            
            h, w = canvas.shape[:2]
            src_pts = np.float32([[0, 0], [pi.shape[1], 0], [0, pi.shape[0]], [pi.shape[1], pi.shape[0]]])
            dst_pts = np.float32([[fx, fy], [fx + fw, fy], [fx, fy + fh], [fx + fw, fy + fh]])
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            ai_warped = cv2.warpPerspective(pi, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            mask_warped = cv2.warpPerspective(mi, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            safe_mask = np.zeros((h, w), dtype=np.uint8)
            # Inset by 1.5 pixels to ensure sub-pixel edge artifacts are killed
            ix, iy, iw, ih = int(fx+1), int(fy+1), int(fw)-2, int(fh)-2
            cv2.rectangle(safe_mask, (ix, iy), (ix + iw, iy + ih), 255, -1)
            
            alpha = (mask_warped.astype(np.float32) / 255.0) * (safe_mask.astype(np.float32) / 255.0)
            
            if feathering > 0:
                f = feathering * 2 + 1
                alpha = cv2.GaussianBlur(alpha, (f, f), 0)
            
            alpha = np.expand_dims(alpha, axis=-1)
            blended = (ai_warped.astype(np.float32) * alpha + canvas.astype(np.float32) * (1.0 - alpha))
            out_frames.append(blended.astype(np.uint8).astype(np.float32) / 255.0)

        return (torch.from_numpy(np.stack(out_frames)),)

NODE_CLASS_MAPPINGS = {"MaskToBBSmoothed": MaskToBBSmoothed, "MaskBBoxStitcher": MaskBBoxStitcher}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskToBBSmoothed": "PC Video Mask Smooth Crop",
    "MaskBBoxStitcher": "PC Video Mask Stitcher"
}