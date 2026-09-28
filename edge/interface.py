import cv2
import numpy as np
import os
import time
import config

def render(frame, last_class, last_conf, ui_detected_class, show_ui_until, text_color):
    height, width, _ = frame.shape

    if config.DISPLAY_MODE == "dev":
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Class: {last_class.upper()}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
        if last_conf > 0:
            cv2.putText(display_frame, f"Conf: {last_conf:.1%}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        cv2.putText(display_frame, f"Current Threshold: {config.CONFIDENCE_THRESHOLD:.1%}", (20, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(display_frame, "[SPACE] to Classify | [ESC] to Exit", (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        return display_frame

    else:
        display_frame = np.zeros((height, width, 3), dtype=np.uint8)

        if time.time() < show_ui_until:
            bg_color = config.RECYCLING_COLORS.get(ui_detected_class, (50, 50, 50))
            display_frame[:] = bg_color

            msg = "NAO RECONHECIDO. Tente novamente." if ui_detected_class == "unsure" else f"DETECTADO: {ui_detected_class.upper()}"
            text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)[0]
            text_x = (width - text_size[0]) // 2
            cv2.putText(display_frame, msg, (text_x, 150), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)

            asset_path_png = os.path.join(config.ASSETS_DIR, f"{ui_detected_class}.png")
            asset_path_jpg = os.path.join(config.ASSETS_DIR, f"{ui_detected_class}.jpg")
            asset_img_path = asset_path_png if os.path.exists(asset_path_png) else asset_path_jpg

            if os.path.exists(asset_img_path):
                img_asset = cv2.imread(asset_img_path, cv2.IMREAD_UNCHANGED)
                if img_asset is not None:
                    img_asset = cv2.resize(img_asset, (400, 400))
                    start_y = (height - 400) // 2 + 50
                    start_x = (width - 400) // 2
                    roi = display_frame[start_y:start_y+400, start_x:start_x+400]

                    if len(img_asset.shape) == 3 and img_asset.shape[2] == 4:
                        alpha = img_asset[:, :, 3]
                        mask_inv = cv2.bitwise_not(alpha)
                        white_icon = np.full((400, 400, 3), 255, dtype=np.uint8)
                        roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                        icon_fg = cv2.bitwise_and(white_icon, white_icon, mask=alpha)
                        display_frame[start_y:start_y+400, start_x:start_x+400] = cv2.add(roi_bg, icon_fg)
                    else:
                        img_gray = cv2.cvtColor(img_asset, cv2.COLOR_BGR2GRAY)
                        _, mask = cv2.threshold(img_gray, 15, 255, cv2.THRESH_BINARY)
                        mask_inv = cv2.bitwise_not(mask)
                        white_icon = np.full((400, 400, 3), 255, dtype=np.uint8)
                        roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                        icon_fg = cv2.bitwise_and(white_icon, white_icon, mask=mask)
                        display_frame[start_y:start_y+400, start_x:start_x+400] = cv2.add(roi_bg, icon_fg)
        else:
            display_frame[:] = (0, 0, 0)
            logo_path = os.path.join(config.ASSETS_DIR, "logo.jpg")

            if os.path.exists(logo_path):
                bg_img = cv2.imread(logo_path)
                if bg_img is not None:
                    scale = width / bg_img.shape[1]
                    new_h = int(bg_img.shape[0] * scale)
                    if new_h >= height:
                        resized_bg = cv2.resize(bg_img, (width, new_h))
                        start_y = (new_h - height) // 2
                        display_frame[:] = resized_bg[start_y:start_y+height, :]
                    else:
                        scale = height / bg_img.shape[0]
                        new_w = int(bg_img.shape[1] * scale)
                        resized_bg = cv2.resize(bg_img, (new_w, height))
                        start_x = (new_w - width) // 2
                        display_frame[:] = resized_bg[:, start_x:start_x+width]
            else:
                cv2.putText(display_frame, "ECOSORT AI", ((width - 300) // 2, height // 2 - 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 4)

        return display_frame