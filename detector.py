import os
import cv2
import torch
import numpy as np

# Ensure torchvision model_urls compatibility for CRAFT (removed in torchvision 0.15+)
try:
    import torchvision.models.vgg
    if not hasattr(torchvision.models.vgg, 'model_urls'):
        torchvision.models.vgg.model_urls = {
            'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        }
except Exception:
    pass

from craft_text_detector import Craft


class CraftDetector:
    """
    CRAFT text detector with local pretrained model weights.
    """

    def __init__(
        self,
        use_cuda=None,
        enable_refiner=True,
        text_threshold=0.4,
        link_threshold=0.4,
        low_text=0.25,
    ):
        self.cuda = (
            torch.cuda.is_available()
            if use_cuda is None
            else use_cuda
        )

        self.enable_refiner = enable_refiner
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.craft = None

        self.initialize_model()

    def initialize_model(self):
        """Load CRAFT and optional refiner weights from local files."""

        try:
            print(
                f"Loading CRAFT model "
                f"(CUDA={self.cuda}, Refiner={self.enable_refiner})..."
            )

            model_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "models",
            )

            craft_weights = os.path.join(
                model_dir, "craft_mlt_25k.pth"
            )
            refiner_weights = os.path.join(
                model_dir, "craft_refiner_CTW1500.pth"
            )

            if not os.path.isfile(craft_weights):
                raise FileNotFoundError(
                    f"CRAFT weights not found: {craft_weights}"
                )

            if (
                self.enable_refiner
                and not os.path.isfile(refiner_weights)
            ):
                raise FileNotFoundError(
                    f"CRAFT refiner weights not found: "
                    f"{refiner_weights}"
                )

            self.craft = Craft(
                output_dir=None,
                crop_type="box",
                cuda=self.cuda,
                refiner=self.enable_refiner,
                text_threshold=self.text_threshold,
                link_threshold=self.link_threshold,
                low_text=self.low_text,
                weight_path_craft_net=craft_weights,
                weight_path_refine_net=(
                    refiner_weights
                    if self.enable_refiner
                    else None
                ),
            )

            print("CRAFT models loaded successfully!")

        except Exception as exc:
            print(f"Error loading CRAFT models: {exc}")
            raise

    def detect_text(self, image):
        """Return CRAFT text bounding boxes from a BGR image."""

        if self.craft is None:
            raise RuntimeError("CRAFT detector is not initialized.")

        if image is None or image.size == 0:
            raise ValueError("Input image is invalid.")

        try:
            prediction = self.craft.detect_text(image)
            return prediction.get("boxes", [])

        except Exception as exc:
            print(f"Error during CRAFT text detection: {exc}")
            return []

    def unload(self):
        """Unload CRAFT models from memory."""

        if self.craft is not None:
            try:
                self.craft.unload_craftnet_model()

                if self.enable_refiner:
                    self.craft.unload_refinenet_model()

                print("CRAFT models unloaded from memory.")

            except Exception as exc:
                print(f"Error unloading CRAFT models: {exc}")


class EastDetector:
    """
    OpenCV DNN EAST text detector.
    """

    def __init__(
        self,
        model_path="frozen_east_text_detection.pb",
        conf_threshold=0.5,
        nms_threshold=0.4,
        net_width=640,
        net_height=640,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.net_width = net_width
        self.net_height = net_height
        self.net = None

        self.initialize_model()

    def initialize_model(self):
        """Load the EAST model."""

        if not os.path.exists(self.model_path):
            print(
                f"ERROR: EAST model file not found: "
                f"{self.model_path}"
            )
            return

        try:
            print(f"Loading EAST model from '{self.model_path}'...")
            self.net = cv2.dnn.readNet(self.model_path)
            print("EAST text detector loaded successfully!")

        except Exception as exc:
            print(f"Error loading EAST model: {exc}")
            self.net = None

    def decode_predictions(self, scores, geometry):
        """Decode EAST output maps into rectangles and confidence scores."""

        num_rows, num_cols = scores.shape[2:4]
        rects = []
        confidences = []

        for y in range(num_rows):
            scores_data = scores[0, 0, y]
            x_data_0 = geometry[0, 0, y]
            x_data_1 = geometry[0, 1, y]
            x_data_2 = geometry[0, 2, y]
            x_data_3 = geometry[0, 3, y]
            angles_data = geometry[0, 4, y]

            for x in range(num_cols):
                score = scores_data[x]

                if score < self.conf_threshold:
                    continue

                offset_x, offset_y = x * 4.0, y * 4.0
                angle = angles_data[x]
                cos_angle = np.cos(angle)
                sin_angle = np.sin(angle)

                h = x_data_0[x] + x_data_2[x]
                w = x_data_1[x] + x_data_3[x]

                end_x = int(
                    offset_x
                    + cos_angle * x_data_1[x]
                    + sin_angle * x_data_2[x]
                )
                end_y = int(
                    offset_y
                    - sin_angle * x_data_1[x]
                    + cos_angle * x_data_2[x]
                )

                start_x = int(end_x - w)
                start_y = int(end_y - h)

                rects.append((start_x, start_y, end_x, end_y))
                confidences.append(float(score))

        return rects, confidences

    def detect_text(self, image):
        """Return EAST text boxes mapped to the original image size."""

        if self.net is None:
            print("EAST detector is not initialized.")
            return []

        if image is None or image.size == 0:
            return []

        orig_h, orig_w = image.shape[:2]

        ratio_w = orig_w / float(self.net_width)
        ratio_h = orig_h / float(self.net_height)

        resized_img = cv2.resize(
            image, (self.net_width, self.net_height)
        )

        layer_names = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3",
        ]

        blob = cv2.dnn.blobFromImage(
            resized_img,
            1.0,
            (self.net_width, self.net_height),
            (123.68, 116.78, 103.94),
            swapRB=True,
            crop=False,
        )

        self.net.setInput(blob)
        scores, geometry = self.net.forward(layer_names)

        rects, confidences = self.decode_predictions(
            scores, geometry
        )

        if not rects:
            return []

        indices = cv2.dnn.NMSBoxes(
            rects,
            confidences,
            self.conf_threshold,
            self.nms_threshold,
        )

        boxes = []

        if len(indices) > 0:
            flat_indices = np.asarray(indices).flatten()

            for i in flat_indices:
                start_x, start_y, end_x, end_y = rects[int(i)]

                sx = max(0, int(start_x * ratio_w))
                sy = max(0, int(start_y * ratio_h))
                ex = min(orig_w, int(end_x * ratio_w))
                ey = min(orig_h, int(end_y * ratio_h))

                if ex <= sx or ey <= sy:
                    continue

                boxes.append([
                    [sx, sy],
                    [ex, sy],
                    [ex, ey],
                    [sx, ey],
                ])

        return boxes

