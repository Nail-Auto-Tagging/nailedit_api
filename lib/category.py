import os
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.models.resnet as resnet

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


device = "cuda" if torch.cuda.is_available() else "cpu"
category_num = 13
conv1x1 = resnet.conv1x1
Bottleneck = resnet.Bottleneck
BasicBlock = resnet.BasicBlock


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, zero_init_residual=True):
        super(ResNet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)  # 3 반복
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 4 반복
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 6 반복
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 3 반복

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):  # planes -> 입력되는 채널 수
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def load_model(path):
    model = ResNet(resnet.Bottleneck, [3, 4, 6, 3], category_num).to(device)
    model = torch.load(path, map_location=torch.device("cpu"))
    model.eval()
    return model


classification_model = load_model("model/resnet50_categories_v1_20.pt")


def classify(input):
    Sig = torch.nn.Sigmoid()
    input = torch.stack(input, dim=0)
    with torch.no_grad():
        input = input.to(device)
        with torch.autocast("cuda"):
            output_regular = Sig(classification_model(input).float()).cpu()

    output_regular = np.array(output_regular)
    return output_regular


def find_hand_old(frame):
    img = frame.copy()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    YCrCb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    YCrCb_frame = cv2.GaussianBlur(YCrCb_frame, (3, 3), 0)
    mask = cv2.inRange(YCrCb_frame, np.array([0, 127, 75]), np.array([255, 177, 130]))
    bin_mask = mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_mask = cv2.dilate(bin_mask, kernel, iterations=5)
    res = cv2.bitwise_and(frame, frame, mask=bin_mask)

    return img, bin_mask, res


def crop(image):
    results = []
    model = tf.Graph()

    with model.as_default():
        graphDef = tf.compat.v1.GraphDef()

        with tf.compat.v2.io.gfile.GFile("model/ssd_object_detection.pb", "rb") as f:
            serializedGraph = f.read()
            graphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(graphDef, name="")

    with model.as_default():
        with tf.compat.v1.Session(graph=model) as sess:
            imageTensor = model.get_tensor_by_name("image_tensor:0")
            boxesTensor = model.get_tensor_by_name("detection_boxes:0")

            # for each bounding box we would like to know the score
            # (i.e., probability) and class label
            scoresTensor = model.get_tensor_by_name("detection_scores:0")
            classesTensor = model.get_tensor_by_name("detection_classes:0")
            numDetections = model.get_tensor_by_name("num_detections:0")

            (H, W) = image.shape[:2]
            output = image.copy()
            img_ff, bin_mask, res = find_hand_old(image.copy())
            image = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)

            (boxes, scores, labels, N) = sess.run(
                [boxesTensor, scoresTensor, classesTensor, numDetections],
                feed_dict={imageTensor: image},
            )
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            labels = np.squeeze(labels)
            for (box, score, label) in zip(boxes, scores, labels):
                if score < 0.8:
                    continue

                # scale the bounding box from the range [0, 1] to [W, H]
                (startY, startX, endY, endX) = box
                startX = int(startX * W)
                startY = int(startY * H)
                endX = int(endX * W)
                endY = int(endY * H)
                X_mid = startX + int(abs(endX - startX) / 2)
                Y_mid = startY + int(abs(endY - startY) / 2)
                cropped_image = output[startY:endY, startX:endX]

                resized = cv2.resize(
                    cropped_image, dsize=(224, 224), interpolation=cv2.INTER_LINEAR
                )
                tmp = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                adjusted = np.zeros_like(resized)
                adjusted[:, :, 0] = tmp
                adjusted[:, :, 1] = tmp
                adjusted[:, :, 2] = tmp
                normalized = adjusted / 255
                normalized = (normalized - 0.5) * 2
                resized = np.transpose(normalized, (2, 0, 1))
                cur_res = torch.from_numpy(resized).float()

                results.append(
                    {
                        "bounding_box": {
                            "start_x": startX,
                            "start_y": startY,
                            "end_x": endX,
                            "end_y": endY,
                        },
                        "pil_image": Image.fromarray(cropped_image),
                        "opencv_image": cur_res,
                    }
                )
            return results
