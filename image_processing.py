import cv2

import numpy as np

from sklearn.cluster import KMeans

from PIL import Image

class ImageProcessing(object):

    @staticmethod
    def crop_image(image, box, padding=0):
        '''
        Crop image
        '''
        top = box[0] - padding if box[0] - padding > 0 else 0
        left = box[1] - padding if box[1] - padding > 0 else 0
        bottom = box[2] + padding if box[2] + padding < image.shape[0] else image.shape[0]
        right = box[3] + padding if box[3] + padding < image.shape[1] else image.shape[1]
        return image[top:bottom, left:right]

    @staticmethod
    def crop(image, padding):
        '''
        Crop image
        '''
        image_h = image.shape[0]
        image_w = image.shape[1]

        return image[padding:image_h-padding, padding:image_w-padding]

    @staticmethod
    def adaptive_threshold(image, block_size=101):
        '''
        Adaptive threshold image
        '''
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 1)

    @staticmethod
    def segment_color_kmean(image, code=None, clusters=3):
        '''
        Segment color from input image by k-means clustering
        '''
        if code is not None:
            image = cv2.cvtColor(image, code)

        reshape_image = image.reshape((image.shape[0] * image.shape[1], 3))
        clt = KMeans(n_clusters=clusters)
        clt.fit(reshape_image)
        return clt.cluster_centers_

    # @staticmethod
    # def remove_color(self, color):
    #     pass

    @staticmethod
    def remove_color_from_image(image, channel, low, high, code=cv2.COLOR_BGR2RGB):
        '''
        Remove color from input image
        '''
        thresh = (low, high)
        image_conv = cv2.cvtColor(image, code)
        single_channel_image = image_conv[:, :, channel]
        binary = np.zeros_like(single_channel_image)
        binary[(single_channel_image > thresh[0]) & (single_channel_image <= thresh[1])] = 255
        result = np.zeros(image.shape, dtype=np.uint8)
        result[binary == 255] = high
        return result

    @classmethod
    def find_background(cls, image):
        try:
            cv2_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_image)
            return max(pil_im.getcolors(pil_im.size[0] * pil_im.size[1]))
        except:
            return (0, (0, 0, 0))

    @classmethod
    def get_color_background_around_stamp(cls, image, box):
        img_size = image.shape
        background_list = []

        top_area = cls.get_area_top(box, 50, img_size[0], img_size[1])
        xmin_temp, ymin_temp, xmax_temp, ymax_temp = top_area
        background_list.append(cls.find_background(image[ymin_temp:ymax_temp, xmin_temp: xmax_temp]))

        bot_area = cls.get_area_bot(box, 50, img_size[0], img_size[1])
        xmin_temp, ymin_temp, xmax_temp, ymax_temp = bot_area
        background_list.append(cls.find_background(image[ymin_temp:ymax_temp, xmin_temp: xmax_temp]))

        left_area = cls.get_area_left(box, 50, img_size[0], img_size[1])
        xmin_temp, ymin_temp, xmax_temp, ymax_temp = left_area
        background_list.append(cls.find_background(image[ymin_temp:ymax_temp, xmin_temp: xmax_temp]))

        righ_area = cls.get_area_righ(box, 50, img_size[0], img_size[1])
        xmin_temp, ymin_temp, xmax_temp, ymax_temp = righ_area
        background_list.append(cls.find_background(image[ymin_temp:ymax_temp, xmin_temp: xmax_temp]))

        for i in range(len(background_list)):
            color = background_list[i][1]
            for j in range(i + 1, len(background_list), 1):
                if color == background_list[j][1]:
                    lst = list(background_list[j])
                    lst[0] = lst[0] + background_list[i][0]
                    background_list[j] = tuple(lst)

        background_list.sort()

        color_background = (background_list[-1][-1][2], background_list[-1][-1][1], background_list[-1][-1][0])
        return color_background

    @classmethod
    def get_area_top(cls, box, padding, h, w):
        xmin, ymin, xmax, ymax = box
        ymax = ymin
        xmin = max(0, xmin - padding)
        ymin = max(0, ymin - padding)
        xmax = min(w, xmax + padding)
        return xmin, ymin, xmax, ymax

    @classmethod
    def get_area_bot(cls, box, padding, h, w):
        xmin, ymin, xmax, ymax = box
        ymin = ymax
        xmin = max(0, xmin - padding)
        xmax = min(w, xmax + padding)
        ymax = min(h, ymax + padding)
        return xmin, ymin, xmax, ymax

    @classmethod
    def get_area_left(cls, box, padding, h, w):
        xmin, ymin, xmax, ymax = box
        xmax = xmin
        xmin = max(0, xmin - padding)
        ymin = max(0, ymin - padding)
        ymax = min(h, ymax + padding)
        return xmin, ymin, xmax, ymax

    @classmethod
    def get_area_righ(cls, box, padding, h, w):
        xmin, ymin, xmax, ymax = box
        xmin = xmax
        ymin = max(0, ymin - padding)
        xmax = min(w, xmax + padding)
        ymax = min(h, ymax + padding)
        return xmin, ymin, xmax, ymax
