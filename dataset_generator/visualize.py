import cv2
import os
import json

SPLIT = 'train'

if __name__ == '__main__':
    train_labels = os.path.join(os.path.dirname(__file__), f'greenhouse_{SPLIT}', 'labels.json')

    with open(train_labels, 'r') as f:
        for line in f:
            json_line = json.loads(line)
            file_path = json_line['raw_file']
            print(file_path)

            image = cv2.imread(os.path.join(os.path.dirname(__file__), f'greenhouse_{SPLIT}', file_path))
            ys = list(range(0, 720, 10))

            h_samples = json_line['h_samples']
            lanes = json_line['lanes']

            for lane in lanes:
                lane_points = []
                for i, x in enumerate(lane):
                    if x == -2:
                        continue
                    lane_points.append((x, ys[i]))
                
                for i, point in enumerate(lane_points):
                    if i > 0:
                        cv2.line(image, lane_points[i-1], point, (0, 255, 0), 2)
                    else:
                        cv2.circle(image, point, 2, (0, 255, 0), -1)

            cv2.imshow('image', image)
            cv2.waitKey(0)

    cv2.destroyAllWindows()