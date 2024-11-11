import cv2
import os
import numpy as np
import json
from scipy.interpolate import InterpolatedUnivariateSpline

H_SAMPLES_STEP = 10
H_SAMPLES = range(0, 720, 10)

def handle_mouse_event(event, x, y, flags, param):
    global points, image, line, colors, lines_equations
    if event == cv2.EVENT_LBUTTONDOWN:
        x, y = round(x), round(y)
        points[line].append((x, y))
        points[line] = sorted(points[line], key=lambda x: x[1])
        x, y = zip(*points[line])
        if len(points[line]) >= 2:
            if len(lines_equations) < line+1:
                lines_equations.append(0)
            lines_equations[line] = InterpolatedUnivariateSpline(y, x, k=min(len(points[line])-1, 5))

if __name__ == '__main__':
    images_path = os.path.join(os.path.dirname(__file__))
    images_path = [os.path.join(images_path, file) for file in os.listdir(images_path)]
    images_path = [file for file in images_path if os.path.isfile(file)]

    # Create a window for drawing points and lines
    cv2.namedWindow('window', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('window', handle_mouse_event)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255)]
    for image_path in images_path:
        points = [[]]
        lines_equations = []
        line = 0

        while True:
            org_image = cv2.imread(image_path)
            image = org_image.copy()
            img_h, img_w = image.shape[:2]
            cv2.putText(image, f"Line {line+1}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[line], 4)

            for i, point in enumerate(points):
                for j in range(1, len(point)):
                    cv2.line(image, point[j-1], point[j], colors[i], 2)
                for x, y in point:
                    cv2.circle(image, (x, y), 2, colors[i], -1)

            cv2.imshow('window', image)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('l'):
                line += 1
                line %= len(colors)
                points.append([])
            elif key == ord('c'):
                points[line] = []
            elif key == ord('n'):
                entry = {'lanes': [],
                         'h_samples': list(H_SAMPLES),
                         'raw_file': '',}
                
                for i, line_equation in enumerate(lines_equations):
                    if len(points[i]) == 0:
                        continue
                    line_xs = [-2 if x > img_w or x < 0 else int(x) for x in line_equation(H_SAMPLES)]
                    min_y = points[i][0][1]
                    max_y = points[i][-1][1]
                    min_y_idx = min_y//H_SAMPLES_STEP
                    max_y_idx = max_y//H_SAMPLES_STEP
                    for i in range(min_y_idx):
                        line_xs[i] = -2 
                    for i in range(max_y_idx, len(line_xs)):
                        line_xs[i] = -2
                    entry['lanes'].append(line_xs)
                
                reedit = False
                if len(entry['lanes']) > 0:
                    split = np.random.choice(['greenhouse_train', 'greenhouse_val', 'greenhouse_test'], p=[0.8, 0.1, 0.1])
                    split_path = os.path.join(os.path.dirname(image_path), split, os.path.basename(image_path))
                    label_path = os.path.join(os.path.dirname(image_path), split, 'labels.json')
                    entry['raw_file'] = os.path.basename(split_path)
                    cv2.imwrite(split_path, org_image)

                    while True:
                        saved_image = cv2.imread(split_path) 
                        ys = list(range(0, 720, 10))
                        h_samples = entry['h_samples']
                        lanes = entry['lanes']

                        for lane in lanes:
                            lane_points = []
                            for i, x in enumerate(lane):
                                if x == -2:
                                    continue
                                lane_points.append((x, ys[i]))
                            
                            for i, point in enumerate(lane_points):
                                if i > 0:
                                    cv2.line(saved_image, lane_points[i-1], point, (0, 255, 0), 2)
                                else:
                                    cv2.circle(saved_image, point, 2, (0, 255, 0), -1)

                        cv2.imshow('window', saved_image)
                        key = cv2.waitKey(1)

                        if key == ord('s'):
                            with open(label_path, 'a') as f:
                                json.dump(entry, f)
                                f.write('\n')
                            
                            os.remove(image_path)
                            break
                        elif key == ord('q'):
                            os.remove(split_path)
                            break
                        elif key == ord('b'):
                            reedit = True
                            break
                    
                if reedit: 
                    continue
                else: 
                    break