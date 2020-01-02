#!/usr/bin/python

import cv2
import numpy as np
import random
import json


def read_points(path):
    points = []
    # Read points
    with open(path) as file:
        for line in file:
            x, y = line.split()
            points.append((int(x), int(y)))

    return points


def draw_point(img, p, color):
    cv2.circle(img, p, 2, color, cv2.FILLED, cv2.LINE_AA, 0)


def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def draw_delaunay(img, subdiv, delaunay_color):
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)

    cv2.imshow(img)
    cv2.waitKey(0)


def get_triangle_vertices(img, points):
    size = img.shape
    r = (0, 0, size[1], size[0])

    subdiv = cv2.Subdiv2D(r)
    for p in points:
        subdiv.insert(p)

    triangleList = subdiv.getTriangleList()

    delaunayTri = []

    for t in triangleList:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)

            if len(ind) == 3:
                delaunayTri.append([ind[0], ind[1], ind[2]])

    return delaunayTri


def delauney_triangulation(img, points):
    # Define window names
    win_delaunay = "Delaunay Triangulation"

    # Turn on animation while drawing triangles
    animate = True

    # Define colors for drawing.
    delaunay_color = (255, 255, 255)
    points_color = (0, 0, 255)

    size = img.shape
    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)

    draw_points(img, p, color)
    draw_delaunay(img, subdiv, delaunay_color)


# get average of landmark points on both the images
def get_points_average(points1, points2):
    avg_points = []
    for i in range(len(points)):
        x = (points1[i][0] + points2[i][0]) / 2
        y = (points1[i][1] + points2[i][1]) / 2
        avg_points.append((x, y))
    return avg_points


def morphTriangle(img1, img2, img, t1, t2, t, alpha):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask


def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


def read_json(filename):
    with open(filename) as json_data:
        data = json.load(json_data)
    return data['coordinates']


def read_json(filename):
    with open(filename) as json_data:
        data = json.load(json_data)
    return data['coordinates']


def detect_face(img1):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray = np.array(gray, dtype='uint8')
    # Detect faces
    return face_cascade.detectMultiScale(gray, 1.1, 4)


def splitface_morph(img1, imgMorph):
    faces = detect_face(img1)
    for (x, y, w, h) in faces:
        w=int(w/2)
    img1_half = img1[0:img1.shape[0], 0:x+w]
    img2_half = imgMorph[0:imgMorph.shape[0], x+w:imgMorph.shape[1]]
    h1, w1 = img1_half.shape[:2]
    h2, w2 = img2_half.shape[:2]

    vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)

    vis[:h1, :w1,:3] = img1_half
    vis[:h2, w1:w1+w2,:3] = img2_half

    return vis
