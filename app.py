import os
from flask import Flask, request, render_template, flash
from landmarks.landmarks import getcoordinates
from morphing.morphing import *
import random
from app_config_utils import afterresponse
from app_config_utils.appconfig import SAMPLE, UPLOADED, ALPHA_BLENDING
from eye_glasses_detector.eyeglass_detector import find_glass
from face_detection.face_crop import face_crop
import time

app = Flask("after_response")
app.secret_key = 'my unobvious secret key'

afterresponse.AfterResponse(app)

@app.after_response
def delete():
    time.sleep(30)
    filelist = [f for f in os.listdir("static/") if f.endswith(".jpg") and f != SAMPLE]
    for f in filelist:
        os.remove(os.path.join("static/", f))

@app.route('/')
def frontpage():
    return render_template('fileform.html')


@app.route('/success')
def success(old_image, new_image):
    return render_template('show.html', old_image=old_image, new_image=new_image)


@app.route("/handleupload", methods=['POST'])
def handleupload():
    print(request.files)
    if 'photo' in request.files:
        photo = request.files['photo']
        if photo.filename != '':
            rechecked_filename = f"{random.getrandbits(128)}.jpg"
            photo.save(os.path.join('static', rechecked_filename))

            landmarks_img2 = json.loads(getcoordinates(os.path.join('static', SAMPLE)))
            landmarks_img1 = json.loads(getcoordinates(os.path.join('static', rechecked_filename)))  # Function call to get the landmarks which returns the JSON

            nose = landmarks_img1['coordinates'][30]
            print(nose[0], nose[1])
            face_crop(os.path.join('static', rechecked_filename), rechecked_filename, nose)
            # reduce uploaded image to area around face.

            img2 = cv2.imread(os.path.join('static', SAMPLE))
            img1 = cv2.imread(os.path.join('static', UPLOADED))
            og = cv2.imread(os.path.join('static', UPLOADED))


            landmarks_img1 = landmarks_img1['coordinates']
            landmarks_img2 = landmarks_img2['coordinates']

            #draw facial landmark points on the image
            for p in landmarks_img1:
                p = np.array(p, dtype=np.float32)
                draw_point(img1, tuple(p) , (0,0,255))

            cv2.imwrite(os.path.join('static', rechecked_filename), img1)
            # Convert Mat to float data type
            img1 = np.float32(img1)
            img2 = np.float32(img2)

            alpha = 0.25
            z = 0

            points = []

            for i in range(0, len(landmarks_img1)):
                a = (1 - z) * landmarks_img1[i][0] + z * landmarks_img2[i][0]
                b = (1 - z) * landmarks_img1[i][1] + z * landmarks_img2[i][1]
                points.append((a, b))

            triangle_coordinates = get_triangle_vertices(img1, points)

            imgMorph = np.zeros(img1.shape, dtype=img1.dtype)

            for coo in triangle_coordinates:
                x = coo[0]
                y = coo[1]
                z = coo[2]

                # get original coordinates of each traingle in both the images and morphed image
                t1 = [landmarks_img1[x], landmarks_img1[y], landmarks_img1[z]]
                t2 = [landmarks_img2[x], landmarks_img2[y], landmarks_img2[z]]
                t = [points[x], points[y], points[z]]

                morphTriangle(img1, img2, imgMorph, t1, t2, t, ALPHA_BLENDING)

            splitface_img = splitface_morph(img1, imgMorph)

            act_hash = random.getrandbits(128)
            new_filename = os.path.join('static', f"{act_hash}.jpg")
            print(new_filename)
            cv2.imwrite(new_filename, splitface_img)
            # cv2.imshow(np.uint8(imgMorph))
            # cv2.waitKey(0)
            old_image = rechecked_filename
            new_image = os.path.basename(new_filename)
            if find_glass(og) is True:
                flash("Disclaimer: This image probably has glasses or other artefacts. Results might be affected.")

    return render_template('show.html', old_image=old_image, new_image=new_image)



if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
