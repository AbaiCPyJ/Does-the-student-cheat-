from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import numpy as np
import math

def distance(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)



def show_webcam(mirror=False):
    print("[INFO] загрузка модели, ориентир лица...")
    detector = dlib.get_frontal_face_detector()
    # https://www.ibug.doc.ic.ac.uk/resources/300-W/ это датасет на котором была обучена модель
    predictor = dlib.shape_predictor(MODEL) # сама модель
    # https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/ вот хорошая статья про лицевые ориентиры
    # инициализировать видеопоток и немного поспать, что позволяет прогреть камеру
    print("[INFO] прогрев датчика камеры...")

    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # 400x225 to 1024x576
    frame_width = 1024
    frame_height = 576


    # цикл по кадрам из видеопотока
    # тоски 2D изображения. Если вы меняете изображение, вам нужно изменить вектор
    image_points = np.array([
        (360, 390),  # Кончик носа 34
        (400, 560),  # Подбородок 9
        (337, 300),  # Левый глаз Левый угол 37
        (513, 301),  # Правый глаз правый угол 46
        (345, 465),  # Левый угол рта 49
        (453, 469)  # Правый угол рта 55
    ], dtype="double")

    # точки 3D модели
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Кончик носа 34
        (0.0, -330.0, -65.0),  # Подбородок 9
        (-225.0, 170.0, -135.0),  # Левый глаз Левый угол 37
        (225.0, 170.0, -135.0),  # Правый глаз правый угол 46
        (-150.0, -150.0, -125.0),  # Левый угол рта 49
        (150.0, -150.0, -125.0)  # Правый угол рта 55

    ])

    while True:
        # захватить кадр из потокового видео, изменить его размер
        # иметь максимальную ширину 400 пикселей и преобразовать его в чернобелый
        frame = vs.read()
        if mirror:
            frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=1024, height=576)


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        size = gray.shape

        # обнаружить лица в кадре серого
        rects = detector(gray, 0)

        # проверьте, было ли обнаружено лицо, и если да, нарисуйте
        # количество граней на раме
        # if len(rects) > 0:
        #     text = "{} face(s) found".format(len(rects))
        #     cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # цикл для лиц
        for rect in rects:
            # вычислите ограничивающую рамку лица и нарисуйте ее на на фрейме
            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            #cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
            # определить лицевые ориентиры для области лица, затем
            # преобразовать координаты лицевой точки (x, y) в массив NumPy
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # цикл по (x, y) -координатам для лицевых ориентиров
            # и нарисуем каждый из них
            for (i, (x, y)) in enumerate(shape):
                if i == 33:
                    # кое-что для наших ключевых ориентиров
                    # сохранить в нашем новом списке ключевых точек
                    # то есть ключевые точки = [(i, (x, y))]
                    image_points[0] = np.array([x, y], dtype='double')
                    # написать на рамке зеленым
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 8:
                    image_points[1] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 36:#point 37
                    image_points[2] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 45:#point 46
                    image_points[3] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 48:
                    image_points[4] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 54:
                    image_points[5] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                else:
                    # для всех других ориентиров
                    # нарисовать на фрейме красным
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                    #cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            focal_length = size[1]
            center = (size[1] / 2, size[0] / 2)
            camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                                     dtype="double")


            dist_coeffs = np.zeros((4, 1))  # Предполагается что объектив не искаженен
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                          dist_coeffs,
                                                                          flags=cv2.SOLVEPNP_ITERATIVE)


            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                             translation_vector, camera_matrix, dist_coeffs)
            for p in image_points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            p3 = (int(image_points[1][0]), int(image_points[1][1]))
            p4 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            p5 = (int(image_points[2][0]), int(image_points[2][1]))
            p6 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            p7 = (int(image_points[3][0]), int(image_points[3][1]))
            p8 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            p9 = (int(image_points[4][0]), int(image_points[4][1]))
            p10 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            p11 = (int(image_points[5][0]), int(image_points[5][1]))
            p12 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            ratio = distance(shape[36], shape[45]) / distance(p1, p2)
            if ratio <= 0.4:
                text = "Don't cheat!"
                cv2.putText(frame, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)

            #print("Расстояние между глазами:" , distance(shape[36], shape[45]))
            #print("Растояние между точками p1 and p2", distance(p1, p2))
            #print("otnoshenie:", distance(shape[36], shape[45]) / distance(p1, p2))
            cv2.line(frame, p1, p2, (0, 180, 0), 2)
            cv2.line(frame, p3, p4, (0, 180, 0), 2)
            cv2.line(frame, p5, p6, (0, 180, 0), 2)
            cv2.line(frame, p7, p8, (0, 180, 0), 2)
            cv2.line(frame, p9, p10, (0, 180, 0), 2)
            cv2.line(frame, p11, p12, (0, 180, 0), 2)
            #print(p1, p2)


        # показать на фрейме
        cv2.imshow("Result", frame)


        key = cv2.waitKey(1) & 0xFF

        # нажать `q` чтобы выйти из цикла
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()
    #out.release()


def diplom_project():
    show_webcam(mirror=True)



if __name__ == '__main__':
    MODEL = 'shape_predictor_68_face_landmarks.dat'
    diplom_project()
