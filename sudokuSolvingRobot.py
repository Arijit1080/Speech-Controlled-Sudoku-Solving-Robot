import cv2
import numpy as np
import pytesseract
import re
import pygame
import numpy
import time
import os
import threading
import speech_recognition as sr
import RPi.GPIO as GPIO
from imutils.video import VideoStream

def intro():
    global talking, face, solve_sudoku, capture
    while True:
        while talking:
            print("Talking")
            os.system(
                "espeak 'Hello I am Sudo. A sudoku solving robot, made by Sparklers. Just show me a sudoku, I will solve it for you.'")
            talking = False
            face = True


def speak(s):
    command = "espeak '" + s + "'"
    os.system(command)


def faceAnimation(display_surface):
    global face, talking
    image = pygame.image.load('face1.png')
    image2 = pygame.image.load('face2.png')

    while face or talking:
        if face:
            display_surface.blit(image, (0, 0))
            pygame.display.update()
        elif talking:
            display_surface.blit(image, (0, 0))
            pygame.display.update()
            time.sleep(0.5)
            display_surface.blit(image2, (0, 0))
            pygame.display.update()
            time.sleep(0.5)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()


def focusGrid(ogimg):
    rx = 500.0 / ogimg.shape[0]
    ry = 500.0 / ogimg.shape[1]
    r = max([rx, ry])
    ogimg = cv2.resize(ogimg, (0, 0), fx=r, fy=r)
    img = cv2.cvtColor(ogimg, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255,
                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 25, 25)
    blur = cv2.GaussianBlur(img, (3, 3), 3)
    edged = cv2.Canny(blur, 100, 180)
    contours, hierarchy = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)#[-2:]

    if len(contours) == 0:
        print("No contours found")
        return None
    cnt = None
    maxArea = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > maxArea:
            maxArea = area
            cnt = c

    if cnt is None:
        print("No biggest contour")
        return None
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if (approx.size != 8):
        print("Wrong shape of grid")
        return None
    approx = approx.reshape(4, 2)
    approx = rearrangeCorners(approx, ogimg.shape[0],
                              ogimg.shape[1])
    approx = np.array(approx.tolist(), np.float32)

    gridSize = cellSize * 9
    final = np.array([
        [0, 0],
        [0, gridSize],
        [gridSize, gridSize],
        [gridSize, 0]], dtype="float32")

    M = cv2.getPerspectiveTransform(approx, final)
    fixed = cv2.warpPerspective(img, M,
                                (gridSize, gridSize))

    return fixed


def distSquared(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def rearrangeCorners(corners, width, height):
    corners = sorted(corners,
                     key=lambda p: distSquared(p, [0, 0]))
    tl = corners[0]
    corners = sorted(corners[1:],
                     key=lambda p: distSquared(p,
                                               [0, height]))
    bl = corners[0]
    corners = sorted(corners[1:],
                     key=lambda p: distSquared(p, [width,
                                                   height]))
    return np.array([tl, bl] + corners)


def splitUp(grid):
    cells = []
    for i in range(0, 9):
        row = []
        for j in range(0, 9):
            cropped = grid[
                      cellSize * i + border:cellSize * (
                                  i + 1) - border,
                      cellSize * j + border:cellSize * (
                                  j + 1) - border]
            row.append(cropped)
        cells.append(row)
    return cells


def highlightDigit(cell):
    if cell is None:
        return None
    img = cv2.cvtColor(cell, cv2.COLOR_GRAY2RGB)
    gray = cv2.bitwise_not(cell)
    output = cv2.connectedComponentsWithStats(gray, 8,
                                              cv2.CV_32S)
    stats = output[2]
    if len(output[2]) <= 1:
        return None
    largest_label = 1 + np.argmax(output[2][1:, -1])
    width, height = gray.shape[:2]
    x, y, w, h, _ = stats[largest_label]
    bX = x + w / 2.0
    bY = y + h / 2.0

    cX = width / 2.0
    cY = height / 2.0

    tX = cX - bX
    tY = cY - bY

    if (abs(tX) + abs(tY) > 10) or (
            w * h > 0.5 * width * height):
        return None
    img = img[y:y + h, x:x + w]
    return img


def highlightCells(cells):
    for i in range(len(cells)):
        for j in range(len(cells[i])):
            cells[i][j] = highlightDigit(cells[i][j])
    return cells


def thresh(x):
    if x < 2:
        return 0
    if x < 5:
        return 0
    return 1


def addPadding(img, border=10):
    return cv2.copyMakeBorder(img, 10, 10, 2, 2,
                              cv2.BORDER_CONSTANT,
                              value=[255, 255, 255])


def hconcat_resize_min(im_list,
                       interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list = [addPadding(im) for im in im_list]
    im_list_resize = [cv2.resize(im, (
    int(im.shape[1] * h_min / im.shape[0]), h_min),
                                 interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def flatten(a):
    temp = []
    for row in a:
        temp = temp + row
    return temp


def getDigits(cells):
    line = flatten(cells)
    cellsWithDigits = list(
        filter(lambda x: x is not None, line))
    line = hconcat_resize_min(cellsWithDigits)
    custom_config = r'--psm 6 outputbase digits'
    text = pytesseract.image_to_string(line,
                                       config=custom_config)
    if len(text) == 0:
        return None
    text = text.partition('\n')
    if len(text) == 0:
        return None
    text = "".join(re.findall('\d+', text[0]))
    if len(text) != len(
            cellsWithDigits) or not text.isdigit():
        return None
    print(text)
    grid = []
    c = 0
    for i in range(0, len(cells)):
        row = []
        for j in range(0, len(cells[i])):
            if cells[i][j] is not None:
                row.append(int(text[c]))
                c += 1
            else:
                row.append(0)
        grid.append(row)

    return grid


def extractGrid(img):
    if img is None:
        print("No such image found")
        return None

    clean = focusGrid(img)
    if clean is None:
        print("Failed")
        return None

    cells = splitUp(clean)
    cells = highlightCells(cells)
    grid = getDigits(cells)
    if grid is None:
        print("Unable to read numbers")
        return None
    return grid


def draw_box():
    for i in range(2):
        pygame.draw.line(screen, (255, 0, 0),
                         (padding + x * dif - 3,
                          (y + i) * dif),
                         (padding + x * dif + dif + 3,
                          (y + i) * dif),
                         4)
        pygame.draw.line(screen, (255, 0, 0),
                         (padding + (x + i) * dif, y * dif),
                         (padding + (x + i) * dif,
                          y * dif + dif), 4)


def draw(grid):
    for i in range(9):
        for j in range(9):
            if grid[i][j] != 0:
                pygame.draw.rect(screen, (101, 152, 224), (
                    padding + i * dif, j * dif, dif + 1,
                    dif + 1))
                text1 = font1.render(str(grid[i][j]), 1,
                                     (255, 255, 255))
                screen.blit(text1,
                            (padding + i * dif + 15,
                             j * dif + 10))
    for i in range(10):
        if i % 3 == 0:
            thick = 7
        else:
            thick = 1
        pygame.draw.line(screen, (255, 255, 255),
                         (padding, i * dif),
                         (padding + 320, i * dif), thick)
        pygame.draw.line(screen, (255, 255, 255),
                         (i * dif + padding, 0),
                         (i * dif + padding, 500), thick)


def draw_val(val):
    text1 = font1.render(str(val), 1, (255, 255, 255))
    screen.blit(text1, (x * dif + 15, y * dif + 15))


def valid(m, i, j, val):
    for it in range(9):
        if m[i][it] == val:
            return False
        if m[it][j] == val:
            return False
    it = i // 3
    jt = j // 3
    for i in range(it * 3, it * 3 + 3):
        for j in range(jt * 3, jt * 3 + 3):
            if m[i][j] == val:
                return False
    return True


def solve(grid, i, j):
    while grid[i][j] != 0:
        if i < 8:
            i += 1
        elif i == 8 and j < 8:
            i = 0
            j += 1
        elif i == 8 and j == 8:
            return True
    pygame.event.pump()
    for it in range(1, 10):
        if valid(grid, i, j, it) == True:
            grid[i][j] = it
            global x, y
            x = i
            y = j
            screen.fill((75, 75, 75))
            draw(grid)
            draw_box()
            pygame.display.update()
            pygame.time.delay(20)
            if solve(grid, i, j) == 1:
                return True
            else:
                grid[i][j] = 0
            screen.fill(((75, 75, 75)))

            draw(grid)
            draw_box()
            pygame.display.update()
            pygame.time.delay(50)
    return False


def show_puzzle(grid):
    screen.fill(((75, 75, 75)))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return
    draw(grid)
    pygame.display.update()

def sudoku_solve():
    global solve_sudoku, show_solution, capture
    val = 0
    vs = VideoStream(usePiCamera=True, resolution=(1280,720)).start()
    time.sleep(1.0)
    img_name = "temp.png"
    while True:
        while solve_sudoku:

            show_solution = True
            initial_frame = vs.read()
            up_points = (screen_size_x, 360)
            frame = cv2.resize(initial_frame, up_points,
                               interpolation=cv2.INTER_LINEAR)
            cv2.normalize(frame, frame, 0, 255,
                          cv2.NORM_MINMAX)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = numpy.rot90(frame, 3)
            frame = numpy.fliplr(frame)
            frame = pygame.surfarray.make_surface(frame)
            screen.blit(frame, (0, 0))
            pygame.display.update()
            if capture:
                capture = False
                cv2.imwrite(img_name, initial_frame)
                time.sleep(2)
                print("{} written!".format(img_name))
                try:
                    grid = extractGrid(cv2.imread(img_name))
                    if grid == None:
                        print("No Sudoku found")
                        no_suduko_message_thread = threading.Thread(
                            target=speak, args=(
                                "I have not found any sudoku in the image",))
                        no_suduko_message_thread.start()
                        continue
                except:
                    print("Error")
                    continue

                grid = [list(i) for i in zip(*grid)]
                show_puzzle(grid)
                recognised_message_thread = threading.Thread(
                    target=speak, args=(
                        "I have recognised the sudoku, and now I am solving it.",))
                recognised_message_thread.start()
                time.sleep(1)

                run = True
                show = True
                flag1 = 0

                while run:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            run = False

                    if solve(grid, 0, 0):
                        run = False
                    if val != 0:
                        draw_val(val)
                        if valid(grid, int(x), int(y),
                                 val) == True:
                            grid[int(x)][int(y)] = val
                            flag1 = 0
                        else:
                            grid[int(x)][int(y)] = 0
                        val = 0

                    draw(grid)
                    if flag1 == 1:
                        draw_box()
                    pygame.display.update()
                    time.sleep(1)
                    solved_message_thread = threading.Thread(
                        target=speak,
                        args=("I have solved the sudoku",))
                    solved_message_thread.start()

                    while show_solution:
                        time.sleep(1)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()

    pygame.quit()

if __name__ == "__main__":
    pos_x = 0
    pos_y = -1
    os.environ['SDL_VIDEO_WINDOW_POS'] = '%i,%i' % (
        pos_x, pos_y)
    os.environ['SDL_VIDEO_CENTERED'] = '0'

    screen_size_x = 500
    screen_size_y = 320
    cellSize = 56
    border = 3
    padding = (screen_size_x - screen_size_y) / 2
    x = 0
    y = 0
    dif = screen_size_y / 9

    talking = False
    solve_sudoku = False
    capture = False
    face = True
    show_solution = False

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode(
        (screen_size_x, screen_size_y), pygame.NOFRAME)
    font1 = pygame.font.SysFont("comicsans", 25)

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(40, GPIO.OUT, initial=GPIO.LOW)

    intro_thread = threading.Thread(target=intro, args=())
    intro_thread.start()
    face_thread = threading.Thread(target=faceAnimation,
                                   args=(screen,))
    face_thread.start()

    sudoku_solve_thread = threading.Thread(target=sudoku_solve, args=())
    sudoku_solve_thread.start()
    
    sample_rate = 48000
    chunk_size = 2048
    r = sr.Recognizer()

    with sr.Microphone(device_index=2,
                       sample_rate=sample_rate,
                       chunk_size=chunk_size) as source:
        while True:
            r.adjust_for_ambient_noise(source)
            print("Say Something")
            GPIO.output(40, GPIO.HIGH)
            audio = r.listen(source)

            try:
                GPIO.output(40, GPIO.LOW)
                text = r.recognize_google(audio)
                print("you said: " + text)
                if any(x in text for x in ["intro"]):
                    talking = True
                    face = False
                elif any(x in text for x in
                         ["start", "sudoku", "solving"]):
                    solve_sudoku = True
                    face = False
                elif any(x in text for x in ["capture"]):
                    capture = True
                elif any(x in text for x in ["stop"]):
                    solve_sudoku = False
                    face = True
                elif any(x in text for x in ["thank"]):
                    show_solution = False
                elif any(x in text for x in ["exit"]):
                    exit()
                elif any(x in text for x in ["sleep"]):
                    os.system("sudo poweroff")

            except sr.UnknownValueError:
                print(
                    "Google Speech Recognition could not understand audio")

            except sr.RequestError as e:
                print("error")




