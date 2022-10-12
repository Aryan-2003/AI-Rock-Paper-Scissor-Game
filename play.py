from keras.models import load_model
import cv2
import numpy as np
from random import choice
import os

CLASS_MAP = {
    0: "None",
    1: "Paper",
    2: "Rock",
    3: "Scissor"
}


def mapper(val):
    return CLASS_MAP[val]


def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "Rock":
        if move2 == "Scissor":
            return "User"
        if move2 == "Paper":
            return "Computer"

    if move1 == "Paper":
        if move2 == "Rock":
            return "User"
        if move2 == "Scissor":
            return "Computer"

    if move1 == "Scissor":
        if move2 == "Paper":
            return "User"
        if move2 == "Rock":
            return "Computer"


model = load_model(os.path.join('models','rpsgame.h5'))

cap = cv2.VideoCapture(0)

prev_move = None

while True:
    ret, frame = cap.read()
    print(frame.shape)
    if ret:
        frame = cv2.resize(frame,(1450,900),fx=0,fy=0,interpolation=cv2.INTER_CUBIC)
    else:
        continue

    # rectangle for user to play
    cv2.rectangle(frame, (50, 100), (450, 500), (255, 0, 0), 2)
    # rectangle for computer to play
    cv2.rectangle(frame, (850, 100), (1250, 500), (255, 0, 0), 2)

    # extract the region of image within the user rectangle
    roi = frame[100:500, 50:450]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))

    # predict the move made
    pred = model.predict(np.expand_dims(img/255,0))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)

    # predict the winner (human vs computer)
    if prev_move != user_move_name:
        if user_move_name != "None":
            computer_move_name = choice(['Paper', 'Rock', 'Scissor'])
            winner = calculate_winner(user_move_name, computer_move_name)
        else:
            computer_move_name = "None"
            winner = "Waiting..."
    prev_move = user_move_name

    # display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your Move: " + user_move_name,
                (50, 50), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Computer's Move: " + computer_move_name,
                (750, 50), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner,
                (400, 600), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

    if computer_move_name != "None":
        icon = cv2.imread("images/{}.png".format(computer_move_name))
        icon = cv2.resize(icon, (400, 400))
        print(icon.shape)
        frame[100:500, 850:1250] = icon

    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()