import os
import numpy as np
def board(arr):
    print("     |     |      ")
    print("  {0}  |  {1}  |  {2}  ".format(arr[1], arr[2], arr[3]))
    print("_____|_____|_____ ")
    print("     |     |      ")
    print("  {0}  |  {1}  |  {2}".format(arr[4], arr[5], arr[6]))
    print("_____|_____|_____ ")
    print("     |     |      ")
    print("  {0}  |  {1}  |  {2}".format(arr[7], arr[8], arr[9]))
    print("     |     |      ")

def check_win(arr):
    #check win
    for i in (1,4,7):
        if arr[i] == arr[i+1] and arr[i] == arr[i+2]:
            return 1
    for j in (1,2,3):
        if arr[j] == arr[j+3] and arr[j] == arr[j+6]:
            return 1
    if arr[1] == arr[5] and arr[1] == arr[9]:
        return 1
    elif arr[3] == arr[5] and arr[3] == arr[7]:
        return 1
    #check draw
    if arr[1] != '1' and arr[2] != '2' and arr[3] != '3' \
        and arr[4] != '4' and arr[5] != '5' and arr[6] != '6' \
            and arr[7] != '7' and arr[8] != '8' and arr[9] != '9':
            return -1
    return 0
def play():
    arr = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9' ]
    player = 0 #player 1 goes first
    flag = 0 #1:someone has won; -1: Draw; 0: still running  
    while flag == 0:
        os.system('clear')
        print("Player1:X and Player2:O\n")
        if player % 2 == 0:
            print("Player X Chance:\n")
            board(arr)
            choice = int(input())
        else:
            print("Player O Chance:\n")
            board(arr)
            #choice = int(input())
            #waiting_choices = np.argsort(get_result(arr)) #ask computer for next step
            test = [0.9, 0.8, 1, 0, 0, 0, 0, 0, 0.1]
            waiting_choices = list(np.argsort(test))  
            choice =  waiting_choices.pop()+1
            while arr[choice] == 'X' or arr[choice] == 'O':
                choice = waiting_choices.pop()+1
           
        if 1 <= choice <= 9 and arr[choice] != 'X' and arr[choice] != 'O':
            if player%2 == 0:
                arr[choice] = 'X'
            else:
                arr[choice] = 'O'
            player += 1
        else:
            print("Error, board is loading again.....\n")
            continue
        flag = check_win(arr)
    os.system('clear')
    board(arr)
    if flag == 1:
        if player%2 == 0:
            print("Player O has won!\n")
        else:
            print("Player X has won!\n")

    else:
        print('Draw\n')


play()
    

        
        




        


 

     
