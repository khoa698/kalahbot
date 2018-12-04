import communication.communicate
def main():
    while(True):
        print(communication.communicate.get_action())
        communication.communicate.send_action("MOVE;1")

if __name__ == "__main__":
    main()
