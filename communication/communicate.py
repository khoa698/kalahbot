from communication.netcat import Netcat

nc = Netcat('127.0.0.1', 12345)

def send_action(action: str):
    nc.write(action + "\n")

def get_action()->str:
    msg = nc.read_until()
    return msg

def set_move_action(hole: int)->str:
    return "MOVE;" + str(hole)

def set_swap_action()->str:
    return "SWAP"

def is_starting(msg: str)-> bool:
    position = msg[6:-1]
    if position == "NORTH":
        return False
    elif position == "SOUTH":
        return True
    else:
        raise Exception("Illegal assigning of starting states")

def get_turn(msg: str)->bool:
    split_msg = msg.split(";", 4)
    if len(split_msg) != 4:
        raise Exception("Invalid state")

    if split_msg[1] == "SWAP":
        return True
    elif split_msg[1] == "MOVE":
        if split_msg[3].find("YOU") != -1:
            return False
        elif split_msg[3].find("OPP") != -1:
            return False
        elif split_msg[3].find("END") != -1:
            return False
        else:
            raise Exception("Illegal state change")
    else:
        raise Exception("Illegal state change")

    return True

def main():
    while(True):
        print(get_action())
        send_action("MOVE;1")

if __name__ == "__main__":
    main()








