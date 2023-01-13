##
##
##

def get_action_id(file: str) -> int:
    pos = file.find('A')
    return int(file[(pos + 1):(pos + 4)])


def get_setup_id(file: str) -> int:
    pos = file.find('S')
    return int(file[(pos + 1):(pos + 4)])


def get_subject_id(file: str) -> int:
    pos = file.find('P')
    return int(file[(pos + 1):(pos + 4)])


def get_camera_id(file: str) -> int:
    pos = file.find('C')
    return int(file[(pos + 1):(pos + 4)])
