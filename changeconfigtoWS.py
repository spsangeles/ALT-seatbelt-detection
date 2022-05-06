a_file = open("core/config.py", "r")
list_of_lines = a_file.readlines()
list_of_lines[13] = '__C.YOLO.CLASSES              = "./data/classes/obj-ws.names"\n'

a_file = open("core/config.py", "w")
a_file.writelines(list_of_lines)
a_file.close()
