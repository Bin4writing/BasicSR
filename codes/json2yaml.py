import yaml
import json
import sys
import os

def main():
    address = "/home/super/BasicSR/codes/options/test"
    address.replace('\\','/')
    list = os.listdir(address)
    #print(list)
    for i in range(0,len(list)):
        path = os.path.join(address,list[i])
        path = path.replace('\\','/')
        if os.path.isfile(path) and "json" in path:
            load_lines = ""
            load_dict = {}
            with open(path,'r') as load_f:
                for line in load_f:
                    unComment = line.split('//')
                    load_lines += (unComment[0])
                    if(len(unComment)>1):
                        load_lines+='\n'
                #print(load_lines)
                load_dict = json.loads(load_lines)
            
            new_path = path.split(".json")[0]+".yaml"
            with open(new_path,'w') as dump_f:
                yaml.dump(load_dict,dump_f)


if __name__ == '__main__':
    main()
