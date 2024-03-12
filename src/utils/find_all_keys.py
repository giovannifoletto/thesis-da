import json
import argparse
from Parser import Parser
from pathlib import Path
import os

def find_all_fields_possible(files, outfile):
    log_fields = set()
    log_keys = set()
    print(f"File {files}")
    with open(files) as filej:
        read_json = json.load(filej)
        for i in read_json["Records"]:
            #cl = ClusterLog(i)
            cl = Parser(i)
            cl.parse_log()
            for i in cl.keys:
                log_keys.add(i)
            for i in cl.element:
                log_fields.add(i)

    if outfile == None:
        print(f"=========== LOG FIELDS ============")
        print(log_fields)
        print(f"=========== LOG KEYS FILES ============")
        print(log_keys)
    else:
        print("Output-ting to file")
        with open(outfile, 'w', encoding='utf-8') as ofile:
            obj = {"log_fields": log_fields, "log_keys": log_keys}
            json.dumps(obj, ofile)

def read_multiple(infiles):
    print(f"=========== READING ALL FILES ============")
    for cloud_file in infiles:
        find_all_fields_possible(cloud_file)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="This program allows to easily find out all the possible keys in a given json file."
        )
    parser.add_argument("--infile", required=True)
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--multiple", choices=["true", "false"], required=False, default=False)
    
    args = parser.parse_args()
    ofile = Path(args.outfile) if args.outfile != "stdout" else "stdout"
    ifile = Path(args.infile)

    print(f"Selected: \n\tINFILE: {ifile}\n\tOUTFILE: {ofile}\n")

    if args.multiple:
        print("currently not supported")

    if ofile == "stdout":
        find_all_fields_possible(files=ifile, outfile=None)
    else:
        # Check is there is a directory, if there is not, it create it

        if not ofile.exists() and not ofile.is_dir():
            print("Creating path")
            os.mkdir(ofile)
        else:
            print("Path already exists")
        
        find_all_fields_possible(files=ifile, outfile=ofile)

